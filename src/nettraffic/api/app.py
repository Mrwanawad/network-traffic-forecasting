"""FastAPI service for on-demand queries over precomputed batch outputs.

The batch job is the source of truth; the API only serves its artifacts. That
keeps p99 latency in the milliseconds, makes the service stateless (scale by
replicas), and avoids model inference on the request path.

    uvicorn nettraffic.api.app:app --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

Metric = Literal["sms_in", "sms_out", "call_in", "call_out", "internet"]
Severity = Literal["low", "medium", "high", "critical"]
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# --------------------------------------------------------------------------- schemas
class ForecastPoint(BaseModel):
    ts: datetime
    lead_hours: int
    yhat: float
    lo: float
    hi: float


class ForecastResponse(BaseModel):
    cell_id: int
    metric: Metric
    interval_level: float
    points: list[ForecastPoint]


class Anomaly(BaseModel):
    event_id: int
    incident_id: int
    cell_id: int
    metric: Metric
    start: datetime
    end: datetime
    peak_ts: datetime
    observed: float
    expected: float
    deviation_pct: float
    score: float = Field(description="ensemble surprisal, -log10 tail probability")
    direction: Literal["drop", "spike"]
    severity: Severity
    n_hours: int


class AnomalyResponse(BaseModel):
    count: int
    items: list[Anomaly]


class CapacityItem(BaseModel):
    cell_id: int
    metric: str
    peak_ts: datetime
    peak_forecast: float
    max_upper: float
    capacity: float
    utilisation: float
    utilisation_upper: float
    status: Literal["over_capacity", "at_risk", "watch", "ok"]
    capacity_source: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    artifacts_dir: str
    loaded: dict[str, int]
    forecast_origin: datetime | None


# --------------------------------------------------------------------------- store
class ArtifactStore:
    """Loads batch outputs once; `reload()` picks up a new batch without restart."""

    FILES: ClassVar[dict[str, str]] = {
        "forecasts": "forecasts.parquet",
        "anomalies": "anomalies.parquet",
        "capacity": "capacity.parquet",
    }

    def __init__(self, root: Path, interval_level: float = 0.9) -> None:
        self.root = root
        self.interval_level = interval_level
        self.frames: dict[str, pd.DataFrame] = {}

    def reload(self) -> None:
        frames = {}
        for key, name in self.FILES.items():
            path = self.root / name
            frames[key] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if not frames["forecasts"].empty:
            frames["forecasts"] = frames["forecasts"].set_index(["cell_id", "metric"]).sort_index()
        self.frames = frames

    def get(self, key: str) -> pd.DataFrame:
        return self.frames.get(key, pd.DataFrame())


def _artifacts_dir() -> Path:
    return Path(os.environ.get("NETTRAFFIC_ARTIFACTS", "artifacts"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ArtifactStore(_artifacts_dir())
    store.reload()
    app.state.store = store
    yield


app = FastAPI(
    title="Network Traffic Forecasting API",
    version="0.1.0",
    description="Per-cell traffic forecasts with prediction intervals, ranked anomalies, capacity flags.",
    lifespan=lifespan,
)


def get_store(request: Request) -> ArtifactStore:
    return request.app.state.store


StoreDep = Annotated[ArtifactStore, Depends(get_store)]


# --------------------------------------------------------------------------- routes
@app.get("/health", response_model=Health)
def health(store: StoreDep) -> Health:
    fc = store.get("forecasts")
    loaded = {k: len(v) for k, v in store.frames.items()}
    origin = None
    if not fc.empty:
        origin = (fc["ts"].min() - pd.Timedelta(hours=1)).to_pydatetime()
    return Health(
        status="ok" if loaded.get("forecasts") else "degraded",
        artifacts_dir=str(store.root),
        loaded=loaded,
        forecast_origin=origin,
    )


@app.post("/admin/reload", response_model=Health)
def reload(store: StoreDep) -> Health:
    store.reload()
    return health(store)


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    store: StoreDep,
    cell_id: Annotated[int, Query(ge=1)],
    metric: Metric = "internet",
    horizon: Annotated[int, Query(ge=1, le=168, description="hours ahead")] = 24,
) -> ForecastResponse:
    fc = store.get("forecasts")
    if fc.empty:
        raise HTTPException(503, "forecasts not available - run the batch pipeline")
    try:
        rows = fc.loc[(cell_id, metric)]
    except KeyError:
        raise HTTPException(404, f"no forecast for cell {cell_id} / {metric}") from None
    rows = rows[rows["lead"] <= horizon].sort_values("lead")
    return ForecastResponse(
        cell_id=cell_id,
        metric=metric,
        interval_level=store.interval_level,
        points=[
            ForecastPoint(ts=r.ts.to_pydatetime(), lead_hours=int(r.lead), yhat=r.yhat, lo=r.lo, hi=r.hi)
            for r in rows.itertuples()
        ],
    )


@app.get("/anomalies", response_model=AnomalyResponse)
def anomalies(
    store: StoreDep,
    cell_id: int | None = None,
    metric: Metric | None = None,
    min_severity: Severity = "low",
    since: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> AnomalyResponse:
    ev = store.get("anomalies")
    if ev.empty:
        return AnomalyResponse(count=0, items=[])
    mask = ev["severity"].map(SEVERITY_RANK) >= SEVERITY_RANK[min_severity]
    if cell_id is not None:
        mask &= ev["cell_id"] == cell_id
    if metric is not None:
        mask &= ev["metric"] == metric
    if since is not None:
        ts = pd.Timestamp(since)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        mask &= ev["end"] >= ts
    sel = ev[mask].head(limit)  # already ranked by severity, then score
    items = [
        Anomaly.model_validate(r)
        for r in sel.drop(columns=["incident_metrics"], errors="ignore").to_dict("records")
    ]
    return AnomalyResponse(count=int(mask.sum()), items=items)


@app.get("/capacity", response_model=list[CapacityItem])
def capacity(
    store: StoreDep,
    status: Literal["over_capacity", "at_risk", "watch", "ok"] | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 100,
) -> list[CapacityItem]:
    cap = store.get("capacity")
    if cap.empty:
        return []
    if status:
        cap = cap[cap["status"] == status]
    return [CapacityItem.model_validate(r) for r in cap.head(limit).to_dict("records")]
