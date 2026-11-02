"""Export compact JSON for the static dashboard (site/data)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nettraffic.panel import Panel


def _clean(o: Any) -> Any:
    """JSON-safe conversion (NaN -> null, numpy -> python, timestamps -> ISO)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), separators=(",", ":")))


def frame_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].map(lambda t: t.isoformat() if pd.notna(t) else None)
    return out.to_dict(orient="records")


def _r(a: np.ndarray, nd: int = 1) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, dtype=float)]


def export_cell_series(
    out_dir: Path,
    panel: Panel,
    cell_ids: list[int],
    history_hours: int,
    points: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> None:
    hist_start = panel.n_times - history_hours
    idx = panel.index[hist_start:]
    sel = points[points["cell_id"].isin(cell_ids)]
    pts = {
        key: g.set_index("ts")[["expected", "lo", "hi", "score"]]
        for key, g in sel.groupby(["cell_id", "metric"])
    }
    for cid in cell_ids:
        c = panel.cell_pos(cid)
        doc: dict[str, Any] = {"cell_id": cid, "history_ts": [t.isoformat() for t in idx], "metrics": {}}
        fc_c = forecasts[forecasts["cell_id"] == cid]
        for k, m in enumerate(panel.metrics):
            p = pts[(cid, m)].reindex(idx) if (cid, m) in pts else None
            f = fc_c[fc_c["metric"] == m].sort_values("ts")
            doc["metrics"][m] = {
                "actual": _r(panel.values[c, hist_start:, k]),
                "expected": _r(p["expected"]) if p is not None else None,
                "lo": _r(p["lo"]) if p is not None else None,
                "hi": _r(p["hi"]) if p is not None else None,
                "score": _r(p["score"], 2) if p is not None else None,
                "forecast_ts": [t.isoformat() for t in f["ts"]],
                "forecast": _r(f["yhat"]),
                "forecast_lo": _r(f["lo"]),
                "forecast_hi": _r(f["hi"]),
            }
        write_json(out_dir / "series" / f"{cid}.json", doc)
