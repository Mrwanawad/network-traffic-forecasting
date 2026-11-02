"""Realistic synthetic telecom traffic in the Telecom Italia Milan schema.

Why synthetic: the real Milan dataset is not reachable from every environment,
and it has no anomaly labels. The generator reproduces the properties that
matter for modelling and *injects labelled anomalies* so detection quality can
be measured, not guessed.

What it simulates
-----------------
* 10-minute grid, UTC storage, local-time (Europe/Rome) behaviour.
* Area types with distinct daily/weekly shapes (business, residential,
  university, stadium, rural) and per-cell heterogeneity.
* Italian public holidays, a slow internet growth trend.
* Zero-inflated quiet cells (Poisson counts on tiny means).
* Scheduled events (matches) -> a *feature*, not an anomaly.
* Labelled anomalies: outage, partial outage, congestion spike, unscheduled
  crowd event.
* Data errors (NOT anomalies): missing rows, long gaps, duplicate rows,
  negative values, post-gap catch-up dumps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nettraffic.calendar import ScheduledEvent, holiday_dates
from nettraffic.config import METRICS, PipelineConfig

AREA_TYPES = ("business", "residential", "university", "stadium", "rural")
BASE_VOLUME = {"business": 30.0, "residential": 20.0, "university": 22.0, "stadium": 8.0, "rural": 1.2}
METRIC_SCALE = {"sms_in": 0.7, "sms_out": 0.6, "call_in": 1.0, "call_out": 0.95, "internet": 25.0}

# Milan grid bounding box (approx.) - 100 x 100 squares in the original data
LAT_MIN, LAT_MAX = 45.358, 45.567
LON_MIN, LON_MAX = 9.011, 9.312
GRID = 100


@dataclass
class SyntheticDataset:
    raw: pd.DataFrame  # long 10-min rows: cell_id, ts, metrics (dirty)
    cells: pd.DataFrame  # cell metadata
    labels: pd.DataFrame  # cell_id, metric, start, end, anomaly_type (UTC, end exclusive)
    events: list[ScheduledEvent]
    error_log: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- shapes
def _bump(h: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    d = np.minimum(np.abs(h - mu), 24 - np.abs(h - mu))  # circular distance in hours
    return np.exp(-0.5 * (d / sigma) ** 2)


def _daily_shape(area: str, h: np.ndarray, offday: bool) -> np.ndarray:
    b = _bump
    if area == "business":
        return 0.12 + (0.35 * b(h, 14, 4) if offday else 1.0 * b(h, 10.5, 2.3) + 0.9 * b(h, 15.5, 2.3))
    if area == "residential":
        if offday:
            return 0.2 + 0.8 * b(h, 12, 3.5) + 0.9 * b(h, 20, 3)
        return 0.2 + 0.5 * b(h, 8, 1.5) + 0.6 * b(h, 13, 3) + 1.0 * b(h, 20.5, 2.5)
    if area == "university":
        return 0.08 + (0.15 * b(h, 15, 4) if offday else 1.1 * b(h, 11, 2) + 0.8 * b(h, 15, 2))
    if area == "stadium":
        return 0.1 + 0.3 * b(h, 14, 4) * (1.1 if offday else 1.0)
    # rural / park
    return 0.1 + (0.6 * b(h, 14, 4) if offday else 0.4 * b(h, 12, 4) + 0.3 * b(h, 18, 3))


def _metric_shape(metric: str, base: np.ndarray, h: np.ndarray) -> np.ndarray:
    if metric == "internet":
        return 0.35 + 0.8 * base  # always-on background traffic
    if metric.startswith("sms"):
        return base * (1.0 + 0.4 * _bump(h, 21, 2))
    return base


# --------------------------------------------------------------------------- cells
def make_cells(n_cells: int, rng: np.random.Generator, slots_per_hour: int = 6) -> pd.DataFrame:
    side = int(np.ceil(np.sqrt(n_cells)))
    r0 = c0 = (GRID - side) // 2
    rows, cols = np.divmod(np.arange(n_cells), side)
    rows, cols = rows + r0, cols + c0
    centre = (GRID - 1) / 2
    dist = np.hypot(rows - centre, cols - centre) / (side / 2)

    area = np.where(dist < 0.35, "business", np.where(dist < 0.85, "residential", "rural")).astype(object)
    candidates = np.flatnonzero(area == "residential")
    uni = rng.choice(candidates, size=max(1, n_cells // 60), replace=False)
    area[uni] = "university"
    stadium_pos = int(rng.choice(np.flatnonzero(area == "residential")))
    area[stadium_pos] = "stadium"

    base = np.array([BASE_VOLUME[a] for a in area]) * rng.lognormal(0, 0.45, n_cells)
    dlat, dlon = (LAT_MAX - LAT_MIN) / GRID, (LON_MAX - LON_MIN) / GRID
    # typical busy-hour internet volume at HOURLY resolution (the unit capacity is planned in)
    peak_internet = base * METRIC_SCALE["internet"] * 1.25 * slots_per_hour
    return pd.DataFrame(
        {
            "cell_id": (rows * GRID + cols + 1).astype(np.int64),  # Milan-style square id
            "row": rows,
            "col": cols,
            "lat": LAT_MIN + (rows + 0.5) * dlat,
            "lon": LON_MIN + (cols + 0.5) * dlon,
            "area_type": area.astype(str),
            "base_volume": base.round(3),
            # hourly internet capacity: a minority of cells is deliberately tight
            "capacity_internet": (peak_internet * rng.uniform(1.0, 2.4, n_cells)).round(1),
        }
    )


# --------------------------------------------------------------------------- generator
def generate(cfg: PipelineConfig) -> SyntheticDataset:
    rng = np.random.default_rng(cfg.seed)
    cells = make_cells(cfg.n_cells, rng, 60 // cfg.raw_step_minutes)
    n_cells = len(cells)
    step = pd.Timedelta(minutes=cfg.raw_step_minutes)
    start_local = pd.Timestamp(cfg.start_date, tz=cfg.timezone)
    n_steps = cfg.n_days * (24 * 60 // cfg.raw_step_minutes)
    index = pd.date_range(start_local.tz_convert("UTC"), periods=n_steps, freq=step)
    local = index.tz_convert(cfg.timezone)

    hours = (local.hour + local.minute / 60).to_numpy()
    local_date = local.normalize().tz_localize(None)
    hol = holiday_dates(sorted(set(local.year)), cfg.country_holidays)
    offday = (local.dayofweek.to_numpy() >= 5) | np.isin(local_date, list(hol))
    weeks = ((index - index[0]) / pd.Timedelta(days=7)).to_numpy()

    # festive SMS bursts (Christmas Eve / New Year's Eve evenings)
    md = local.strftime("%m-%d")
    festive = np.isin(md, ["12-24", "12-31"]) & (local.hour >= 19)

    values = np.zeros((n_cells, n_steps, len(METRICS)), dtype=np.float32)
    shift = rng.normal(0, 0.4, n_cells)  # per-cell phase heterogeneity (hours)
    off_ratio = rng.lognormal(0, 0.15, n_cells)
    for i, row in enumerate(cells.itertuples(index=False)):
        h = (hours + shift[i]) % 24
        wd = _daily_shape(row.area_type, h, offday=False)
        od = _daily_shape(row.area_type, h, offday=True) * off_ratio[i]
        base = np.where(offday, od, wd)
        for k, m in enumerate(METRICS):
            shape = _metric_shape(m, base, h)
            if m == "internet":
                shape = shape * (1 + 0.004 * weeks)
            if m.startswith("sms"):
                shape = shape * np.where(festive, 2.2, 1.0)
            values[i, :, k] = row.base_volume * METRIC_SCALE[m] * shape

    events = _scheduled_events(cells, index, cfg, rng)
    _apply_events(values, cells, index, events)

    noise = rng.lognormal(0, 0.12, values.shape).astype(np.float32)
    values = rng.poisson(values * noise).astype(np.float32)

    labels = _inject_anomalies(values, cells, index, cfg, rng)
    raw, error_log = _corrupt(values, cells, index, rng)
    return SyntheticDataset(raw=raw, cells=cells, labels=labels, events=events, error_log=error_log)


def _scheduled_events(
    cells: pd.DataFrame, index: pd.DatetimeIndex, cfg: PipelineConfig, rng: np.random.Generator
) -> list[ScheduledEvent]:
    st = cells.loc[cells["area_type"] == "stadium"].iloc[0]
    near = cells.loc[
        (cells["row"] - st["row"]).abs().le(1) & (cells["col"] - st["col"]).abs().le(1), "cell_id"
    ]
    ids = tuple(int(c) for c in near)
    local_days = pd.date_range(
        index[0].tz_convert(cfg.timezone).normalize(), index[-1].tz_convert(cfg.timezone), freq="D"
    )
    events: list[ScheduledEvent] = []
    for d in local_days:
        if d.dayofweek == 6 and d.isocalendar().week % 2 == 0:  # home match every other Sunday
            events.append(
                ScheduledEvent("home_match", (d + pd.Timedelta(hours=15)).tz_convert("UTC"), 3, ids)
            )
        elif d.dayofweek == 2 and rng.random() < 0.3:  # occasional midweek concert
            events.append(ScheduledEvent("concert", (d + pd.Timedelta(hours=20)).tz_convert("UTC"), 3, ids))
    return events


def _apply_events(
    values: np.ndarray, cells: pd.DataFrame, index: pd.DatetimeIndex, events: list[ScheduledEvent]
) -> None:
    pos = {int(c): i for i, c in enumerate(cells["cell_id"])}
    stadium = set(cells.loc[cells["area_type"] == "stadium", "cell_id"].astype(int))
    mult = np.array([3.5, 3.0, 4.0, 3.5, 2.5], dtype=np.float32)  # sms_in..internet
    for ev in events:
        t = (index >= ev.start) & (index < ev.start + pd.Timedelta(hours=ev.hours))
        for c in ev.cell_ids:
            f = mult if c in stadium else 1 + (mult - 1) * 0.3
            values[pos[c], t, :] *= f
            # stadium baseline is tiny: add an absolute crowd load as well
            if c in stadium:
                values[pos[c], t, :] += np.array([15, 12, 25, 22, 600], dtype=np.float32)


ANOMALY_TYPES = {
    # name: (metrics affected, multiplier range, duration hours range)
    "outage": (METRICS, (0.0, 0.05), (1, 6)),
    "partial_outage": (("internet",), (0.25, 0.5), (2, 8)),
    "voice_degradation": (("call_in", "call_out"), (0.2, 0.45), (2, 6)),
    "congestion_spike": (("internet",), (2.2, 3.5), (1, 4)),
    "unscheduled_event": (("sms_in", "sms_out", "call_in", "call_out"), (2.5, 4.0), (2, 4)),
}


def _inject_anomalies(
    values: np.ndarray,
    cells: pd.DataFrame,
    index: pd.DatetimeIndex,
    cfg: PipelineConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    steps_per_hour = 60 // cfg.raw_step_minutes
    n_hours = len(index) // steps_per_hour
    det_start_h = n_hours - cfg.detection_days * 24
    weight = np.sqrt(cells["base_volume"].to_numpy())
    weight = weight / weight.sum()
    occupied = np.zeros((len(cells), n_hours), dtype=bool)
    rows = []
    # a few in history (realistic contamination) + a denser set in the detection window
    plan = [
        (48, det_start_h, max(10, cfg.n_cells // 25)),
        (det_start_h, n_hours - 6, max(25, cfg.n_cells // 8)),
    ]
    types = list(ANOMALY_TYPES)
    for lo, hi, n in plan:
        placed = 0
        while placed < n:
            ci = int(rng.choice(len(cells), p=weight))
            kind = types[int(rng.integers(len(types)))]
            metrics, (mlo, mhi), (dlo, dhi) = ANOMALY_TYPES[kind]
            dur = int(rng.integers(dlo, dhi + 1))
            h0 = int(rng.integers(lo, hi - dur))
            if occupied[ci, max(0, h0 - 24) : h0 + dur + 24].any():
                continue
            # spikes are only meaningful when the cell is busy enough to be observed
            occupied[ci, h0 : h0 + dur] = True
            sl = slice(h0 * steps_per_hour, (h0 + dur) * steps_per_hour)
            mult = rng.uniform(mlo, mhi)
            for m in metrics:
                k = METRICS.index(m)
                values[ci, sl, k] = np.round(values[ci, sl, k] * mult)
                rows.append(
                    {
                        "cell_id": int(cells["cell_id"].iat[ci]),
                        "metric": m,
                        "start": index[h0 * steps_per_hour],
                        "end": index[h0 * steps_per_hour] + pd.Timedelta(hours=dur),
                        "anomaly_type": kind,
                    }
                )
            placed += 1
    return pd.DataFrame(rows)


def _corrupt(
    values: np.ndarray, cells: pd.DataFrame, index: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Inject data-quality errors and flatten to long raw rows."""
    n_cells, n_steps, _n_metrics = values.shape
    vals = values.copy()
    present = np.ones((n_cells, n_steps), dtype=bool)
    log: dict[str, int] = {}

    # 1) catch-up dumps: a short outage of the *feed*, next row carries the backlog
    n_dumps = max(20, n_cells // 10)
    for _ in range(n_dumps):
        ci, k = int(rng.integers(n_cells)), int(rng.integers(2, 5))
        t0 = int(rng.integers(10, n_steps - k - 2))
        vals[ci, t0 + k, :] += vals[ci, t0 : t0 + k, :].sum(axis=0)
        present[ci, t0 : t0 + k] = False
    log["catchup_dumps"] = n_dumps

    # 2) random missing rows (short gaps)
    drop = rng.random((n_cells, n_steps)) < 0.003
    present &= ~drop
    log["missing_rows_random"] = int(drop.sum())

    # 3) long gaps (probe down for hours)
    n_long = max(3, n_cells // 80)
    for _ in range(n_long):
        ci, k = int(rng.integers(n_cells)), int(rng.integers(36, 180))
        t0 = int(rng.integers(0, n_steps - k))
        present[ci, t0 : t0 + k] = False
    log["long_gaps"] = n_long

    # 4) negative values (counter/sensor glitch)
    neg = rng.random(vals.shape) < 0.0002
    vals[neg] = -np.abs(vals[neg]) - 1
    log["negative_values"] = int(neg.sum())

    ci, ti = np.nonzero(present)
    raw = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()[ci], "ts": index[ti]})
    for k, m in enumerate(METRICS):
        raw[m] = vals[ci, ti, k]

    # 5) duplicate rows (re-sent records)
    dup = raw.sample(frac=0.001, random_state=int(rng.integers(1 << 31)))
    log["duplicate_rows"] = len(dup)
    raw = pd.concat([raw, dup], ignore_index=True)
    return raw, log
