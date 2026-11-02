"""Validation & cleaning: raw dirty rows -> regular hourly Panel.

Rules (and why):

1. **Duplicates** -> keep the last received row per (cell, ts): re-sent records
   supersede earlier ones.
2. **Timestamps** -> truncated to the 10-min grid, stored in UTC. DST is handled
   by never storing local time; calendar features convert later.
3. **Negative counts** -> null. Traffic counts cannot be negative; this is a
   probe/counter glitch, never a real event.
4. **Catch-up dumps** -> when a row right after a feed gap carries the backlog
   (total activity ~ (gap+1) x normal), spread it evenly over the gap. Leaving it
   would create a fake spike *and* a fake drop -> two false alerts.
5. **Gaps** -> <= `max_interp_steps` slots: linear interpolation. Longer gaps stay
   missing; we never fabricate hours of traffic (that would hide outages and bias
   the models).
6. **Hourly aggregation** -> sum of slots, rescaled for partially observed hours,
   NaN when coverage < `min_hour_coverage`.

Real behaviour (outages, spikes, events) is deliberately left untouched: the
cleaner only removes things that are physically impossible or are transport
artefacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import polars as pl

from nettraffic.config import PipelineConfig
from nettraffic.panel import Panel


@dataclass
class CleaningReport:
    input_rows: int = 0
    duplicate_rows: int = 0
    negative_values: int = 0
    missing_slots: int = 0
    catchup_dumps_fixed: int = 0
    interpolated_slots: int = 0
    long_gap_slots: int = 0
    missing_hours: int = 0
    total_hours: int = 0

    def as_dict(self) -> dict[str, int | float]:
        d: dict[str, int | float] = asdict(self)
        d["missing_hour_rate"] = round(self.missing_hours / max(self.total_hours, 1), 5)
        return d


# --------------------------------------------------------------------------- helpers
def _gap_runs(missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start positions and lengths of True-runs in a 1-D bool array."""
    if not missing.any():
        return np.empty(0, int), np.empty(0, int)
    padded = np.concatenate([[False], missing, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return starts, ends - starts


def interpolate_short_gaps(series: np.ndarray, max_len: int) -> tuple[np.ndarray, int]:
    """Linear interpolation of interior NaN runs of length <= max_len (in-place copy)."""
    out = series.astype(np.float64, copy=True)
    starts, lengths = _gap_runs(np.isnan(out))
    filled = 0
    n = len(out)
    for s, ln in zip(starts, lengths, strict=True):
        e = s + ln
        if ln > max_len or s == 0 or e >= n:
            continue
        left, right = out[s - 1], out[e]
        out[s:e] = left + (right - left) * (np.arange(1, ln + 1) / (ln + 1))
        filled += ln
    return out, filled


def fix_catchup_dumps(values: np.ndarray, present: np.ndarray, max_gap: int = 6, context: int = 6) -> int:
    """Detect and redistribute post-gap backlog rows. Mutates `values` and `present`.

    values: (T, M) for one cell, present: (T,) rows that exist in the feed.
    """
    total = np.nansum(values, axis=1)
    starts, lengths = _gap_runs(~present)
    fixed = 0
    t_len = len(present)
    for s, k in zip(starts, lengths, strict=True):
        t = s + k  # first row after the gap
        if k > max_gap or t >= t_len or s == 0:
            continue
        lo, hi = max(0, s - context), min(t_len, t + 1 + context)
        neigh = np.r_[total[lo:s][present[lo:s]], total[t + 1 : hi][present[t + 1 : hi]]]
        if neigh.size < 3:
            continue
        ref = float(np.median(neigh))
        ratio = total[t] / max(ref, 1.0)
        if ratio >= max(2.5, 0.6 * (k + 1)):
            values[s : t + 1, :] = values[t, :] / (k + 1)
            present[s : t + 1] = True
            fixed += 1
    return fixed


# --------------------------------------------------------------------------- main
def clean(raw: pl.DataFrame | pd.DataFrame, cfg: PipelineConfig) -> tuple[Panel, CleaningReport]:
    if isinstance(raw, pd.DataFrame):
        raw = pl.from_pandas(raw)
    rep = CleaningReport(input_rows=raw.height)
    step = f"{cfg.raw_step_minutes}m"
    metrics = cfg.metrics

    df = raw.with_columns(pl.col("ts").dt.convert_time_zone("UTC").dt.truncate(step))
    deduped = df.unique(subset=["cell_id", "ts"], keep="last", maintain_order=True)
    rep.duplicate_rows = df.height - deduped.height

    neg_exprs = [pl.when(pl.col(m) < 0).then(None).otherwise(pl.col(m)).alias(m) for m in metrics]
    rep.negative_values = int(sum(deduped.select((pl.col(m) < 0).sum()).item() for m in metrics))
    deduped = deduped.with_columns(neg_exprs)

    # ---- dense 10-min grid, aligned to whole hours
    cells = np.sort(deduped["cell_id"].unique().to_numpy()).astype(np.int64)
    ts_min = deduped["ts"].min()
    ts_max = deduped["ts"].max()
    start = pd.Timestamp(ts_min).floor("1h")
    end = pd.Timestamp(ts_max).floor("1h") + pd.Timedelta(hours=1)
    index = pd.date_range(start, end, freq=pd.Timedelta(minutes=cfg.raw_step_minutes), inclusive="left")
    steps_per_hour = 60 // cfg.raw_step_minutes

    vals = np.full((len(cells), len(index), len(metrics)), np.nan, dtype=np.float64)
    ci = np.searchsorted(cells, deduped["cell_id"].to_numpy())
    ti = index.get_indexer(pd.DatetimeIndex(deduped["ts"].to_numpy()).tz_localize("UTC"))
    for k, m in enumerate(metrics):
        vals[ci, ti, k] = deduped[m].to_numpy().astype(np.float64)
    present = np.zeros((len(cells), len(index)), dtype=bool)
    present[ci, ti] = True
    rep.missing_slots = int((~present).sum())

    # ---- per-cell sequence repairs
    for c in range(len(cells)):
        rep.catchup_dumps_fixed += fix_catchup_dumps(vals[c], present[c])
        vals[c, ~present[c], :] = np.nan
        for k in range(len(metrics)):
            filled, n = interpolate_short_gaps(vals[c, :, k], cfg.max_interp_steps)
            vals[c, :, k] = filled
            if k == 0:
                rep.interpolated_slots += int(n)
    rep.long_gap_slots = int(np.isnan(vals[:, :, 0]).sum())

    # ---- hourly aggregation with coverage rule
    n_hours = len(index) // steps_per_hour
    v = vals[:, : n_hours * steps_per_hour, :].reshape(len(cells), n_hours, steps_per_hour, len(metrics))
    n_obs = np.sum(~np.isnan(v), axis=2)
    hourly = np.nansum(v, axis=2) * steps_per_hour / np.maximum(n_obs, 1)
    hourly[n_obs < cfg.min_hour_coverage * steps_per_hour] = np.nan
    rep.missing_hours = int(np.isnan(hourly[:, :, 0]).sum())
    rep.total_hours = int(hourly[:, :, 0].size)

    panel = Panel(
        values=hourly.astype(np.float32),
        cells=cells,
        index=pd.date_range(start, periods=n_hours, freq="1h"),
        metrics=metrics,
    )
    return panel, rep
