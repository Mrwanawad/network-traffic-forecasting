"""Combine detector scores, flag points, deduplicate into ranked alert events.

Every detector emits *surprisal* = -log10(tail probability under its own null
distribution): 2 means "1-in-100 under normal behaviour", 3 means 1-in-1000.
A weighted mean of surprisals is unit-consistent, unlike averaging raw scores
from models with different scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


@dataclass
class EnsembleOutput:
    points: pd.DataFrame  # one row per (cell, ts, metric) in the detection window
    events: pd.DataFrame  # deduplicated alerts


def combine(scores: dict[str, np.ndarray | None], weights: dict[str, float]) -> np.ndarray:
    num, den = None, 0.0
    for name, s in scores.items():
        if s is None:
            continue
        w = weights.get(name, 0.0)
        term = w * np.nan_to_num(s, nan=0.0)
        num = term if num is None else num + term
        den += w
    if num is None or den == 0:
        raise ValueError("no detector produced scores")
    return num / den


def build_points(
    cells: np.ndarray,
    index: pd.DatetimeIndex,
    metrics: tuple[str, ...],
    observed: np.ndarray,
    expected: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    combined: np.ndarray,
    detector_scores: dict[str, np.ndarray | None],
) -> pd.DataFrame:
    c, w, m = observed.shape
    df = pd.DataFrame(
        {
            "cell_id": np.repeat(cells, w * m),
            "ts": np.tile(np.repeat(index, m), c),
            "metric": np.tile(np.array(metrics), c * w),
            "observed": observed.reshape(-1),
            "expected": expected.reshape(-1),
            "lo": lo.reshape(-1),
            "hi": hi.reshape(-1),
            "score": combined.reshape(-1),
        }
    )
    for name, s in detector_scores.items():
        if s is not None:
            df[f"score_{name}"] = s.reshape(-1)
    return df


def build_events(
    points: pd.DataFrame,
    threshold: float,
    merge_gap_hours: int,
    cell_meta: pd.DataFrame | None = None,
    busy_quantile: float = 0.8,
) -> pd.DataFrame:
    """Merge consecutive flagged hours per (cell, metric) into events, then rank."""
    flagged = points[(points["score"] >= threshold) & points["observed"].notna()].sort_values(
        ["cell_id", "metric", "ts"]
    )
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "incident_id",
                "cell_id",
                "metric",
                "start",
                "end",
                "peak_ts",
                "observed",
                "expected",
                "deviation_pct",
                "score",
                "direction",
                "severity",
                "n_hours",
            ]
        )
    gap = pd.Timedelta(hours=merge_gap_hours + 1)
    new = (
        (flagged["cell_id"] != flagged["cell_id"].shift())
        | (flagged["metric"] != flagged["metric"].shift())
        | (flagged["ts"] - flagged["ts"].shift() > gap)
    )
    flagged = flagged.assign(event_key=new.cumsum())
    rows = []
    for _, g in flagged.groupby("event_key", sort=False):
        peak = g.loc[g["score"].idxmax()]
        dev = (peak["observed"] - peak["expected"]) / max(peak["expected"], 1e-9)
        rows.append(
            {
                "cell_id": int(peak["cell_id"]),
                "metric": peak["metric"],
                "start": g["ts"].min(),
                "end": g["ts"].max() + pd.Timedelta(hours=1),
                "peak_ts": peak["ts"],
                "observed": float(peak["observed"]),
                "expected": float(peak["expected"]),
                "deviation_pct": float(100 * dev),
                "score": float(peak["score"]),
                "direction": "drop" if peak["observed"] < peak["expected"] else "spike",
                "n_hours": len(g),
            }
        )
    ev = pd.DataFrame(rows).sort_values(["cell_id", "start"]).reset_index(drop=True)

    # incident = overlapping events on the same cell (one outage -> one incident, many metrics)
    inc, cur_end, cur_cell, inc_id = [], None, None, 0
    for r in ev.itertuples():
        if r.cell_id != cur_cell or cur_end is None or r.start > cur_end:
            inc_id += 1
            cur_cell, cur_end = r.cell_id, r.end
        else:
            cur_end = max(cur_end, r.end)
        inc.append(inc_id)
    ev["incident_id"] = inc
    ev["incident_metrics"] = ev.groupby("incident_id")["metric"].transform("nunique")

    busy: set[int] = set()
    if cell_meta is not None and "base_volume" in cell_meta:
        thr = cell_meta["base_volume"].quantile(busy_quantile)
        busy = set(cell_meta.loc[cell_meta["base_volume"] >= thr, "cell_id"].astype(int))
    ev["severity"] = [_severity(r, busy) for r in ev.itertuples()]
    ev["severity_rank"] = ev["severity"].map({s: i for i, s in enumerate(SEVERITY_ORDER)})
    ev = ev.sort_values(["severity_rank", "score"], ascending=[False, False]).reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))
    return ev.drop(columns="severity_rank")


def _severity(r, busy: set[int]) -> str:
    level = 2 if r.score >= 4 else 1 if r.score >= 3 else 0
    if r.direction == "drop" and r.deviation_pct <= -80 and r.incident_metrics >= 3:
        return "critical"  # looks like a site outage
    if r.direction == "drop" and r.cell_id in busy:
        level += 1  # customers notice drops on busy cells first
    return SEVERITY_ORDER[min(level, 2)]
