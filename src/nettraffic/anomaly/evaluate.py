"""Detection evaluation against labelled anomalies (point- and event-level)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nettraffic.metrics import detection_summary


def label_mask(
    labels: pd.DataFrame, cells: np.ndarray, index: pd.DatetimeIndex, metrics: tuple[str, ...]
) -> np.ndarray:
    mask = np.zeros((len(cells), len(index), len(metrics)), dtype=bool)
    pos = {int(c): i for i, c in enumerate(cells)}
    for r in labels.itertuples():
        if r.cell_id not in pos or r.metric not in metrics:
            continue
        t = (index >= r.start) & (index < r.end)
        mask[pos[r.cell_id], t, metrics.index(r.metric)] = True
    return mask


def evaluate_detection(
    points: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    threshold: float,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    scheduled: np.ndarray | None = None,
) -> dict:
    lab = labels[(labels["end"] > window_start) & (labels["start"] < window_end)]
    pts = points[points["observed"].notna()].copy()
    key = pd.MultiIndex.from_frame(pts[["cell_id", "ts", "metric"]])
    hours = [
        (r.cell_id, ts, r.metric)
        for r in lab.itertuples()
        for ts in pd.date_range(r.start, r.end, freq="1h", inclusive="left")
    ]
    y_true = (
        key.isin(pd.MultiIndex.from_tuples(hours, names=["cell_id", "ts", "metric"]))
        if hours
        else np.zeros(len(pts), bool)
    )

    out: dict = {"point_level": {}, "event_level": {}, "by_type": {}}
    for col in [c for c in pts.columns if c.startswith("score")]:
        name = "ensemble" if col == "score" else col.removeprefix("score_")
        out["point_level"][name] = detection_summary(y_true, pts[col].to_numpy(), threshold)

    # ---- event level: a labelled anomaly is caught if any alert on (cell, metric) overlaps it
    detected, ttd = [], []
    for r in lab.itertuples():
        hit = events[
            (events["cell_id"] == r.cell_id)
            & (events["metric"] == r.metric)
            & (events["start"] < r.end)
            & (events["end"] > r.start)
        ]
        detected.append(not hit.empty)
        if not hit.empty:
            ttd.append(max(0.0, (hit["start"].min() - r.start) / pd.Timedelta(hours=1)))
    lab = lab.assign(detected=detected)

    tp_alerts = 0
    for e in events.itertuples():
        m = lab[
            (lab["cell_id"] == e.cell_id)
            & (lab["metric"] == e.metric)
            & (lab["start"] < e.end)
            & (lab["end"] > e.start)
        ]
        tp_alerts += int(not m.empty)
    n_alerts = len(events)
    precision = tp_alerts / n_alerts if n_alerts else float("nan")
    recall = float(np.mean(detected)) if detected else float("nan")
    days = max((window_end - window_start) / pd.Timedelta(days=1), 1e-9)
    out["event_level"] = {
        "labelled_events": len(lab),
        "alerts": n_alerts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if n_alerts and detected and (precision + recall)
        else 0.0,
        "mean_time_to_detect_h": float(np.mean(ttd)) if ttd else float("nan"),
        "alerts_per_day": n_alerts / days,
    }
    out["by_type"] = lab.groupby("anomaly_type")["detected"].mean().round(3).to_dict()
    if scheduled is not None:
        out["event_level"]["alerts_during_scheduled_events"] = int(scheduled)
    return out
