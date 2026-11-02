from __future__ import annotations

import numpy as np
import pandas as pd

from nettraffic.anomaly.ensemble import build_events, combine
from nettraffic.anomaly.evaluate import evaluate_detection
from nettraffic.anomaly.tail import TailModel
from nettraffic.capacity import capacity_report


def test_tail_model_is_calibrated_and_monotone() -> None:
    rng = np.random.default_rng(1)
    null = rng.exponential(1.0, 50_000)
    tail = TailModel.fit(null)
    assert abs(tail.tail_prob(np.array([np.quantile(null, 0.5)]))[0] - 0.5) < 0.01
    x = np.linspace(0, 30, 200)
    s = tail.surprisal(x)
    assert np.all(np.diff(s) >= 0)
    assert s[-1] > np.log10(len(null))  # extrapolates beyond the sample size
    # exponential null: P(X >= 10) = e^-10 -> surprisal ~4.34
    assert abs(tail.surprisal(np.array([10.0]))[0] - 10 / np.log(10)) < 0.4


def test_combine_skips_missing_detectors() -> None:
    a = np.array([4.0, 0.0])
    out = combine({"residual": a, "lstm_ae": None}, {"residual": 0.5, "lstm_ae": 0.5})
    assert np.allclose(out, a)


def _points(scores: list[float], cell: int = 7, metric: str = "internet") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=len(scores), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "cell_id": cell,
            "ts": ts,
            "metric": metric,
            "observed": 1.0,
            "expected": 100.0,
            "lo": 50.0,
            "hi": 150.0,
            "score": scores,
        }
    )


def test_events_merge_small_gaps_and_split_large_ones() -> None:
    pts = _points([0, 5, 5, 0, 5, 0, 0, 0, 6, 0])
    ev = build_events(pts, threshold=4.5, merge_gap_hours=1)
    assert len(ev) == 2
    first = ev.sort_values("start").iloc[0]
    assert first["n_hours"] == 3 and first["direction"] == "drop"


def test_outage_on_many_metrics_is_critical_single_incident() -> None:
    pts = pd.concat([_points([0, 6, 6, 0], metric=m) for m in ("internet", "call_in", "sms_in")])
    ev = build_events(pts, threshold=4.5, merge_gap_hours=1)
    assert (ev["severity"] == "critical").all()
    assert ev["incident_id"].nunique() == 1


def test_event_level_evaluation() -> None:
    pts = _points([0, 6, 6, 0, 0, 0, 5, 0])
    ev = build_events(pts, 4.5, 1)
    start = pts["ts"].iloc[1]
    labels = pd.DataFrame(
        [
            {
                "cell_id": 7,
                "metric": "internet",
                "start": start,
                "end": start + pd.Timedelta(hours=2),
                "anomaly_type": "outage",
            }
        ]
    )
    res = evaluate_detection(pts, ev, labels, 4.5, pts["ts"].min(), pts["ts"].max() + pd.Timedelta(hours=1))
    assert res["event_level"]["recall"] == 1.0
    assert res["event_level"]["precision"] == 0.5  # one true, one false alert
    assert res["point_level"]["ensemble"]["recall"] == 1.0


def test_capacity_status_rules() -> None:
    ts = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    fc = pd.DataFrame(
        {
            "cell_id": np.repeat([1, 2, 3], 3),
            "metric": "internet",
            "ts": np.tile(ts, 3),
            "yhat": [50, 120, 60, 80, 90, 70, 10, 20, 30],
            "lo": 0.0,
            "hi": [60, 140, 70, 95, 105, 80, 15, 25, 35],
        }
    )
    meta = pd.DataFrame({"cell_id": [1, 2, 3], "capacity_internet": [100.0, 100.0, 100.0]})
    rep = capacity_report(fc, pd.DataFrame(), meta).set_index("cell_id")
    assert rep.loc[1, "status"] == "over_capacity"
    assert rep.loc[2, "status"] == "at_risk"
    assert rep.loc[3, "status"] == "ok"
