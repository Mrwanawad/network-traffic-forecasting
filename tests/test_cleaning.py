from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from nettraffic.cleaning import clean, fix_catchup_dumps, interpolate_short_gaps
from nettraffic.config import METRICS, PipelineConfig


def _raw(n_slots: int = 36, value: float = 10.0) -> pd.DataFrame:
    ts = pd.date_range("2024-03-01", periods=n_slots, freq="10min", tz="UTC")
    df = pd.DataFrame({"cell_id": 1, "ts": ts})
    for m in METRICS:
        df[m] = value
    return df


def test_interpolates_only_short_gaps() -> None:
    s = np.array([1, np.nan, np.nan, 4, 5, np.nan, np.nan, np.nan, np.nan, np.nan, 11], dtype=float)
    out, n = interpolate_short_gaps(s, max_len=3)
    assert np.allclose(out[:4], [1, 2, 3, 4])
    assert np.isnan(out[5:10]).all()  # long gap stays missing, never fabricated
    assert n == 2


def test_edge_gaps_are_not_extrapolated() -> None:
    out, n = interpolate_short_gaps(np.array([np.nan, 1.0, 2.0, np.nan]), max_len=3)
    assert np.isnan(out[0]) and np.isnan(out[-1]) and n == 0


def test_catchup_dump_is_redistributed() -> None:
    vals = np.full((20, 2), 10.0)
    present = np.ones(20, dtype=bool)
    present[8:11] = False  # 3-slot feed outage
    vals[11] = 40.0  # backlog of 3 slots + own slot
    fixed = fix_catchup_dumps(vals, present)
    assert fixed == 1
    assert np.allclose(vals[8:12], 10.0) and present[8:12].all()


def test_real_spike_without_gap_is_untouched() -> None:
    vals = np.full((20, 2), 10.0)
    vals[11] = 80.0
    present = np.ones(20, dtype=bool)
    assert fix_catchup_dumps(vals, present) == 0
    assert vals[11, 0] == 80.0


def test_clean_dedupes_negatives_and_aggregates() -> None:
    cfg = dataclasses.replace(PipelineConfig(), metrics=METRICS)
    raw = _raw()
    dup = raw.iloc[[3]].assign(internet=999.0)  # re-sent record supersedes
    raw.loc[5, "sms_in"] = -4  # impossible value -> null -> interpolated
    raw = pd.concat([raw, dup], ignore_index=True)
    panel, rep = clean(raw, cfg)
    assert rep.duplicate_rows == 1 and rep.negative_values == 1
    assert panel.n_times == 6
    internet = panel.metric("internet")[0]
    assert internet[0] == 10 * 5 + 999  # kept the last duplicate
    assert np.allclose(panel.metric("sms_in")[0], 60)


def test_hour_with_low_coverage_is_missing() -> None:
    cfg = PipelineConfig()
    raw = _raw(18)
    raw = raw.drop(index=range(6, 12))  # whole second hour missing
    panel, rep = clean(raw, cfg)
    assert np.isnan(panel.metric("internet")[0, 1])
    assert rep.missing_hours == len(METRICS) * 0 + 1


def test_synthetic_end_to_end_cleaning(panel, synth) -> None:
    assert panel.values.shape[1] == 42 * 24
    assert np.isnan(panel.values).mean() < 0.01
    assert (np.nan_to_num(panel.values) >= 0).all()
    assert set(synth.labels["anomaly_type"]) <= {
        "outage",
        "partial_outage",
        "voice_degradation",
        "congestion_spike",
        "unscheduled_event",
    }
