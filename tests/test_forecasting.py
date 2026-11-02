from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nettraffic.features import build_features, hour_of_week_profile
from nettraffic.forecasting.backtest import fold_cutoffs, run_backtest, summarize
from nettraffic.forecasting.base import ConformalCalibrator, lead_bucket
from nettraffic.forecasting.baselines import SeasonalNaive
from nettraffic.forecasting.lgbm import GlobalLGBM
from nettraffic.metrics import mae, seasonal_naive_scale, smape
from tests.conftest import make_panel


def test_lead_bucket_mapping() -> None:
    assert list(lead_bucket(np.array([1, 2, 24, 25, 168]), (1, 24, 168))) == [
        "1h",
        "24h",
        "24h",
        "168h",
        "168h",
    ]


def test_seasonal_naive_repeats_last_week() -> None:
    week = np.arange(168, dtype=float)
    p = make_panel(np.tile(week, 3)[None, :])
    fc = SeasonalNaive(168).forecast(p, None, cutoff=336, metric="internet", horizon=168)
    assert np.allclose(fc[0], week)


def test_seasonal_naive_falls_back_on_missing() -> None:
    vals = np.tile(np.arange(168, dtype=float), 3)
    vals[168:336] = np.nan
    p = make_panel(vals[None, :])
    fc = SeasonalNaive(168).forecast(p, None, cutoff=336, metric="internet", horizon=24)
    assert np.allclose(fc[0], np.arange(24))


def test_metrics_edge_cases() -> None:
    assert smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert mae(np.array([1.0, np.nan]), np.array([2.0, 5.0])) == 1.0
    scale = seasonal_naive_scale(np.tile([1.0, 3.0], 10)[None, :], 2)
    assert np.isnan(scale[0])  # perfectly periodic -> undefined scale, caller substitutes


def test_features_do_not_leak_future(panel, ctx) -> None:
    """Changing values at/after t-L+1 must not change features for target t."""
    logy = np.log1p(panel.metric("internet").astype(float))
    prof = hour_of_week_profile(logy, ctx.cal.hour_of_week, 600)
    for L in (1, 24, 168):
        for t in (600, 611, 623):
            times = np.array([t])
            base = build_features(logy, times, L, ctx, prof)
            tampered = logy.copy()
            tampered[:, t - L + 1 :] = 999.0  # everything the L-model must not see
            again = build_features(tampered, times, L, ctx, prof)
            pd.testing.assert_frame_equal(base, again)


def test_global_lgbm_ignores_post_cutoff_data(panel, ctx, small_cfg) -> None:
    cut = panel.n_times - 168
    a = GlobalLGBM(small_cfg).forecast(panel, ctx, cut, "internet", 168)
    tampered = dataclasses.replace(panel, values=panel.values.copy())
    tampered.values[:, cut:, :] = 1e6
    b = GlobalLGBM(small_cfg).forecast(tampered, ctx, cut, "internet", 168)
    assert np.allclose(a, b)


def test_conformal_interval_reaches_nominal_coverage() -> None:
    rng = np.random.default_rng(0)
    yhat = rng.uniform(10, 100, 20_000)
    y = yhat * (1 + rng.normal(0, 0.1, yhat.size))
    scale = np.full(yhat.size, 5.0)
    b = np.full(yhat.size, "24h")
    cal = ConformalCalibrator(0.9).fit(b[:10_000], y[:10_000], yhat[:10_000], scale[:10_000])
    lo, hi = cal.interval(b[10_000:], yhat[10_000:], scale[10_000:])
    cov = np.mean((y[10_000:] >= lo) & (y[10_000:] <= hi))
    assert 0.88 <= cov <= 0.92


def test_fold_cutoffs_are_ordered_and_disjoint(small_cfg) -> None:
    cuts = fold_cutoffs(42 * 24, small_cfg)
    assert cuts == sorted(cuts) and len(cuts) == small_cfg.n_backtest_folds + 1
    assert cuts[-1] + small_cfg.horizon == 42 * 24
    with pytest.raises(ValueError):
        fold_cutoffs(500, small_cfg)


@pytest.mark.slow
def test_backtest_global_model_beats_baseline(panel, ctx, small_cfg) -> None:
    res = run_backtest(
        panel, ctx, small_cfg, [SeasonalNaive(), GlobalLGBM(small_cfg)], metrics=("internet", "call_in")
    )
    vol = pd.Series(np.nanmean(panel.metric("internet"), axis=1), index=panel.cells)
    overall = summarize(res.predictions, small_cfg, vol, set()).get("overall").set_index("model")
    assert overall.loc["lightgbm_global", "mase"] < overall.loc["seasonal_naive", "mase"]
    assert 0.8 <= overall.loc["lightgbm_global", "coverage"] <= 0.97
    assert res.predictions.loc[res.predictions["fold"] == 0, "lo"].isna().all()  # calibration fold
