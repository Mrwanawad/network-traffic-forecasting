"""Rolling-origin backtesting, out-of-sample interval calibration, and reporting.

Fold layout (H = horizon, S = fold step)::

    fold 0 (calibration only)  |---train---|cut0|--H--|
    fold 1                      |----train----|cut1|--H--|
    ...
    fold n                      |-------train-------|cutN|--H--| <- data end

Fold k's prediction intervals are calibrated on residuals of folds < k only, so
reported coverage is genuinely out-of-sample.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nettraffic.config import PipelineConfig
from nettraffic.features import FeatureContext
from nettraffic.forecasting.base import ConformalCalibrator, Forecaster, lead_bucket
from nettraffic.metrics import forecast_summary, seasonal_naive_scale
from nettraffic.panel import Panel

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    calibrators: dict[tuple[str, str], ConformalCalibrator] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def fold_cutoffs(n_times: int, cfg: PipelineConfig) -> list[int]:
    last = n_times - cfg.horizon
    cuts = [last - i * cfg.fold_step_hours for i in range(cfg.n_backtest_folds + 1)][::-1]
    if cuts[0] < 3 * 168:
        raise ValueError("not enough history for the requested folds (need >= 3 weeks before fold 0)")
    return cuts


def subset_context(ctx: FeatureContext, pos: np.ndarray) -> FeatureContext:
    return FeatureContext(ctx.index, ctx.cal, ctx.events[pos], ctx.area[pos])


def run_backtest(
    panel: Panel,
    ctx: FeatureContext,
    cfg: PipelineConfig,
    global_models: list[Forecaster],
    series_models: list[Forecaster] | None = None,
    sample_cells: np.ndarray | None = None,
    metrics: tuple[str, ...] | None = None,
    series_metrics: tuple[str, ...] | None = None,
) -> BacktestResult:
    metrics = metrics or cfg.metrics
    cutoffs = fold_cutoffs(panel.n_times, cfg)
    frames: list[pd.DataFrame] = []
    timings: dict[str, float] = {}

    plan: list[tuple[Forecaster, Panel, FeatureContext, tuple[str, ...]]] = [
        (m, panel, ctx, metrics) for m in global_models
    ]
    if series_models and sample_cells is not None and len(sample_cells):
        pos = np.array([panel.cell_pos(int(c)) for c in sample_cells])
        sub, sub_ctx = panel.subset_cells(sample_cells), subset_context(ctx, pos)
        plan += [(m, sub, sub_ctx, series_metrics or metrics) for m in series_models]

    for model, pnl, cx, mets in plan:
        t0 = time.perf_counter()
        for metric in mets:
            y_full = pnl.metric(metric).astype(np.float64)
            for fold, cut in enumerate(cutoffs):
                yhat = model.forecast(pnl, cx, cut, metric, cfg.horizon)
                scale = seasonal_naive_scale(y_full[:, :cut], cfg.season_length)
                scale = np.where(np.isnan(scale), 1.0, scale)
                frames.append(_to_frame(model.name, metric, fold, cut, pnl, y_full, yhat, scale, cfg))
        timings[model.name] = round(time.perf_counter() - t0, 2)
        log.info("backtest %-16s done in %.1fs", model.name, timings[model.name])

    preds = pd.concat(frames, ignore_index=True)
    preds, calibrators = add_intervals(preds, cfg)
    return BacktestResult(preds, calibrators, timings)


def _to_frame(
    model: str,
    metric: str,
    fold: int,
    cut: int,
    pnl: Panel,
    y: np.ndarray,
    yhat: np.ndarray,
    scale: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    c, h = yhat.shape
    leads = np.tile(np.arange(1, h + 1), c)
    times = cut + leads - 1
    return pd.DataFrame(
        {
            "model": model,
            "metric": metric,
            "fold": fold,
            "cutoff": pnl.index[cut],
            "cell_id": np.repeat(pnl.cells, h),
            "lead": leads,
            "bucket": lead_bucket(leads, cfg.horizon_groups),
            "ts": pnl.index[times],
            "y": y[np.repeat(np.arange(c), h), times],
            "yhat": yhat.reshape(-1),
            "scale": np.repeat(scale, h),
        }
    )


def add_intervals(
    preds: pd.DataFrame, cfg: PipelineConfig
) -> tuple[pd.DataFrame, dict[tuple[str, str], ConformalCalibrator]]:
    preds = preds.copy()
    preds["lo"], preds["hi"] = np.nan, np.nan
    final: dict[tuple[str, str], ConformalCalibrator] = {}
    for (model, metric), g in preds.groupby(["model", "metric"], sort=False):
        for fold in sorted(g["fold"].unique()):
            if fold == 0:
                continue
            past = g[g["fold"] < fold]
            cal = ConformalCalibrator(cfg.interval_level).fit(
                past["bucket"].to_numpy(),
                past["y"].to_numpy(),
                past["yhat"].to_numpy(),
                past["scale"].to_numpy(),
            )
            idx = g.index[g["fold"] == fold]
            lo, hi = cal.interval(
                preds.loc[idx, "bucket"].to_numpy(),
                preds.loc[idx, "yhat"].to_numpy(),
                preds.loc[idx, "scale"].to_numpy(),
            )
            preds.loc[idx, "lo"], preds.loc[idx, "hi"] = lo, hi
        final[(str(model), str(metric))] = ConformalCalibrator(cfg.interval_level).fit(
            g["bucket"].to_numpy(), g["y"].to_numpy(), g["yhat"].to_numpy(), g["scale"].to_numpy()
        )
    return preds, final


# --------------------------------------------------------------------------- reporting
def _summ(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k, strict=True)), **forecast_summary(g)})
    return pd.DataFrame(rows)


def summarize(
    preds: pd.DataFrame, cfg: PipelineConfig, cell_volume: pd.Series, offday_ts: set
) -> dict[str, pd.DataFrame]:
    """Evaluation tables (fold 0 excluded: it only calibrates intervals)."""
    ev = preds[preds["fold"] > 0].copy()
    # point metrics at the canonical horizons (1h, 24h, 7d) + bucketed + overall
    at_h = ev[ev["lead"].isin(cfg.report_horizons)].assign(horizon=lambda d: d["lead"].astype(str) + "h")

    q_hi, q_lo = cell_volume.quantile(0.8), cell_volume.quantile(0.5)
    tier = np.where(cell_volume >= q_hi, "busy", np.where(cell_volume <= q_lo, "quiet", "mid"))
    ev["cell_tier"] = ev["cell_id"].map(dict(zip(cell_volume.index, tier, strict=True)))
    ev["day_type"] = np.where(ev["ts"].isin(offday_ts), "offday", "weekday")

    full_models = ev.groupby("model")["cell_id"].nunique()
    population = full_models[full_models == full_models.max()].index
    sample_cells = ev.loc[~ev["model"].isin(population), "cell_id"].unique()
    sample = ev[ev["cell_id"].isin(sample_cells)] if len(sample_cells) else ev.iloc[0:0]
    sample = sample[sample["metric"].isin(sample.loc[~sample["model"].isin(population), "metric"].unique())]

    return {
        "by_horizon": _summ(at_h[at_h["model"].isin(population)], ["model", "metric", "horizon"]),
        "by_bucket": _summ(ev[ev["model"].isin(population)], ["model", "metric", "bucket"]),
        "overall": _summ(ev[ev["model"].isin(population)], ["model"]),
        "sample_comparison": _summ(sample, ["model", "bucket"]) if len(sample) else pd.DataFrame(),
        "slices": _summ(ev[ev["model"].isin(population)], ["model", "metric", "cell_tier", "day_type"]),
    }
