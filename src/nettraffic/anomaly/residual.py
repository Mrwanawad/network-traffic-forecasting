"""Residual-based detector: observed vs. model-expected traffic.

Expected value at time t comes from the frozen day-ahead (L=24) global model,
whose features use observations up to t-24. Its out-of-sample residual
distribution (from the backtest) is the null distribution, so the score is a
calibrated tail probability, not an arbitrary z-score.

Echo suppression (two passes): an outage at day d would otherwise poison the
lag features of day d+1 and create a fake "spike" alert 24h later. Pass 1
finds very strong deviations, replaces them with the expected value in a copy
of the series, and pass 2 recomputes expectations from that cleaned history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nettraffic.anomaly.tail import TailModel
from nettraffic.features import FeatureContext
from nettraffic.forecasting.base import ConformalCalibrator
from nettraffic.forecasting.lgbm import GlobalLGBM
from nettraffic.panel import Panel

DETECT_LAG = 24
DETECT_BUCKET = "24h"


@dataclass
class ResidualNull:
    """Per-cell standardisation + pooled tail model of out-of-sample residuals.

    Pooling raw residual ratios across cells lets noisy quiet cells dominate the
    tail, so a 3x spike on a busy cell looks ordinary. Instead each cell gets a
    robust scale (1.4826 x MAD of its own backtest residual ratios), and the tail
    model is fitted on the pooled *standardised* residuals.
    """

    cell_scale: pd.Series  # index cell_id -> robust scale of the signed ratio
    tail: TailModel

    @classmethod
    def fit(cls, preds: pd.DataFrame, bucket: str = DETECT_BUCKET) -> ResidualNull:
        g = preds[(preds["bucket"] == bucket) & preds["y"].notna()]
        r = (g["y"] - g["yhat"]) / (g["yhat"] + g["scale"])
        df = pd.DataFrame({"cell_id": g["cell_id"].to_numpy(), "r": r.to_numpy()})
        med = df.groupby("cell_id")["r"].transform("median")
        mad = (df["r"] - med).abs().groupby(df["cell_id"]).median() * 1.4826
        floor = float(np.nanquantile(mad, 0.05)) if len(mad) else 1e-3
        scale = mad.clip(lower=max(floor, 1e-3))
        z = (df["r"] - med).abs() / df["cell_id"].map(scale)
        return cls(scale, TailModel.fit(z.to_numpy()))

    def surprisal(self, cells: np.ndarray, ratio: np.ndarray) -> np.ndarray:
        s = self.cell_scale.reindex(cells).fillna(self.cell_scale.median()).to_numpy()[:, None]
        return self.tail.surprisal(np.abs(ratio) / s)


@dataclass
class ResidualOutput:
    expected: np.ndarray  # (C, W, M)
    lo: np.ndarray
    hi: np.ndarray
    signed_ratio: np.ndarray  # (y - yhat) / (yhat + scale)
    surprisal: np.ndarray  # -log10 tail probability


def residual_detector(
    panel: Panel,
    ctx: FeatureContext,
    model: GlobalLGBM,
    calibrators: dict[str, ConformalCalibrator],
    nulls: dict[str, ResidualNull],
    scales: dict[str, np.ndarray],
    times: np.ndarray,
    echo_threshold: float = 4.0,
) -> ResidualOutput:
    shape = (panel.n_cells, len(times), len(panel.metrics))
    expected, lo, hi, ratio, surprisal = (np.full(shape, np.nan) for _ in range(5))
    for k, metric in enumerate(panel.metrics):
        cal, null, scale = calibrators[metric], nulls[metric], scales[metric][:, None]
        y = panel.metric(metric).astype(np.float64)
        logy = np.log1p(y)
        logy_ext = np.concatenate(
            [logy, np.full((logy.shape[0], len(ctx.index) - logy.shape[1]), np.nan)], axis=1
        )
        yw = y[:, times]

        # pass 1
        exp1 = model.predict_observed(panel, ctx, metric, times, DETECT_LAG, override_logy=logy_ext)
        s1 = null.surprisal(panel.cells, (yw - exp1) / (exp1 + scale))
        # pass 2: replace strong deviations with expectation, recompute
        cleaned = logy_ext.copy()
        strong = np.nan_to_num(s1, nan=0.0) >= echo_threshold
        block = cleaned[:, times]
        block[strong] = np.log1p(exp1[strong])
        cleaned[:, times] = block
        exp2 = model.predict_observed(panel, ctx, metric, times, DETECT_LAG, override_logy=cleaned)

        r = (yw - exp2) / (exp2 + scale)
        s = null.surprisal(panel.cells, r)
        bucket = np.full(exp2.shape, DETECT_BUCKET)
        l_, h_ = cal.interval(bucket, exp2, np.broadcast_to(scale, exp2.shape))
        expected[:, :, k], lo[:, :, k], hi[:, :, k] = exp2, l_, h_
        ratio[:, :, k] = r
        surprisal[:, :, k] = s
    return ResidualOutput(expected, lo, hi, ratio, surprisal)
