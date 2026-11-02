"""Seasonal-naive baseline: the bar every model has to beat."""

from __future__ import annotations

import numpy as np

from nettraffic.features import FeatureContext
from nettraffic.panel import Panel


class SeasonalNaive:
    """y_hat(t) = y(t - k*m) with the smallest k that lands before the cutoff.

    Falls back one more season back when that value is missing.
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 168) -> None:
        self.season = season

    def forecast(
        self, panel: Panel, ctx: FeatureContext, cutoff: int, metric: str, horizon: int
    ) -> np.ndarray:
        y = panel.metric(metric)[:, :cutoff].astype(np.float64)
        out = np.full((y.shape[0], horizon), np.nan)
        for h in range(1, horizon + 1):
            k = int(np.ceil(h / self.season))
            for extra in range(3):
                src = cutoff + h - 1 - (k + extra) * self.season
                if src < 0:
                    break
                col = out[:, h - 1]
                fill = np.isnan(col)
                col[fill] = y[fill, src]
                if not np.isnan(col).any():
                    break
        return out
