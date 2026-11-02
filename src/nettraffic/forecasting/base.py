"""Common forecaster contract + prediction-interval calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from nettraffic.features import FeatureContext
from nettraffic.panel import Panel


class Forecaster(Protocol):
    name: str

    def forecast(
        self, panel: Panel, ctx: FeatureContext, cutoff: int, metric: str, horizon: int
    ) -> np.ndarray:
        """Point forecasts (n_cells, horizon) for times [cutoff, cutoff + horizon).

        Implementations must only use observations at positions < cutoff.
        """
        ...


def lead_bucket(lead: np.ndarray | int, groups: tuple[int, ...]) -> np.ndarray:
    """Map a 1-based lead to the label of its horizon group (e.g. '1h', '24h', '168h')."""
    lead = np.asarray(lead)
    labels = np.array([f"{g}h" for g in groups])
    idx = np.searchsorted(np.array(groups), lead, side="left")
    return labels[np.clip(idx, 0, len(groups) - 1)]


@dataclass
class ConformalCalibrator:
    """Normalised split-conformal intervals.

    Non-conformity score ``r = |y - yhat| / (yhat + s)`` where ``s`` is the cell's
    seasonal-naive scale. Dividing by ``yhat + s`` makes the interval width scale
    with the expected traffic (heteroscedastic counts) while ``s`` keeps a floor
    for near-zero cells. Quantiles are learned per horizon bucket from *out-of-
    sample* backtest residuals, so coverage is honest.
    """

    level: float = 0.9
    quantiles: dict[str, float] = field(default_factory=dict)
    null_scores: dict[str, np.ndarray] = field(default_factory=dict)
    _tails: dict = field(default_factory=dict, repr=False)

    def fit(
        self, bucket: np.ndarray, y: np.ndarray, yhat: np.ndarray, scale: np.ndarray
    ) -> ConformalCalibrator:
        r = self.score(y, yhat, scale)
        df = pd.DataFrame({"b": bucket, "r": r}).dropna()
        for b, g in df.groupby("b"):
            n = len(g)
            q = min(1.0, np.ceil((n + 1) * self.level) / n)
            self.quantiles[str(b)] = float(np.quantile(g["r"], q))
            sample = g["r"].to_numpy()
            self.null_scores[str(b)] = np.sort(
                sample if n <= 200_000 else np.random.default_rng(0).choice(sample, 200_000)
            )
        return self

    @staticmethod
    def score(y: np.ndarray, yhat: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(y) - np.asarray(yhat)) / (np.asarray(yhat) + np.asarray(scale))

    def interval(
        self, bucket: np.ndarray, yhat: np.ndarray, scale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.quantiles.get(str(b), np.nan) for b in np.asarray(bucket).ravel()]).reshape(
            np.shape(yhat)
        )
        half = q * (np.asarray(yhat) + np.asarray(scale))
        return np.maximum(0.0, yhat - half), yhat + half

    def tail_prob(self, bucket: str, ratio: np.ndarray) -> np.ndarray:
        """P(null score >= ratio) from the out-of-sample residual distribution
        (empirical in the bulk, exponential peaks-over-threshold in the tail)."""
        from nettraffic.anomaly.tail import TailModel

        if bucket not in self._tails:
            self._tails[bucket] = TailModel.fit(self.null_scores[bucket])
        return self._tails[bucket].tail_prob(ratio)
