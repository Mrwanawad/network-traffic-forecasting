"""Calibrated tail probabilities with peaks-over-threshold extrapolation.

An empirical null of n samples cannot express probabilities below ~1/n, so
every strong anomaly would saturate at the same score and could not be ranked.
Beyond the null's 95th percentile we fit an exponential tail (GPD with shape 0)
on the exceedances:

    P(S >= x) = 0.05 * exp(-(x - u) / beta),   x > u = q95,  beta = mean(S - u | S > u)

This keeps calibrated values inside the bulk and a monotone, unbounded
surprisal in the tail - which is what an alert ranking needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TailModel:
    null_sorted: np.ndarray
    u: float
    beta: float
    tail_mass: float = 0.05

    @classmethod
    def fit(cls, null: np.ndarray, tail_mass: float = 0.05) -> TailModel:
        x = np.sort(np.asarray(null, dtype=float)[np.isfinite(null)])
        if x.size < 50:
            raise ValueError("null sample too small for a tail model")
        u = float(np.quantile(x, 1 - tail_mass))
        exc = x[x > u] - u
        beta = float(exc.mean()) if exc.size else 1e-6
        return cls(x, u, max(beta, 1e-9), tail_mass)

    def tail_prob(self, score: np.ndarray) -> np.ndarray:
        s = np.asarray(score, dtype=float)
        n = len(self.null_sorted)
        rank = np.searchsorted(self.null_sorted, s, side="left")
        emp = (n - rank + 1) / (n + 1)
        pot = self.tail_mass * np.exp(-(s - self.u) / self.beta)
        p = np.where(s > self.u, pot, emp)
        return np.clip(p, 1e-12, 1.0)

    def surprisal(self, score: np.ndarray) -> np.ndarray:
        """-log10 P(S >= score); 0 = typical, 3 = 1-in-1000, 6 = 1-in-a-million."""
        out = -np.log10(self.tail_prob(score))
        return np.where(np.isnan(score), np.nan, out)
