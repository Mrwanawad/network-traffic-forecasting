"""Global LightGBM forecaster: one model per (metric, horizon group) across ALL cells.

Why global: thousands of short, noisy series share the same diurnal/weekly
physics. A single model learns that shape once, cold-start cells borrow
strength from neighbours, and training cost scales with rows, not with
the number of series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from nettraffic.config import PipelineConfig
from nettraffic.features import CATEGORICAL, FeatureContext, build_features, hour_of_week_profile
from nettraffic.panel import Panel

log = logging.getLogger(__name__)


@dataclass
class FittedGroup:
    min_lag: int
    model: lgb.LGBMRegressor


@dataclass
class GlobalLGBM:
    cfg: PipelineConfig
    params: dict[str, Any] = field(default_factory=dict)
    name: str = "lightgbm_global"
    # fitted state (per metric)
    _groups: dict[str, list[FittedGroup]] = field(default_factory=dict, repr=False)
    _profile: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _cutoff: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.params:
            self.params = dict(self.cfg.lgbm_params)

    # ------------------------------------------------------------------ fit
    def fit(self, panel: Panel, ctx: FeatureContext, cutoff: int, metric: str) -> GlobalLGBM:
        logy = _log_masked(panel, metric, cutoff)
        profile = hour_of_week_profile(logy, ctx.cal.hour_of_week, cutoff)
        start = max(0, cutoff - self.cfg.train_window_hours)
        times = np.arange(start, cutoff)
        y = logy[:, times].reshape(-1)
        groups = []
        for L in self.cfg.horizon_groups:
            X = build_features(logy, times, L, ctx, profile)
            ok = ~np.isnan(y)
            model = lgb.LGBMRegressor(**self.params, random_state=self.cfg.seed, n_jobs=-1)
            model.fit(X[ok], y[ok], categorical_feature=CATEGORICAL)
            groups.append(FittedGroup(L, model))
        self._groups[metric], self._profile[metric], self._cutoff[metric] = groups, profile, cutoff
        return self

    # -------------------------------------------------------------- predict
    def predict_horizon(self, panel: Panel, ctx: FeatureContext, metric: str, horizon: int) -> np.ndarray:
        """Forecast (C, horizon) after the fitted cutoff, picking the group per lead."""
        if horizon > max(self.cfg.horizon_groups):
            raise ValueError(
                f"horizon {horizon} exceeds largest horizon group {max(self.cfg.horizon_groups)}"
            )
        cutoff = self._cutoff[metric]
        logy = _log_masked(panel, metric, cutoff, total=len(ctx.index))
        out = np.full((panel.n_cells, horizon), np.nan)
        leads = np.arange(1, horizon + 1)
        prev = 0
        for g in self._groups[metric]:
            sel = (leads > prev) & (leads <= g.min_lag)
            prev = g.min_lag
            if not sel.any():
                continue
            times = cutoff + leads[sel] - 1
            X = build_features(logy, times, g.min_lag, ctx, self._profile[metric])
            out[:, sel] = g.model.predict(X).reshape(panel.n_cells, -1)
        return np.clip(np.expm1(out), 0, None)

    def predict_observed(
        self,
        panel: Panel,
        ctx: FeatureContext,
        metric: str,
        times: np.ndarray,
        min_lag: int,
        override_logy: np.ndarray | None = None,
    ) -> np.ndarray:
        """'Expected' traffic for already-observed times using the min_lag group.

        Used by anomaly detection: the model is frozen at its cutoff, features
        use real observations up to t - min_lag (optionally an anomaly-cleaned copy).
        """
        group = next(g for g in self._groups[metric] if g.min_lag == min_lag)
        logy = (
            override_logy if override_logy is not None else np.log1p(panel.metric(metric).astype(np.float64))
        )
        X = build_features(logy, np.asarray(times), min_lag, ctx, self._profile[metric])
        return np.clip(np.expm1(group.model.predict(X).reshape(panel.n_cells, -1)), 0, None)

    def forecast(
        self, panel: Panel, ctx: FeatureContext, cutoff: int, metric: str, horizon: int
    ) -> np.ndarray:
        self.fit(panel, ctx, cutoff, metric)
        return self.predict_horizon(panel, ctx, metric, horizon)

    def feature_importance(self, metric: str) -> pd.DataFrame:
        rows = []
        for g in self._groups.get(metric, []):
            booster = g.model.booster_
            for f, v in zip(booster.feature_name(), booster.feature_importance("gain"), strict=True):
                rows.append({"metric": metric, "min_lag": g.min_lag, "feature": f, "gain": float(v)})
        return pd.DataFrame(rows)


def _log_masked(panel: Panel, metric: str, cutoff: int, total: int | None = None) -> np.ndarray:
    """log1p traffic with everything at/after `cutoff` hidden (leakage guard)."""
    logy = np.log1p(panel.metric(metric).astype(np.float64))
    total = total or logy.shape[1]
    out = np.full((logy.shape[0], total), np.nan)
    out[:, :cutoff] = logy[:, :cutoff]
    return out


# --------------------------------------------------------------------------- tuning
def tune_lgbm(
    panel: Panel, ctx: FeatureContext, cfg: PipelineConfig, metric: str, cutoff: int
) -> dict[str, Any]:
    """Optuna search on a time-ordered split (train < cutoff-168 <= valid < cutoff), L=24 group."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    valid_start = cutoff - 168
    logy = _log_masked(panel, metric, cutoff)
    train_logy = _log_masked(panel, metric, valid_start, total=logy.shape[1])
    profile = hour_of_week_profile(train_logy, ctx.cal.hour_of_week, valid_start)
    tr_times = np.arange(max(0, valid_start - cfg.train_window_hours), valid_start)
    va_times = np.arange(valid_start, cutoff)
    # validation features may use real values up to t-24 (all < cutoff): that is the L=24 setting
    X_tr = build_features(train_logy, tr_times, 24, ctx, profile)
    X_va = build_features(logy, va_times, 24, ctx, profile)
    y_tr, y_va = train_logy[:, tr_times].reshape(-1), logy[:, va_times].reshape(-1)
    m_tr, m_va = ~np.isnan(y_tr), ~np.isnan(y_va)

    def objective(trial: optuna.Trial) -> float:
        params = dict(cfg.lgbm_params)
        params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            n_estimators=trial.suggest_int("n_estimators", 150, 800, step=50),
        )
        model = lgb.LGBMRegressor(**params, random_state=cfg.seed, n_jobs=-1)
        model.fit(X_tr[m_tr], y_tr[m_tr], categorical_feature=CATEGORICAL)
        pred = model.predict(X_va[m_va])
        return float(np.mean(np.abs(np.expm1(pred) - np.expm1(y_va[m_va]))))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.optuna_trials, show_progress_bar=False)
    best = dict(cfg.lgbm_params)
    best.update(study.best_params)
    log.info("optuna best MAE=%.3f params=%s", study.best_value, study.best_params)
    return best
