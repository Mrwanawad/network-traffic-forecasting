"""Central pipeline configuration.

Everything that changes model behaviour lives here so a run is fully described
by one object (and can be logged to MLflow as params).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

METRICS: tuple[str, ...] = ("sms_in", "sms_out", "call_in", "call_out", "internet")


@dataclass(frozen=True)
class PipelineConfig:
    # --- paths -------------------------------------------------------------
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    site_data_dir: Path = Path("site/data")
    mlflow_uri: str | None = "sqlite:///mlflow.db"  # MLflow 3 deprecates the file store

    # --- data ---------------------------------------------------------------
    timezone: str = "Europe/Rome"
    country_holidays: str = "IT"
    raw_step_minutes: int = 10
    max_interp_steps: int = 3  # gaps <= 30 min are interpolated, longer stay missing
    min_hour_coverage: float = 0.5  # hour kept only if >= 50% of its 10-min slots exist
    metrics: tuple[str, ...] = METRICS

    # --- synthetic data -----------------------------------------------------
    n_cells: int = 500
    start_date: str = "2013-11-01"  # mirrors the Telecom Italia Milan dataset window
    n_days: int = 61
    seed: int = 42

    # --- forecasting ---------------------------------------------------------
    season_length: int = 168  # weekly seasonality at hourly resolution
    horizon: int = 168  # 7 days ahead
    horizon_groups: tuple[int, ...] = (1, 24, 168)  # direct models, min lag per group
    report_horizons: tuple[int, ...] = (1, 24, 168)  # 1h, 24h, 7d
    train_window_hours: int = 28 * 24
    n_backtest_folds: int = 3  # evaluated folds (one extra fold is used for calibration)
    fold_step_hours: int = 168
    interval_level: float = 0.9
    optuna_trials: int = 0  # 0 disables tuning
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "l2",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_estimators": 400,
            "verbose": -1,
        }
    )

    # statistical models are per-series -> evaluated on a sample only
    stat_sample_cells: int = 8
    stat_metrics: tuple[str, ...] = ("internet", "call_in")
    stat_fit_hours: int = 21 * 24

    # --- anomaly detection ---------------------------------------------------
    detection_days: int = 14
    detector_weights: dict[str, float] = field(
        default_factory=lambda: {"residual": 0.5, "iforest": 0.2, "lstm_ae": 0.3}
    )
    alert_threshold: float = 4.5  # ensemble surprisal (-log10 tail prob); ~10 alerts/day on 500 cells
    event_merge_gap_hours: int = 1
    lstm_window: int = 24
    lstm_epochs: int = 4

    # --- capacity -------------------------------------------------------------
    capacity_warn_utilisation: float = 0.85

    # --- export ----------------------------------------------------------------
    export_top_cells: int = 60
    export_history_days: int = 14

    def to_params(self) -> dict[str, Any]:
        """Flat, MLflow-friendly representation."""
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, dict):
                out.update({f"{k}.{kk}": vv for kk, vv in v.items()})
            else:
                out[k] = str(v) if isinstance(v, (Path, tuple)) else v
        return out
