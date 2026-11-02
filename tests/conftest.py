from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nettraffic.cleaning import clean
from nettraffic.config import PipelineConfig
from nettraffic.data.synthetic import SyntheticDataset, generate
from nettraffic.features import FeatureContext
from nettraffic.panel import Panel


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    root = tmp_path_factory.mktemp("run")
    return dataclasses.replace(
        PipelineConfig(),
        n_cells=36,
        n_days=42,
        n_backtest_folds=2,
        detection_days=7,
        lstm_epochs=1,
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        site_data_dir=root / "site",
        mlflow_uri=None,
        lgbm_params={**PipelineConfig().lgbm_params, "n_estimators": 60, "num_leaves": 15},
    )


@pytest.fixture(scope="session")
def synth(small_cfg: PipelineConfig) -> SyntheticDataset:
    return generate(small_cfg)


@pytest.fixture(scope="session")
def panel(synth: SyntheticDataset, small_cfg: PipelineConfig) -> Panel:
    p, _ = clean(synth.raw, small_cfg)
    return p


@pytest.fixture(scope="session")
def ctx(panel: Panel, small_cfg: PipelineConfig, synth: SyntheticDataset) -> FeatureContext:
    return FeatureContext.build(
        panel.index, panel.cells, small_cfg, small_cfg.horizon, synth.events, synth.cells
    )


def make_panel(
    values: np.ndarray, start: str = "2024-01-01", metrics: tuple[str, ...] = ("internet",)
) -> Panel:
    if values.ndim == 2:
        values = values[:, :, None]
    idx = pd.date_range(start, periods=values.shape[1], freq="1h", tz="UTC")
    return Panel(values.astype(np.float32), np.arange(1, values.shape[0] + 1), idx, metrics)
