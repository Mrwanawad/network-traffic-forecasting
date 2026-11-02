"""Anomaly detection orchestration over the detection window."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nettraffic.anomaly.ensemble import EnsembleOutput, build_events, build_points, combine
from nettraffic.anomaly.residual import ResidualNull, residual_detector
from nettraffic.anomaly.unsupervised import IForestDetector, LSTMAutoencoderDetector, deviations
from nettraffic.config import PipelineConfig
from nettraffic.features import FeatureContext
from nettraffic.forecasting.base import ConformalCalibrator
from nettraffic.forecasting.lgbm import GlobalLGBM
from nettraffic.metrics import seasonal_naive_scale
from nettraffic.panel import Panel

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    output: EnsembleOutput
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    times: np.ndarray


def detect(
    panel: Panel,
    ctx: FeatureContext,
    cfg: PipelineConfig,
    calibrators: dict[str, ConformalCalibrator],
    backtest_preds: pd.DataFrame,
    cell_meta: pd.DataFrame | None = None,
) -> DetectionResult:
    det_start = panel.n_times - cfg.detection_days * 24
    times = np.arange(det_start, panel.n_times)
    train_times = np.arange(max(0, det_start - 21 * 24), det_start)

    # 1) residual detector: global model frozen at the start of the window
    model = GlobalLGBM(cfg)
    scales = {}
    for m in panel.metrics:
        model.fit(panel, ctx, det_start, m)
        s = seasonal_naive_scale(panel.metric(m)[:, :det_start].astype(np.float64), cfg.season_length)
        scales[m] = np.where(np.isnan(s), 1.0, s)
    lgbm_preds = backtest_preds[backtest_preds["model"] == model.name]
    nulls = {m: ResidualNull.fit(lgbm_preds[lgbm_preds["metric"] == m]) for m in panel.metrics}
    res = residual_detector(panel, ctx, model, calibrators, nulls, scales, times)
    log.info("residual detector done")

    # 2) model-free detectors on profile deviations
    dev = deviations(panel, ctx, det_start)
    if_scores = IForestDetector(seed=cfg.seed).run(dev, ctx, train_times, times)
    log.info("isolation forest done")
    ae_scores = LSTMAutoencoderDetector(window=cfg.lstm_window, epochs=cfg.lstm_epochs, seed=cfg.seed).run(
        dev, train_times, times
    )
    log.info("lstm autoencoder %s", "done" if ae_scores is not None else "skipped")

    scores = {"residual": res.surprisal, "iforest": if_scores, "lstm_ae": ae_scores}
    combined = combine(scores, cfg.detector_weights)
    observed = panel.values[:, times, :].astype(np.float64)
    combined[np.isnan(observed)] = np.nan  # missing data is a data-quality issue, not an anomaly

    points = build_points(
        panel.cells,
        panel.index[times],
        panel.metrics,
        observed,
        res.expected,
        res.lo,
        res.hi,
        combined,
        scores,
    )
    events = build_events(points, cfg.alert_threshold, cfg.event_merge_gap_hours, cell_meta)
    return DetectionResult(
        EnsembleOutput(points, events), panel.index[det_start], panel.index[-1] + pd.Timedelta(hours=1), times
    )
