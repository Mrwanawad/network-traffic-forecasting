"""Forecast and detection metrics. All functions ignore NaN targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def _valid(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    return y[m], yhat[m]


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _valid(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if y.size else np.nan


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = _valid(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if y.size else np.nan


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in percent; 0/0 counts as a perfect forecast (zero-inflated cells)."""
    y, yhat = _valid(y, yhat)
    if not y.size:
        return np.nan
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(y - yhat) / np.where(denom == 0, 1, denom))
    return float(100 * np.mean(ratio))


def seasonal_naive_scale(history: np.ndarray, m: int) -> np.ndarray:
    """In-sample MAE of the seasonal-naive forecast per series (rows). MASE denominator."""
    diff = np.abs(history[:, m:] - history[:, :-m])
    with np.errstate(all="ignore"):
        scale = np.nanmean(diff, axis=1)
    return np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = ~np.isnan(y)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else np.nan


def forecast_summary(df: pd.DataFrame) -> dict[str, float]:
    """df columns: y, yhat, lo, hi, scale (MASE denominator)."""
    abs_err = (df["y"] - df["yhat"]).abs()
    return {
        "mae": mae(df["y"], df["yhat"]),
        "rmse": rmse(df["y"], df["yhat"]),
        "smape": smape(df["y"], df["yhat"]),
        "mase": float(np.nanmean(abs_err / df["scale"])),
        "coverage": coverage(df["y"], df["lo"], df["hi"]),
        "n": int(df["y"].notna().sum()),
    }


def detection_summary(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, bool)
    score = np.nan_to_num(np.asarray(score, float), nan=0.0)
    pred = score >= threshold
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "pr_auc": float(average_precision_score(y_true, score)) if y_true.any() else np.nan,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "positives": int(y_true.sum()),
        "flagged": int(pred.sum()),
    }
