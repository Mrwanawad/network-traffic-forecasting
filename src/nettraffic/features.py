"""Leakage-safe feature engineering for the global forecasting model.

Direct multi-horizon strategy
-----------------------------
A model for horizon group ``L`` only sees information at ``t - L`` or older.
Forecasting lead ``h`` uses the model with the smallest ``L >= h``:

    lead 1        -> L=1   (lags 1,2,3,...)
    lead 2..24    -> L=24  (lags 24,48,...)
    lead 25..168  -> L=168 (lags 168,336,504)

So every feature used at prediction time is observed before the cutoff, with no
recursive error accumulation and only 3 models per metric instead of 168.

Everything is computed on log1p(traffic): multiplicative seasonality becomes
additive and busy/quiet cells share one model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nettraffic.calendar import CalendarFeatures, ScheduledEvent, calendar_features, event_matrix
from nettraffic.config import PipelineConfig

LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 504)
ROLL_WINDOWS: tuple[int, ...] = (6, 24, 168)
SEASONAL_LAGS: tuple[int, ...] = (168, 336, 504)
AREA_CODES = {"business": 0, "residential": 1, "university": 2, "stadium": 3, "rural": 4}


@dataclass
class FeatureContext:
    """Exogenous information over an index that extends past the data end."""

    index: pd.DatetimeIndex
    cal: CalendarFeatures
    events: np.ndarray  # (C, T_ext) bool
    area: np.ndarray  # (C,) int codes

    @classmethod
    def build(
        cls,
        index: pd.DatetimeIndex,
        cells: np.ndarray,
        cfg: PipelineConfig,
        extra_hours: int,
        events: list[ScheduledEvent] | None = None,
        cell_meta: pd.DataFrame | None = None,
    ) -> FeatureContext:
        ext = pd.date_range(index[0], periods=len(index) + extra_hours, freq="1h")
        cal = calendar_features(ext, cfg.timezone, cfg.country_holidays)
        ev = event_matrix(ext, cells, events or [])
        if cell_meta is not None and "area_type" in cell_meta:
            m = cell_meta.set_index("cell_id")["area_type"].reindex(cells)
            area = m.map(AREA_CODES).fillna(-1).astype(int).to_numpy()
        else:
            area = np.full(len(cells), -1)
        return cls(ext, cal, ev, area)


def extend(logy: np.ndarray, extra: int) -> np.ndarray:
    if extra <= 0:
        return logy
    return np.concatenate([logy, np.full((logy.shape[0], extra), np.nan, logy.dtype)], axis=1)


def _cum(logy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(logy)
    cs = np.zeros((logy.shape[0], logy.shape[1] + 1))
    cn = np.zeros_like(cs)
    cs[:, 1:] = np.cumsum(np.where(valid, logy, 0.0), axis=1)
    cn[:, 1:] = np.cumsum(valid, axis=1)
    return cs, cn


def hour_of_week_profile(logy: np.ndarray, how: np.ndarray, stop: int, weeks: int = 4) -> np.ndarray:
    """(C, 168) median log-traffic per hour-of-week over the `weeks` before `stop`."""
    start = max(0, stop - weeks * 168)
    prof = np.full((logy.shape[0], 168), np.nan)
    seg, h = logy[:, start:stop], how[start:stop]
    for k in range(168):
        cols = seg[:, h == k]
        if cols.shape[1]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof[:, k] = np.nanmedian(cols, axis=1)
    # cells with no history at that slot: fall back to their overall median
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = np.nanmedian(seg, axis=1) if seg.size else np.zeros(logy.shape[0])
    return np.where(np.isnan(prof), np.nan_to_num(fill)[:, None], prof)


def build_features(
    logy: np.ndarray,
    times: np.ndarray,
    min_lag: int,
    ctx: FeatureContext,
    profile: np.ndarray,
) -> pd.DataFrame:
    """Feature rows for every cell x target time in `times` (cell-major order).

    `logy` must already be masked beyond the information horizon by the caller.
    """
    n_cells = logy.shape[0]
    times = np.asarray(times, dtype=int)
    cs, cn = _cum(logy)
    feats: dict[str, np.ndarray] = {}

    def at(lag: int) -> np.ndarray:
        src = times - lag
        out = np.full((n_cells, len(times)), np.nan, dtype=np.float32)
        ok = src >= 0
        out[:, ok] = logy[:, src[ok]]
        return out

    for lag in LAGS:
        if lag >= min_lag:
            feats[f"lag_{lag}"] = at(lag)

    end = times - min_lag + 1  # exclusive end in cumulative coordinates
    for w in ROLL_WINDOWS:
        e = np.clip(end, 0, None)
        s = np.clip(end - w, 0, None)
        num = cs[:, e] - cs[:, s]
        cnt = cn[:, e] - cn[:, s]
        with np.errstate(all="ignore"):
            feats[f"roll_mean_{w}"] = np.where(cnt > 0, num / cnt, np.nan).astype(np.float32)

    seas = np.stack([at(lag) for lag in SEASONAL_LAGS if lag >= min_lag], axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN is intended
        feats["seasonal_median"] = np.nanmedian(seas, axis=0).astype(np.float32)

    how = ctx.cal.hour_of_week[times]
    feats["profile"] = profile[:, how].astype(np.float32)
    feats["level_gap"] = feats["roll_mean_168"] - profile.mean(axis=1, keepdims=True).astype(np.float32)

    def bcast(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a[None, :], (n_cells, len(times)))

    feats["hour"] = bcast(ctx.cal.hour[times])
    feats["dow"] = bcast(ctx.cal.dow[times])
    feats["hour_of_week"] = bcast(how)
    feats["is_offday"] = bcast(ctx.cal.is_offday[times].astype(np.int8))
    feats["is_holiday"] = bcast(ctx.cal.is_holiday[times].astype(np.int8))
    feats["is_eve"] = bcast(ctx.cal.is_eve[times].astype(np.int8))
    feats["is_event"] = ctx.events[:, times].astype(np.int8)
    feats["area"] = np.broadcast_to(ctx.area[:, None], (n_cells, len(times)))

    return pd.DataFrame({k: np.ascontiguousarray(v).reshape(-1) for k, v in feats.items()})


CATEGORICAL = ["area"]
