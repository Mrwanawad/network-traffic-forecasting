"""Per-series statistical models (SARIMA, Prophet).

They are fitted one series at a time, so they are evaluated on a *sample* of
cells only. That cost profile is exactly why production uses the global model;
these exist as interpretable benchmarks.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from nettraffic.features import FeatureContext
from nettraffic.forecasting.baselines import SeasonalNaive
from nettraffic.panel import Panel

log = logging.getLogger(__name__)


class SarimaForecaster:
    """SARIMAX(1,0,1)(1,1,1)_24 on log1p with an off-day (weekend/holiday) regressor.

    The weekly pattern is carried by the exogenous off-day flag instead of a
    168-period seasonal term (which would make the state space ~170-dim and
    the fit minutes per series).
    """

    name = "sarima"

    def __init__(self, fit_hours: int = 21 * 24) -> None:
        self.fit_hours = fit_hours
        self._fallback = SeasonalNaive()

    def forecast(
        self, panel: Panel, ctx: FeatureContext, cutoff: int, metric: str, horizon: int
    ) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_all = np.log1p(panel.metric(metric).astype(np.float64))
        start = max(0, cutoff - self.fit_hours)
        exog_all = ctx.cal.is_offday.astype(float)
        out = np.full((panel.n_cells, horizon), np.nan)
        for c in range(panel.n_cells):
            y = pd.Series(y_all[c, start:cutoff]).interpolate(limit_direction="both").to_numpy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(
                        y,
                        exog=exog_all[start:cutoff],
                        order=(1, 0, 1),
                        seasonal_order=(1, 1, 1, 24),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=50)
                    fc = res.forecast(horizon, exog=exog_all[cutoff : cutoff + horizon])
                out[c] = np.clip(np.expm1(fc), 0, None)
            except Exception as exc:  # numerical failure on a degenerate series
                log.warning("SARIMA failed for cell %s (%s); seasonal-naive fallback", panel.cells[c], exc)
                out[c] = self._fallback.forecast(panel, ctx, cutoff, metric, horizon)[c]
        return out


class ProphetForecaster:
    """Prophet with daily + weekly seasonality and country holidays, on log1p."""

    name = "prophet"

    def __init__(self, fit_hours: int = 21 * 24, country: str = "IT", tz: str = "Europe/Rome") -> None:
        self.fit_hours = fit_hours
        self.country = country
        self.tz = tz
        self._fallback = SeasonalNaive()

    def forecast(
        self, panel: Panel, ctx: FeatureContext, cutoff: int, metric: str, horizon: int
    ) -> np.ndarray:
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        start = max(0, cutoff - self.fit_hours)
        # Prophet wants naive local timestamps (seasonality in wall-clock time)
        ds = ctx.index.tz_convert(self.tz).tz_localize(None)
        y_all = np.log1p(panel.metric(metric).astype(np.float64))
        out = np.full((panel.n_cells, horizon), np.nan)
        for c in range(panel.n_cells):
            train = pd.DataFrame({"ds": ds[start:cutoff], "y": y_all[c, start:cutoff]}).dropna()
            try:
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                m.add_country_holidays(country_name=self.country)
                m.fit(train)
                fut = pd.DataFrame({"ds": ds[cutoff : cutoff + horizon]})
                out[c] = np.clip(np.expm1(m.predict(fut)["yhat"].to_numpy()), 0, None)
            except Exception as exc:
                log.warning("Prophet failed for cell %s (%s); seasonal-naive fallback", panel.cells[c], exc)
                out[c] = self._fallback.forecast(panel, ctx, cutoff, metric, horizon)[c]
        return out
