"""EDA summaries that feed the dashboard and justify modelling choices."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nettraffic.calendar import CalendarFeatures
from nettraffic.panel import Panel


def seasonality_strength(series: np.ndarray, period: int) -> dict[str, float]:
    """Hyndman's strength of seasonality / trend from a STL decomposition."""
    from statsmodels.tsa.seasonal import STL

    s = pd.Series(series).interpolate(limit_direction="both").to_numpy()
    if np.allclose(s, s[0]):
        return {"seasonal": 0.0, "trend": 0.0}
    res = STL(np.log1p(s), period=period, robust=False).fit()
    r, sea, tr = res.resid, res.seasonal, res.trend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = max(0.0, 1 - np.var(r) / np.var(sea + r))
        ft = max(0.0, 1 - np.var(r) / np.var(tr + r))
    return {"seasonal": float(fs), "trend": float(ft)}


def run_eda(
    panel: Panel,
    cal: CalendarFeatures,
    cell_meta: pd.DataFrame | None,
    sample: int = 10,
    seed: int = 0,
    tz: str = "Europe/Rome",
) -> dict:
    rng = np.random.default_rng(seed)
    T = panel.n_times
    area = (
        cell_meta.set_index("cell_id")["area_type"].reindex(panel.cells).fillna("unknown").to_numpy()
        if cell_meta is not None and "area_type" in cell_meta
        else np.full(panel.n_cells, "all")
    )

    # seasonality strength (daily vs weekly) on a sample of cells, internet + calls
    picks = rng.choice(panel.n_cells, size=min(sample, panel.n_cells), replace=False)
    strengths = []
    for m in ("internet", "call_in"):
        if m not in panel.metrics:
            continue
        y = panel.metric(m)
        for c in picks:
            for period in (24, 168):
                st = seasonality_strength(y[c], period)
                strengths.append({"metric": m, "period": period, "area": area[c], **st})
    sdf = pd.DataFrame(strengths)

    # area x hour-of-week heatmap of normalised traffic (each cell scaled by its mean)
    how = cal.hour_of_week[:T]
    heat = {}
    for m in panel.metrics:
        y = panel.metric(m).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm = y / np.nanmean(y, axis=1, keepdims=True)
        rows = {}
        for a in np.unique(area):
            sel = norm[area == a]
            rows[str(a)] = [
                float(np.nanmean(sel[:, how == k])) if (how == k).any() else None for k in range(168)
            ]
        heat[m] = rows

    # daily totals per LOCAL calendar day; drop partial first/last days (UTC offset artefacts)
    hourly = pd.DataFrame(
        {m: np.nansum(panel.metric(m), axis=0) for m in panel.metrics}, index=panel.index.tz_convert(tz)
    )
    counts = hourly.iloc[:, 0].resample("1D").count()
    totals = hourly.resample("1D").sum()[counts >= 23]
    corr = pd.DataFrame({m: np.nanmean(panel.metric(m), axis=1) for m in panel.metrics}).corr()

    return {
        "seasonality_strength": sdf.groupby(["metric", "period"])[["seasonal", "trend"]]
        .median()
        .round(3)
        .reset_index()
        .to_dict(orient="records"),
        "seasonality_by_area": sdf[sdf["metric"] == "internet"]
        .groupby(["area", "period"])["seasonal"]
        .median()
        .round(3)
        .reset_index()
        .to_dict(orient="records"),
        "area_how_heatmap": heat,
        "daily_totals": {
            "dates": [d.strftime("%Y-%m-%d") for d in totals.index],
            **{m: totals[m].round(1).tolist() for m in panel.metrics},
        },
        "metric_correlation": corr.round(3).to_dict(),
        "zero_share": {m: float(np.nanmean(panel.metric(m) == 0)) for m in panel.metrics},
    }
