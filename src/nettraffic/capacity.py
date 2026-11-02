"""Capacity report: which cells are forecast to approach / exceed capacity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def capacity_report(
    forecasts: pd.DataFrame,
    history: pd.DataFrame,
    cell_meta: pd.DataFrame | None,
    metric: str = "internet",
    warn_utilisation: float = 0.85,
    fallback_quantile: float = 0.995,
    fallback_headroom: float = 1.10,
) -> pd.DataFrame:
    """forecasts: cell_id, metric, ts, yhat, lo, hi. history: cell_id, ts, <metric>.

    Capacity comes from cell metadata (`capacity_<metric>`) when available,
    otherwise a proxy: the historical 99.5th percentile x 1.10 headroom. The proxy
    is flagged so planners do not mistake it for an engineering limit.
    """
    fc = forecasts[forecasts["metric"] == metric]
    idx = fc.groupby("cell_id")["yhat"].idxmax()
    peak = fc.loc[idx, ["cell_id", "ts", "yhat", "hi"]].rename(
        columns={"ts": "peak_ts", "yhat": "peak_forecast", "hi": "peak_upper"}
    )
    peak_upper = fc.groupby("cell_id")["hi"].max().rename("max_upper")
    peak = peak.merge(peak_upper, on="cell_id")

    col = f"capacity_{metric}"
    if cell_meta is not None and col in cell_meta:
        cap = cell_meta.set_index("cell_id")[col].rename("capacity")
        source = "metadata"
    else:
        cap = (history.groupby("cell_id")[metric].quantile(fallback_quantile) * fallback_headroom).rename(
            "capacity"
        )
        source = "historical_proxy"
    out = peak.merge(cap, left_on="cell_id", right_index=True, how="left")
    out["utilisation"] = out["peak_forecast"] / out["capacity"]
    out["utilisation_upper"] = out["max_upper"] / out["capacity"]
    out["status"] = np.select(
        [out["utilisation"] >= 1.0, out["utilisation_upper"] >= 1.0, out["utilisation"] >= warn_utilisation],
        ["over_capacity", "at_risk", "watch"],
        default="ok",
    )
    out["capacity_source"] = source
    out["metric"] = metric
    order = {"over_capacity": 0, "at_risk": 1, "watch": 2, "ok": 3}
    return (
        out.assign(_o=out["status"].map(order))
        .sort_values(["_o", "utilisation"], ascending=[True, False])
        .drop(columns="_o")
        .reset_index(drop=True)
    )
