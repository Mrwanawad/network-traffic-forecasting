"""Production monitoring: forecast-error drift, data health, alert load, retrain trigger."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nettraffic.cleaning import CleaningReport

log = logging.getLogger(__name__)


def error_drift(preds: pd.DataFrame, model: str, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """MASE per metric and backtest fold for the production model vs. the baseline."""
    ev = preds[preds["model"].isin([model, baseline])].copy()
    ev["ase"] = (ev["y"] - ev["yhat"]).abs() / ev["scale"]
    g = ev.groupby(["metric", "fold", "cutoff", "model"])["ase"].mean().unstack("model").reset_index()
    g = g.rename(columns={model: "mase_model", baseline: "mase_baseline"})
    g["beats_baseline"] = g["mase_model"] < g["mase_baseline"]
    return g


def per_cell_drift(preds: pd.DataFrame, model: str, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Per-cell MASE on the latest fold vs. the baseline and vs. the cell's own history.

    A cell is `degrading` when the model loses to seasonal-naive on the latest
    fold AND its error grew vs. its mean over earlier folds by > 25%. Comparing
    to the baseline filters out periods that are hard for everyone (holidays).
    """
    ev = preds[preds["model"].isin([model, baseline]) & (preds["fold"] > 0)].copy()
    ev["ase"] = (ev["y"] - ev["yhat"]).abs() / ev["scale"]
    per = ev.groupby(["model", "cell_id", "fold"])["ase"].mean()
    last = int(ev["fold"].max())
    m = per.loc[model].unstack("fold")
    b = per.loc[baseline].unstack("fold")
    out = pd.DataFrame(
        {
            "mase_last": m[last],
            "mase_prev_mean": m.drop(columns=last).mean(axis=1) if m.shape[1] > 1 else m[last],
            "baseline_last": b[last],
        }
    )
    out["degrading"] = (out["mase_last"] > out["baseline_last"]) & (
        out["mase_last"] > 1.25 * out["mase_prev_mean"]
    )
    return out.reset_index()


def retrain_decision(drift: pd.DataFrame, consecutive: int = 2) -> dict:
    """Retrain when the model loses to the baseline for `consecutive` folds on any metric."""
    reasons = []
    for metric, g in drift.sort_values("fold").groupby("metric"):
        losing = (~g["beats_baseline"]).astype(int).to_numpy()
        run = 0
        for v in losing:
            run = run + 1 if v else 0
            if run >= consecutive:
                reasons.append(f"{metric}: worse than seasonal-naive for {run} consecutive folds")
                break
    return {"retrain": bool(reasons), "reasons": reasons}


def data_health(report: CleaningReport, last_ts: pd.Timestamp, now: pd.Timestamp | None = None) -> dict:
    now = now or last_ts + pd.Timedelta(hours=1)  # batch mode: "now" = end of the data
    return {
        "last_observation": last_ts.isoformat(),
        "freshness_hours": round((now - last_ts) / pd.Timedelta(hours=1), 2),
        **report.as_dict(),
    }


def alert_load(
    events: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp, shifts_per_day: int = 3
) -> dict:
    days = max((window_end - window_start) / pd.Timedelta(days=1), 1e-9)
    incidents = events["incident_id"].nunique() if len(events) else 0
    return {
        "alerts_per_day": round(len(events) / days, 2),
        "incidents_per_day": round(incidents / days, 2),
        "incidents_per_shift": round(incidents / days / shifts_per_day, 2),
        "by_severity": events["severity"].value_counts().to_dict() if len(events) else {},
    }


def feedback_precision(feedback_csv: Path) -> dict | None:
    """Operator feedback loop: CSV with columns event_id, verdict (confirmed|dismissed)."""
    if not feedback_csv.exists():
        return None
    fb = pd.read_csv(feedback_csv)
    n = len(fb)
    conf = int((fb["verdict"].str.lower() == "confirmed").sum())
    return {"reviewed": n, "confirmed": conf, "precision": conf / n if n else np.nan}


def evidently_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> bool:
    """Optional Evidently data-drift report (skipped when evidently is not installed)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        log.info("evidently not installed - drift report skipped")
        return False
    try:  # Evidently's API changes between majors; never let it break the batch run
        snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
        snapshot.save_html(str(out_html))
    except Exception as exc:
        log.warning("evidently report failed: %s", exc)
        return False
    return True
