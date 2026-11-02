"""End-to-end batch pipeline + CLI.

    nettraffic run                      # synthetic data, full pipeline
    nettraffic run --milan-dir data/milan --no-stat-models
    nettraffic run --quick              # small, fast config for CI / smoke tests

Stages: ingest -> clean -> EDA -> backtest -> final forecast -> anomaly detection
-> capacity -> monitoring -> exports (artifacts/*.parquet + site/data/*.json).
Each stage writes its output so the API and the dashboard never recompute.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nettraffic.anomaly.detect import detect
from nettraffic.anomaly.evaluate import evaluate_detection
from nettraffic.calendar import ScheduledEvent, events_to_frame
from nettraffic.capacity import capacity_report
from nettraffic.cleaning import CleaningReport, clean
from nettraffic.config import PipelineConfig
from nettraffic.data.ingest import scan_partitioned, write_partitioned
from nettraffic.eda import run_eda
from nettraffic.export import export_cell_series, frame_records, write_json
from nettraffic.features import FeatureContext
from nettraffic.forecasting.backtest import BacktestResult, run_backtest, summarize
from nettraffic.forecasting.base import lead_bucket
from nettraffic.forecasting.baselines import SeasonalNaive
from nettraffic.forecasting.lgbm import GlobalLGBM, tune_lgbm
from nettraffic.metrics import seasonal_naive_scale
from nettraffic.monitoring import (
    alert_load,
    data_health,
    error_drift,
    evidently_drift_report,
    feedback_precision,
    per_cell_drift,
    retrain_decision,
)
from nettraffic.panel import Panel

log = logging.getLogger("nettraffic")


# --------------------------------------------------------------------------- mlflow
@contextlib.contextmanager
def tracking(cfg: PipelineConfig, run_name: str) -> Iterator[Any]:
    """MLflow run if available/configured, otherwise a no-op recorder."""
    ml = None
    if cfg.mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.mlflow_uri)
            mlflow.set_experiment("network-traffic-forecasting")
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: str(v)[:250] for k, v in cfg.to_params().items()})
            ml = mlflow
        except ImportError:
            log.info("mlflow not installed - tracking disabled")
        except Exception as exc:  # tracking must never break the batch job
            log.warning("mlflow unavailable (%s) - tracking disabled", exc)
    try:
        yield ml
    except BaseException:
        if ml is not None:
            ml.end_run(status="FAILED")
        raise
    else:
        if ml is not None:
            ml.end_run()


def _log_metrics(ml: Any, prefix: str, d: dict[str, Any]) -> None:
    if ml is None:
        return
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            ml.log_metric(f"{prefix}.{k}".replace(" ", "_"), float(v))


# --------------------------------------------------------------------------- stages
@dataclasses.dataclass
class LoadedData:
    raw_root: Path
    cell_meta: pd.DataFrame | None
    labels: pd.DataFrame | None
    events: list[ScheduledEvent]
    generator_log: dict[str, int]


def stage_ingest(cfg: PipelineConfig, milan_dir: Path | None) -> LoadedData:
    raw_root = cfg.data_dir / "raw"
    if milan_dir is not None:
        from nettraffic.data.milan import load_milan

        df = load_milan(milan_dir)
        write_partitioned(df, raw_root, overwrite=True)
        return LoadedData(raw_root, None, None, [], {})

    from nettraffic.data.synthetic import generate

    ds = generate(cfg)
    write_partitioned(ds.raw, raw_root, overwrite=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    ds.cells.to_parquet(cfg.data_dir / "cells.parquet", index=False)
    ds.labels.to_parquet(cfg.data_dir / "labels.parquet", index=False)
    events_to_frame(ds.events).to_parquet(cfg.data_dir / "scheduled_events.parquet", index=False)
    return LoadedData(raw_root, ds.cells, ds.labels, ds.events, ds.error_log)


def stage_clean(cfg: PipelineConfig, data: LoadedData) -> tuple[Panel, CleaningReport]:
    raw = scan_partitioned(data.raw_root).drop("date").collect()
    panel, rep = clean(raw, cfg)
    panel.save(cfg.data_dir / "clean" / "hourly.parquet")
    log.info("cleaning: %s", rep.as_dict())
    return panel, rep


def final_forecast(
    panel: Panel, ctx: FeatureContext, cfg: PipelineConfig, bt: BacktestResult, params: dict[str, Any]
) -> pd.DataFrame:
    model = GlobalLGBM(cfg, params=params)
    frames = []
    future = pd.date_range(panel.index[-1] + pd.Timedelta(hours=1), periods=cfg.horizon, freq="1h")
    leads = np.arange(1, cfg.horizon + 1)
    buckets = lead_bucket(leads, cfg.horizon_groups)
    for m in panel.metrics:
        model.fit(panel, ctx, panel.n_times, m)
        yhat = model.predict_horizon(panel, ctx, m, cfg.horizon)
        scale = seasonal_naive_scale(panel.metric(m).astype(np.float64), cfg.season_length)
        scale = np.where(np.isnan(scale), 1.0, scale)
        cal = bt.calibrators[(model.name, m)]
        lo, hi = cal.interval(
            np.broadcast_to(buckets, yhat.shape), yhat, np.broadcast_to(scale[:, None], yhat.shape)
        )
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": np.repeat(panel.cells, cfg.horizon),
                    "metric": m,
                    "ts": np.tile(future, panel.n_cells),
                    "lead": np.tile(leads, panel.n_cells),
                    "yhat": yhat.reshape(-1),
                    "lo": lo.reshape(-1),
                    "hi": hi.reshape(-1),
                }
            )
        )
    imp = pd.concat([model.feature_importance(m) for m in panel.metrics], ignore_index=True)
    imp.to_parquet(cfg.artifacts_dir / "feature_importance.parquet", index=False)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- run
def run(cfg: PipelineConfig, milan_dir: Path | None = None, stat_models: bool = True) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def tick(name: str, t0: float) -> float:
        timings[name] = round(time.perf_counter() - t0, 2)
        log.info("stage %-14s %6.1fs", name, timings[name])
        return time.perf_counter()

    with tracking(cfg, "batch") as ml:
        t = time.perf_counter()
        data = stage_ingest(cfg, milan_dir)
        t = tick("ingest", t)
        panel, cleaning = stage_clean(cfg, data)
        t = tick("clean", t)

        ctx = FeatureContext.build(panel.index, panel.cells, cfg, cfg.horizon, data.events, data.cell_meta)
        eda = run_eda(panel, ctx.cal, data.cell_meta, seed=cfg.seed, tz=cfg.timezone)
        write_json(cfg.artifacts_dir / "eda.json", eda)
        t = tick("eda", t)

        # ---- optional hyper-parameter search (time-ordered split, never random)
        params = dict(cfg.lgbm_params)
        if cfg.optuna_trials > 0:
            cut = panel.n_times - cfg.horizon - cfg.n_backtest_folds * cfg.fold_step_hours
            params = tune_lgbm(
                panel, ctx, cfg, "internet" if "internet" in panel.metrics else panel.metrics[0], cut
            )
            t = tick("tuning", t)

        # ---- backtest
        series_models: list[Any] = []
        sample = None
        if stat_models:
            from nettraffic.forecasting.statistical import ProphetForecaster, SarimaForecaster

            series_models = [SarimaForecaster(cfg.stat_fit_hours)]
            try:
                import prophet  # noqa: F401

                series_models.append(
                    ProphetForecaster(cfg.stat_fit_hours, cfg.country_holidays, cfg.timezone)
                )
            except ImportError:
                log.warning("prophet not installed - skipped")
            vol = np.nanmean(panel.metric(panel.metrics[-1]), axis=1)
            order = np.argsort(vol)
            # stratified sample over the volume distribution (quiet -> busy)
            sample = panel.cells[order[np.linspace(0, len(order) - 1, cfg.stat_sample_cells).astype(int)]]
        bt = run_backtest(
            panel,
            ctx,
            cfg,
            [SeasonalNaive(cfg.season_length), GlobalLGBM(cfg, params=params)],
            series_models,
            sample,
            series_metrics=cfg.stat_metrics,
        )
        bt.predictions.to_parquet(cfg.artifacts_dir / "backtest_predictions.parquet", index=False)
        volume = pd.Series(np.nanmean(panel.metric(panel.metrics[-1]), axis=1), index=panel.cells)
        offday_ts = set(ctx.index[: panel.n_times][ctx.cal.is_offday[: panel.n_times]])
        tables = summarize(bt.predictions, cfg, volume, offday_ts)
        for name, df in tables.items():
            df.to_csv(cfg.artifacts_dir / f"backtest_{name}.csv", index=False)
        for r in tables["overall"].to_dict(orient="records"):
            _log_metrics(
                ml, f"backtest.{r['model']}", {k: r[k] for k in ("mae", "rmse", "smape", "mase", "coverage")}
            )
        t = tick("backtest", t)

        # ---- production forecast
        forecasts = final_forecast(panel, ctx, cfg, bt, params)
        forecasts.to_parquet(cfg.artifacts_dir / "forecasts.parquet", index=False)
        t = tick("forecast", t)

        # ---- anomaly detection
        cals = {m: bt.calibrators[("lightgbm_global", m)] for m in panel.metrics}
        det = detect(panel, ctx, cfg, cals, bt.predictions, data.cell_meta)
        events = det.output.events
        det.output.points.to_parquet(cfg.artifacts_dir / "anomaly_scores.parquet", index=False)
        events.to_parquet(cfg.artifacts_dir / "anomalies.parquet", index=False)
        evaluation: dict[str, Any] | None = None
        if data.labels is not None and len(data.labels):
            ev_mask = ctx.events[:, det.times]
            ev_alerts = 0
            if ev_mask.any() and len(events):
                pos = {int(c): i for i, c in enumerate(panel.cells)}
                for r in events.itertuples():
                    tt = (panel.index[det.times] >= r.start) & (panel.index[det.times] < r.end)
                    ev_alerts += int(ev_mask[pos[r.cell_id], tt].any())
            evaluation = evaluate_detection(
                det.output.points,
                events,
                data.labels,
                cfg.alert_threshold,
                det.window_start,
                det.window_end,
                ev_alerts,
            )
            write_json(cfg.artifacts_dir / "detection_eval.json", evaluation)
            _log_metrics(ml, "detection.event", evaluation["event_level"])
            _log_metrics(ml, "detection.point", evaluation["point_level"]["ensemble"])
        t = tick("detection", t)

        # ---- capacity
        hist = panel.to_long()
        cap = capacity_report(forecasts, hist, data.cell_meta, "internet", cfg.capacity_warn_utilisation)
        cap.to_parquet(cfg.artifacts_dir / "capacity.parquet", index=False)
        t = tick("capacity", t)

        # ---- monitoring
        drift = error_drift(bt.predictions, "lightgbm_global")
        cell_drift = per_cell_drift(bt.predictions, "lightgbm_global")
        monitoring = {
            "data_health": data_health(cleaning, panel.index[-1]),
            "error_drift": frame_records(drift),
            "degrading_cells": int(cell_drift["degrading"].sum()),
            "retrain": retrain_decision(drift),
            "alert_load": alert_load(events, det.window_start, det.window_end),
            "operator_feedback": feedback_precision(cfg.data_dir / "alert_feedback.csv"),
        }
        ref = hist[hist["ts"] < det.window_start].drop(columns=["cell_id", "ts"]).dropna()
        cur = hist[hist["ts"] >= det.window_start].drop(columns=["cell_id", "ts"]).dropna()
        ref = ref.sample(min(20_000, len(ref)), random_state=0)
        cur = cur.sample(min(20_000, len(cur)), random_state=0)
        monitoring["evidently_report"] = evidently_drift_report(
            ref, cur, cfg.artifacts_dir / "data_drift.html"
        )
        write_json(cfg.artifacts_dir / "monitoring.json", monitoring)
        t = tick("monitoring", t)

        # ---- dashboard exports
        export_site(
            cfg, panel, data, cleaning, eda, tables, bt, forecasts, det, cap, evaluation, monitoring, timings
        )
        tick("export", t)

        summary = {
            "cells": panel.n_cells,
            "hours": panel.n_times,
            "overall": tables["overall"].to_dict(orient="records"),
            "detection": evaluation["event_level"] if evaluation else None,
            "alerts": len(events),
            "capacity_flags": cap["status"].value_counts().to_dict(),
            "timings": timings,
            "total_seconds": round(time.perf_counter() - t_start, 1),
        }
        write_json(cfg.artifacts_dir / "run_summary.json", summary)
        if ml is not None:
            for f in (
                "run_summary.json",
                "monitoring.json",
                "backtest_overall.csv",
                "backtest_by_horizon.csv",
            ):
                ml.log_artifact(str(cfg.artifacts_dir / f))
    return summary


def export_site(
    cfg: PipelineConfig,
    panel: Panel,
    data: LoadedData,
    cleaning: CleaningReport,
    eda: dict,
    tables: dict[str, pd.DataFrame],
    bt: BacktestResult,
    forecasts: pd.DataFrame,
    det: Any,
    cap: pd.DataFrame,
    evaluation: dict | None,
    monitoring: dict,
    timings: dict,
) -> None:
    out = cfg.site_data_dir
    events = det.output.events
    meta = data.cell_meta if data.cell_meta is not None else pd.DataFrame({"cell_id": panel.cells})
    means = {m: np.nanmean(panel.metric(m), axis=1) for m in panel.metrics}
    cells = meta.set_index("cell_id").reindex(panel.cells).reset_index()
    for m in panel.metrics:
        cells[f"mean_{m}"] = means[m]
    alert_counts = events.groupby("cell_id").size() if len(events) else pd.Series(dtype=int)
    cells["alerts"] = cells["cell_id"].map(alert_counts).fillna(0).astype(int)
    cap_status = cap.set_index("cell_id")["status"]
    cells["capacity_status"] = cells["cell_id"].map(cap_status)

    # top-N busiest cells + every alerted / capacity-flagged cell (bounded)
    busiest = cells.sort_values(f"mean_{panel.metrics[-1]}", ascending=False)["cell_id"].head(
        cfg.export_top_cells
    )
    flagged = (
        pd.concat(
            [
                events.sort_values("score", ascending=False)["cell_id"],
                cap.loc[cap["status"] != "ok", "cell_id"],
            ]
        )
        .drop_duplicates()
        .head(cfg.export_top_cells * 2)
    )
    export_ids = list(dict.fromkeys([*busiest.astype(int), *flagged.astype(int)]))
    cells["has_series"] = cells["cell_id"].isin(export_ids)

    # local-hour grid frames for the map (mean internet per local hour of day)
    local_hour = panel.index.tz_convert(cfg.timezone).hour.to_numpy()
    y = panel.metric(panel.metrics[-1]).astype(float)
    hour_means = np.stack([np.nanmean(y[:, local_hour == h], axis=1) for h in range(24)], axis=1)

    write_json(
        out / "cells.json", {"cells": frame_records(cells), "hourly_mean_internet": hour_means.round(1)}
    )
    write_json(
        out / "meta.json",
        {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "source": "synthetic" if data.labels is not None else "milan",
            "data_start": panel.index[0].isoformat(),
            "data_end": panel.index[-1].isoformat(),
            "timezone": cfg.timezone,
            "n_cells": panel.n_cells,
            "metrics": list(panel.metrics),
            "horizon_hours": cfg.horizon,
            "interval_level": cfg.interval_level,
            "alert_threshold": cfg.alert_threshold,
            "detection_window": [det.window_start.isoformat(), det.window_end.isoformat()],
            "cleaning": cleaning.as_dict(),
            "generator_errors": data.generator_log,
            "timings": timings,
        },
    )
    write_json(out / "eda.json", eda)
    write_json(out / "alerts.json", {"alerts": frame_records(events), "evaluation": evaluation})
    write_json(
        out / "models.json",
        {k: frame_records(v) for k, v in tables.items()} | {"timings": bt.timings},
    )
    write_json(out / "capacity.json", {"cells": frame_records(cap)})
    write_json(out / "monitoring.json", monitoring)
    export_cell_series(out, panel, export_ids, cfg.export_history_days * 24, det.output.points, forecasts)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="nettraffic", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full batch pipeline")
    r.add_argument(
        "--milan-dir", type=Path, default=None, help="folder with sms-call-internet-mi-*.txt files"
    )
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--days", type=int, default=None)
    r.add_argument("--seed", type=int, default=None, help="synthetic scenario seed (default 42)")
    r.add_argument("--tune-trials", type=int, default=None, help="Optuna trials (0 = off)")
    r.add_argument("--no-stat-models", action="store_true", help="skip SARIMA/Prophet benchmarks")
    r.add_argument("--no-mlflow", action="store_true")
    r.add_argument("--quick", action="store_true", help="small config for smoke tests")
    r.add_argument("--artifacts", type=Path, default=None)
    r.add_argument("--site-data", type=Path, default=None)
    r.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    for noisy in ("cmdstanpy", "prophet", "prophet.plot", "matplotlib", "lightgbm", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = PipelineConfig()
    overrides: dict[str, Any] = {}
    if a.quick:
        overrides.update(
            n_cells=60,
            n_days=49,
            n_backtest_folds=2,
            stat_sample_cells=3,
            lstm_epochs=2,
            export_top_cells=15,
            detection_days=10,
        )
    if a.cells is not None:
        overrides["n_cells"] = a.cells
    if a.days is not None:
        overrides["n_days"] = a.days
    if a.seed is not None:
        overrides["seed"] = a.seed
    if a.tune_trials is not None:
        overrides["optuna_trials"] = a.tune_trials
    if a.no_mlflow:
        overrides["mlflow_uri"] = None
    if a.artifacts:
        overrides["artifacts_dir"] = a.artifacts
    if a.site_data:
        overrides["site_data_dir"] = a.site_data
    cfg = dataclasses.replace(cfg, **overrides)

    summary = run(cfg, a.milan_dir, stat_models=not a.no_stat_models)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
