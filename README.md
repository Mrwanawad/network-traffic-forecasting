# Network Traffic Forecasting & Anomaly Detection

Forecast telecom traffic (SMS, calls, internet) per cell, flag abnormal drops and spikes before customers complain, and point out cells that are about to run out of capacity.

**Live dashboard:** https://network-traffic-forecasting.mrwaanawad.workers.dev, refreshed every day by GitHub Actions (see [Automated daily refresh](#automated-daily-refresh)).

<p align="center"><img src="docs/demo.gif" width="100%" alt="Dashboard walkthrough: fleet KPIs, opening the top critical alert, actual vs. expected traffic with the flagged drop and 7-day forecast, and the capacity-risk chart"/></p>

The full design is in [`docs/architecture.md`](docs/architecture.md). This README covers how the code maps to it, how to run it, and what the numbers look like.

## Results (synthetic run: 500 cells, 61 days, 3 evaluated backtest folds)

Forecasting, all cells, rolling-origin backtest (MASE < 1 means better than seasonal naive in-sample):

| Model | MAE | sMAPE | MASE | 90% PI coverage |
|---|---|---|---|---|
| LightGBM global (Optuna, 12 trials) | 31.1 | 37.0% | **0.909** | 89.1% |
| Seasonal naive (weekly) | 36.5 | 38.4% | 1.093 | 89.2% |

By horizon bucket the global model wins everywhere except one spot: at lead exactly 1h, SMS is worse than naive (0.85 vs 0.68). That is 1.5k single-hour points landing on festive midnights; it's noisy but real, so it stays in the report. The 2-24h bucket is above 1.0 for both models because it covers Dec 8 and Christmas.

Per-series models on a volume-stratified sample of 8 cells (internet + call_in), 25-168h bucket:

| Model | MASE | Coverage | Backtest wall time |
|---|---|---|---|
| LightGBM global | **0.901** | 87.3% | 287 s for all 500 cells x 5 metrics |
| Seasonal naive | 1.066 | 88.0% | <1 s |
| Prophet | 1.249 | 88.4% | 9 s for 8 cells x 2 metrics |
| SARIMA(1,0,1)(1,1,1)24 | 1.509 | 89.0% | 185 s for 8 cells x 2 metrics |

Anomaly detection on the last 14 days (153 labelled anomalies, alert threshold 4.5):

| | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Event level | 73.8% | 67.3% | 0.70 | 10.4 alerts/day, 1.8 incidents per operator shift, mean time to detect 0.4 h |
| Point level, ensemble | 78.3% | 51.0% | 0.62 | PR-AUC 0.548 |
| Point level, LSTM-AE alone | 24.0% | 48.6% | 0.32 | PR-AUC 0.388 |
| Point level, residual alone | 17.8% | 30.7% | 0.23 | PR-AUC 0.212 |
| Point level, Isolation Forest alone | 5.2% | 27.7% | 0.09 | PR-AUC 0.159 |

Recall by type: outage 100%, congestion spike 100%, partial outage 73%, unscheduled event 63%, voice degradation 0%. The ensemble beats every single detector, which is the point of having three.


## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

```bash
nettraffic run                 # synthetic data, 500 cells x 61 days, full pipeline (~15 min on 4 cores)
```

```bash
nettraffic run --quick --no-stat-models   # 60 cells, ~2 min, good for a smoke test
```

```bash
nettraffic run --milan-dir data/milan --no-stat-models   # real Telecom Italia Milan files
```

```bash
uvicorn nettraffic.api.app:app --port 8000   # API over the batch outputs
```

```bash
python -m http.server 8080 --directory site   # dashboard at http://localhost:8080
```

```bash
pytest && ruff check .
```

Useful flags: `--tune-trials N` (Optuna), `--cells`, `--days`, `--no-mlflow`. MLflow logs to `sqlite:///mlflow.db`; browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Data

The real [Telecom Italia Milan dataset](https://doi.org/10.7910/DVN/EGZHFV) is the reference input (10-min grid, `sms_in/out`, `call_in/out`, `internet` per square). `nettraffic.data.milan` reads its daily `.txt` files and sums the per-country rows.

It has no anomaly labels, so by default the pipeline runs on a synthetic generator (`nettraffic.data.synthetic`) with the same schema, built to stress the same things the real data does:

| Property | How it is simulated |
|---|---|
| Daily / weekly seasonality in local time | Area-specific profiles (business, residential, university, stadium, rural), Europe/Rome wall clock |
| Holidays | Italian public holidays via `holidays`, festive SMS bursts |
| Zero-inflated quiet cells | Poisson counts on tiny means |
| Known events (not anomalies) | Stadium matches/concerts, passed to the model as a feature |
| **Labelled anomalies** | outage, partial outage, voice degradation, congestion spike, unscheduled crowd event |
| Data errors (not anomalies) | missing rows, long gaps, duplicates, negative values, post-gap catch-up dumps |

Labels are what make the detection numbers below measurable instead of anecdotal.

## How the pieces fit

```
src/nettraffic/
  data/            ingest (polars -> date-partitioned parquet), Milan loader, synthetic generator
  cleaning.py      dedupe, negatives, catch-up dumps, gap policy, hourly aggregation
  calendar.py      local-time calendar, holidays, scheduled events
  eda.py           STL seasonality strength, area x hour-of-week heatmaps, daily totals
  features.py      leakage-safe lags / rolling / profile features (direct multi-horizon)
  forecasting/     seasonal-naive, SARIMA, Prophet, global LightGBM (+Optuna), backtest, conformal PIs
  anomaly/         residual detector, Isolation Forest, LSTM autoencoder (PyTorch), tail model, ensemble, eval
  capacity.py      cells forecast to exceed capacity
  monitoring.py    error drift, retrain trigger, data health, alert load, operator feedback, Evidently
  export.py        compact JSON for the static dashboard
  pipeline.py      orchestration + CLI
  api/app.py       FastAPI: /forecast, /anomalies, /capacity, /health
site/              static dashboard (Plotly.js), served by Cloudflare
dags/              Airflow DAG (thin wrapper around the CLI)
.github/workflows/ CI (lint, tests, smoke run, Docker build) + daily dashboard refresh
wrangler.jsonc     Cloudflare static-assets config (serves site/)
```

### Decisions worth knowing

**Hourly modelling, 10-min storage.** Cleaning runs on the native 10-min grid; models run on hourly sums. Capacity is planned in hours, hourly series are far less noisy, and a 7-day horizon is 168 steps instead of 1008.

**UTC storage, local-time features.** No DST ambiguity in storage; human behaviour follows wall-clock time, so every calendar feature is derived after converting to `Europe/Rome`.

**Cleaning never invents traffic.** Gaps up to 30 min are interpolated; longer gaps stay missing (filling them would hide outages and bias training). Catch-up dumps (a row after a feed gap that carries the whole backlog) are spread back over the gap, otherwise they become a fake spike *and* a fake drop.

**Global LightGBM, direct multi-horizon.** One model per (metric, horizon group) across all cells. Lead `h` uses the model whose minimum lag `L >= h`: `L=1` for 1h, `L=24` for 2-24h, `L=168` for 25-168h. No recursive error build-up, 3 models per metric instead of 168, and a unit test proves features never see data inside the forbidden window.

**Honest intervals.** Normalised split-conformal: the score is `|y - yhat| / (yhat + s_cell)`, so width grows with expected traffic but keeps a floor on quiet cells. Fold *k* is calibrated only on folds *< k*; fold 0 exists only for calibration.

**SARIMA and Prophet run on a sample.** They fit one series at a time. They are evaluated on a volume-stratified sample of cells and compared with the global models on exactly those cells. Their wall time per series is the scaling argument for the global model.

**Anomaly scores are probabilities, not magic numbers.** Every detector's raw score is mapped to `-log10(P(score >= x | normal))` using its own held-out null distribution, with an exponential peaks-over-threshold tail so strong anomalies keep separating instead of saturating at `1/n`. A weighted mean of those surprisals is unit-consistent.

**The residual detector is standardised per cell.** Pooling residuals across cells let noisy quiet cells define the tail, which made a 3x spike on a busy cell look normal. Per-cell robust scale (MAD of out-of-sample residuals) fixed that: in a development run (500 cells) ensemble PR-AUC went from 0.36 to 0.49 with nothing else changed.

**Echo suppression.** The day-ahead model uses `t-24` lags, so yesterday's outage would create today's fake spike. A second pass replaces strong deviations with their expected value before recomputing expectations.

**Alert dedup.** Consecutive flagged hours (gap <= 1h) merge into one event; overlapping events on the same cell form one incident; an all-metric drop of >80% is `critical` (site outage).

## Known limitations

- Detection numbers are on synthetic labels. The ensemble weights and threshold were sanity-checked on the same detection window, so treat them as optimistic until validated on operator feedback (`data/alert_feedback.csv` hooks into monitoring).
- Voice degradation (calls down to 20-45% for a few hours) is the weakest class: on quiet cells it sits inside normal Poisson noise. A per-metric threshold or a longer-window detector would help.
- 61 days of history contain only a couple of holidays, so Christmas-week forecasts are weaker. Real deployments need at least a year for holiday effects.
- Statistical models use a fixed specification, not per-series order selection.
- The Docker image is built and health-checked in CI, not locally.

## Deployment

The dashboard is static: everything it shows is precomputed into `site/data/`, so there is no server to scale, nothing to cold-start, and nothing to pay for. Cloudflare serves `site/` as static assets (`wrangler.jsonc`) and redeploys on every push to `main`.

```
GitHub Actions (daily 03:17 UTC)                     Cloudflare
  install -> nettraffic run -> site/data/*.json  --push main-->  rebuild -> *.workers.dev
```

## Automated daily refresh

`.github/workflows/refresh-dashboard.yml` closes the loop from model to screen with nobody in it. Every day it:

1. Installs the package on a clean runner (CPU-only PyTorch, about 1.5 GB smaller than the CUDA wheel).
2. Runs the **whole** batch pipeline from scratch: ingest -> clean -> backtest -> retrain -> 7-day forecast with intervals -> anomaly detection -> capacity -> monitoring -> export.
3. Commits the new `site/data/` to `main` as `github-actions[bot]`, with `[skip ci]` so the data commit does not re-trigger CI.
4. Cloudflare picks up the push and redeploys. The run's metrics (`run_summary.json`, `detection_eval.json`, backtest CSVs) are kept as a workflow artifact for 14 days, so you can compare model quality across days.

Details that matter:

- **A new scenario every day.** On synthetic data the seed is the run date (`--seed 20261102`), so each refresh is a different month of traffic with different outages and spikes, not the same picture re-rendered. A manual run can pin a seed to reproduce a day exactly.
- **Two modes.** The scheduled run skips SARIMA/Prophet and Optuna (about 10 min on a 4-core runner). Run it manually with **full = true** (Actions -> Refresh dashboard -> Run workflow) to include the per-series benchmarks and 12 Optuna trials (about 15-20 min; 13 min on a 4-core machine during development).
- **Safe by construction.** `concurrency` stops two refreshes racing on `main`. The commit step exits cleanly when nothing changed. The job has a 75-minute timeout, and only `contents: write` permission.
- **Retrain is built in.** Every refresh retrains from scratch, and `monitoring.json` records whether the model lost to seasonal-naive for 2 consecutive folds (the retrain trigger). The Monitoring tab shows it.
- **Switching to real data** is one line: replace the `nettraffic run $ARGS` command with `nettraffic run --milan-dir <path> --no-stat-models` once the feed lands somewhere the runner can read.

If `main` gets branch protection, allow `github-actions[bot]` to push, or the commit step fails.

## CI

`.github/workflows/ci.yml` runs on every PR and every push to `main`: ruff lint and format check, the pytest suite, a 60-cell end-to-end pipeline smoke run, and a Docker build whose container must answer `/health` within 60 s.

The API image (`Dockerfile`, `docker-compose.yml`) serves the same batch outputs for on-demand queries:

| Endpoint | Purpose |
|---|---|
| `GET /forecast?cell_id=&metric=&horizon=` | point forecast + 90% interval, 1-168h |
| `GET /anomalies?cell_id=&metric=&min_severity=&since=&limit=` | ranked alert events |
| `GET /capacity?status=` | cells over / at risk of capacity |
| `GET /health`, `POST /admin/reload` | liveness and hot reload after a batch run |
