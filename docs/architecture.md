# Network Traffic Forecasting & Anomaly Detection — Architecture

## Scenario
A telecom operator needs to (1) forecast traffic (calls, SMS, internet) per cell/area for capacity planning, and (2) detect abnormal drops or spikes (outages, congestion, events) before customers complain. The system forecasts many time series at once, compares reality against the forecast, and raises ranked alerts to the network operations team.

## Input
- Time series of traffic activity per cell/grid area at fixed intervals (e.g. 10 min): call-in, call-out, SMS-in, SMS-out, internet.
- Calendar features: hour, weekday, holidays, special events.
- Optional: cell metadata (area type, location).
- Labeled anomalies (where available) for evaluation.

## Output
- Forecasts per cell: point forecast + prediction interval for next horizon (e.g. 1h, 24h, 7d).
- Anomaly events: cell_id, timestamp, metric, observed vs. expected, anomaly score, severity.
- Capacity report: cells forecast to exceed capacity thresholds.
- Dashboard: traffic maps/heatmaps, forecast vs. actual, active alerts.
- REST endpoints: `/forecast`, `/anomalies`.

## Architecture

```
 Raw traffic files / feed
        |
        v
 [Ingestion]  polars -> parquet partitioned by date
        |
        v
 [Validation & Cleaning]
   - regularize timestamps to fixed grid, timezone + DST handling
   - fill gaps (short: interpolate; long: mark missing, not fake)
   - zero-inflated cells, duplicate rows, counter resets
   - separate true events from data errors
        |
        v
 [EDA]  seasonality (daily/weekly), STL decomposition, area/hour heatmaps
        |
        v
 [Feature Engineering]  lags, rolling stats, calendar, holiday flags,
                        cell-level aggregates (tsfresh for candidate features)
        |
        +--> [Forecasting]
        |      baselines: seasonal-naive
        |      statistical: SARIMA (statsmodels), Prophet
        |      ML: global LightGBM across cells (+ Optuna)
        |      rolling-origin backtesting -> MLflow
        |
        +--> [Anomaly Detection]
               residual-based: |actual - forecast| vs. prediction interval / robust z-score
               unsupervised: Isolation Forest on feature windows
               deep: LSTM autoencoder (PyTorch, reconstruction error)
               ensemble score -> severity + alert dedup
        |
        v
 [Serving]  scheduled batch (Airflow) writes forecasts & alerts;
            FastAPI for on-demand queries; Docker
        |
        v
 [Dashboard]  Grafana or Streamlit
        |
        v
 [Monitoring]  forecast error per cell over time, alert volume,
               alert precision from operator feedback, data freshness -> retrain
```

## Tools Used
- Data: polars, pandas, NumPy, pyarrow/parquet
- EDA & viz: Matplotlib, Seaborn, Plotly Express
- Time series: statsmodels (SARIMA, STL), Prophet, sktime, tsfresh
- ML: LightGBM, scikit-learn (IsolationForest, metrics, TimeSeriesSplit)
- Deep learning: PyTorch (LSTM autoencoder) — chosen over TensorFlow/Keras: lighter CPU install, same model
- Tuning: Optuna
- MLOps: MLflow, Airflow, Docker, Evidently
- Serving & UI: FastAPI, Pydantic, Grafana or Streamlit
- Quality: pytest, ruff

## Tools Not Used (and why)
- Random train/test split or plain K-fold: leaks future data into training.
- Transformers / foundation time-series models: heavier, harder to explain; global GBM + statistical models are strong baselines for this scope.
- One deep model per cell: does not scale to thousands of series; use a global model.
- Spark / Kafka: batch scheduled forecasting is enough at this scope.
- LLMs: not needed for numeric forecasting.

## Evaluation Metrics
- Forecasting: MAE, RMSE, sMAPE, MASE (scale-free across cells); prediction-interval coverage.
- Always compared against seasonal-naive baseline.
- Validation: rolling-origin backtest across multiple cutoffs; report per-horizon error.
- Anomaly detection (labeled data): precision, recall, F1, PR-AUC; time-to-detect.
- Anomaly detection (unlabeled): alert rate per day, operator-confirmed precision.
- Slices: busy vs. quiet cells, weekdays vs. weekends, holidays.

## Production Monitoring
- Forecast error drift per cell (rolling MASE).
- Data freshness and missing-interval rate.
- Alert volume and alert fatigue (alerts per operator per shift).
- Feedback loop: operator confirms/dismisses alerts -> labels for threshold tuning.
- Retrain trigger: error above baseline for consecutive windows, or seasonality change (e.g. Ramadan, holidays).

## Deployment Plan — Cloudflare Pages (static dashboard)
**Why Cloudflare Pages:** forecasting and anomaly detection run as scheduled batch jobs, so the UI only needs to display precomputed results. A static site is instant, never sleeps, has no server cost, and handles maps/charts smoothly. (Vercel works the same way; Cloudflare Pages chosen for unlimited static bandwidth and zero cold start.)

**What gets deployed:**
- Static dashboard (HTML + JS, Plotly.js / ECharts, optional Leaflet map):
  1. Traffic heatmap by area and hour.
  2. Cell explorer: pick a cell -> actual vs. forecast with prediction interval.
  3. Alerts table: anomalies ranked by severity, click to see the window.
  4. Model comparison: seasonal-naive vs. SARIMA vs. Prophet vs. LightGBM (MASE, sMAPE per horizon).
  5. Capacity report: cells forecast to exceed threshold.

**Data flow:**
- Python pipeline (run locally or GitHub Actions on schedule) -> writes compact JSON / parquet-to-JSON exports into `site/data/`.
- Commit / push -> Cloudflare Pages auto-builds and redeploys.
- Keep exported files small: top-N cells, downsampled history, all alerts.

**Optional live API:** if on-demand forecasting is needed, the same FastAPI `/forecast` and `/anomalies` service can be added on Render later; the dashboard stays static.

**Limits to know:** no server-side Python on Pages; everything shown must be precomputed. Always on, no warm-up needed.
