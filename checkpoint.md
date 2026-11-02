# Checkpoint

**Last updated:** 2026-11-02 · **State:** deployed, CI green, daily refresh working

## Where things stand

| Item | Status |
|---|---|
| Pipeline (ingest → clean → forecast → detect → capacity → monitoring → export) | Done, on `main` |
| Tests | 29 passing, ruff clean |
| CI (`ci.yml`) | Green on PR #1 and on `main` (lint, tests, smoke run, Docker build + `/health`) |
| Dashboard | Live at https://network-traffic-forecasting.mrwaanawad.workers.dev, Cloudflare production branch = `main` |
| Daily refresh (`refresh-dashboard.yml`) | First run passed (manual, 10.5 min), committed `ee56c7e` to `main`. Scheduled daily at 03:17 UTC |
| PR #1 | Merged (`0ecd35d`) |

## Latest numbers (refresh run 2026-11-02, 500 cells, seed 20261102, no SARIMA/Prophet/Optuna)

| Metric | Value |
|---|---|
| LightGBM MASE vs seasonal-naive | 0.917 vs 1.091 |
| 90% interval coverage | 89.4% |
| Alert precision / recall / F1 (per event) | 63.8% / 56.7% / 0.60 |
| Alerts per day | 10.6 |
| Mean time to detect | 0.14 h |

Reference run (seed 42, full mode with tuning): MASE 0.909, coverage 89.1%, event F1 0.70. Detection quality changes from one seed to the next; F1 around 0.6–0.7 is the honest range so far.

## Open items

1. **Voice degradation recall is 0%.** Call drops to 20–45% on quiet cells sit inside normal noise. Next step: per-metric thresholds, or a detector over a longer window.
2. **Detection numbers are optimistic.** The ensemble weights were checked on the same window they're scored on. Next step: tune on one week and score on another.
3. **Real Milan data has not been run.** The loader is tested on a sample file only. To switch, change the refresh command to `nettraffic run --milan-dir <path>`.
4. **The Claude GitHub App can't start workflows** (403). To let Claude start runs, give the app write access to Actions.
5. **Alerts during scheduled stadium events:** 6–7 per run, even though events are a model feature.

## Decisions log

- **Synthetic data by default:** Harvard Dataverse, Kaggle and Hugging Face are blocked from the dev container, and real data has no labels.
- **PyTorch instead of TensorFlow** for the LSTM autoencoder: lighter install, matches the main stack.
- **Static dashboard + batch jobs, no live API hosting:** you skipped Render. The FastAPI app and Docker image exist and are tested in CI.
- **`main` was created at the first commit**, so PR #1 covered only the later 3 commits; there was no history rewrite.
- **The daily refresh skips SARIMA/Prophet/Optuna** to stay around 10 minutes. Full mode is available on manual dispatch.

## How to resume

```bash
pip install -e ".[all]"
```

```bash
pytest && ruff check .
```

```bash
nettraffic run --quick --no-stat-models
```
