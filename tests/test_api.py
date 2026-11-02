from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    ts = pd.date_range("2024-01-01", periods=168, freq="1h", tz="UTC")
    pd.DataFrame(
        {
            "cell_id": 42,
            "metric": "internet",
            "ts": ts,
            "lead": np.arange(1, 169),
            "yhat": 100.0,
            "lo": 80.0,
            "hi": 120.0,
        }
    ).to_parquet(tmp_path / "forecasts.parquet")
    pd.DataFrame(
        [
            {
                "event_id": 1,
                "incident_id": 1,
                "cell_id": 42,
                "metric": "internet",
                "start": ts[0],
                "end": ts[3],
                "peak_ts": ts[1],
                "observed": 2.0,
                "expected": 100.0,
                "deviation_pct": -98.0,
                "score": 7.1,
                "direction": "drop",
                "severity": "critical",
                "n_hours": 3,
                "incident_metrics": 5,
            },
            {
                "event_id": 2,
                "incident_id": 2,
                "cell_id": 43,
                "metric": "sms_in",
                "start": ts[10],
                "end": ts[11],
                "peak_ts": ts[10],
                "observed": 300.0,
                "expected": 100.0,
                "deviation_pct": 200.0,
                "score": 4.6,
                "direction": "spike",
                "severity": "low",
                "n_hours": 1,
                "incident_metrics": 1,
            },
        ]
    ).to_parquet(tmp_path / "anomalies.parquet")
    pd.DataFrame(
        [
            {
                "cell_id": 42,
                "metric": "internet",
                "peak_ts": ts[5],
                "peak_forecast": 130.0,
                "max_upper": 150.0,
                "capacity": 100.0,
                "utilisation": 1.3,
                "utilisation_upper": 1.5,
                "status": "over_capacity",
                "capacity_source": "metadata",
                "peak_upper": 150.0,
            }
        ]
    ).to_parquet(tmp_path / "capacity.parquet")
    monkeypatch.setenv("NETTRAFFIC_ARTIFACTS", str(tmp_path))
    from nettraffic.api.app import app

    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["loaded"]["forecasts"] == 168


def test_forecast_horizon_and_404(client) -> None:
    r = client.get("/forecast", params={"cell_id": 42, "metric": "internet", "horizon": 24})
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 24 and body["points"][0]["lo"] <= body["points"][0]["yhat"]
    assert client.get("/forecast", params={"cell_id": 999}).status_code == 404
    assert client.get("/forecast", params={"cell_id": 42, "horizon": 500}).status_code == 422


def test_anomaly_filters(client) -> None:
    assert client.get("/anomalies").json()["count"] == 2
    crit = client.get("/anomalies", params={"min_severity": "high"}).json()
    assert crit["count"] == 1 and crit["items"][0]["severity"] == "critical"
    assert client.get("/anomalies", params={"metric": "sms_in"}).json()["items"][0]["cell_id"] == 43


def test_capacity(client) -> None:
    r = client.get("/capacity", params={"status": "over_capacity"}).json()
    assert r[0]["cell_id"] == 42
