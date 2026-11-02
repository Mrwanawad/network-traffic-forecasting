"""Airflow DAG: hourly ingest, daily retrain/forecast, alert publish.

Thin by design - every task calls the same `nettraffic` functions the CLI uses,
so the DAG holds scheduling only, no business logic. Requires apache-airflow>=2.7
in the scheduler image (not a dependency of this package).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "network-analytics",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="network_traffic_forecasting",
    description="Clean -> backtest -> forecast -> detect -> export",
    schedule="15 2 * * *",  # daily, after the previous day's files land
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["forecasting", "anomaly-detection"],
) as dag:
    run_pipeline = BashOperator(
        task_id="run_pipeline",
        # point --milan-dir at the landing folder of the real feed
        bash_command="nettraffic run --milan-dir {{ var.value.traffic_landing_dir }} --no-stat-models",
    )
    publish_site = BashOperator(
        task_id="publish_dashboard",
        # Cloudflare Pages redeploys on push; credentials come from the worker env
        bash_command=(
            "cd {{ var.value.repo_dir }} && git add site/data && "
            "git commit -m 'data: refresh dashboard {{ ds }}' && git push"
        ),
    )
    run_pipeline >> publish_site
