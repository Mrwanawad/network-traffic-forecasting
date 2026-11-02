"""Loader for the real Telecom Italia "Telecommunications - SMS, Call, Internet - MI"
dataset (Harvard Dataverse, doi:10.7910/DVN/EGZHFV).

File format (one file per day, tab separated, no header)::

    square_id  time_interval(ms epoch, UTC)  country_code  sms_in  sms_out  call_in  call_out  internet

Rows are split by country code; we sum them per (square, interval). Empty
fields mean "no activity recorded", so they count as 0 inside the sum. A square
with *no* row at all for an interval stays missing and is handled by cleaning
(interpolated if the gap is short, left missing if long).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nettraffic.config import METRICS

RAW_COLUMNS = ["cell_id", "time_ms", "country_code", *METRICS]


def scan_milan(paths: list[Path] | Path) -> pl.LazyFrame:
    if isinstance(paths, Path):
        paths = sorted(paths.glob("sms-call-internet-mi-*.txt")) if paths.is_dir() else [paths]
    if not paths:
        raise FileNotFoundError("no Milan files found (expected sms-call-internet-mi-*.txt)")
    frames = [
        pl.scan_csv(
            p,
            separator="\t",
            has_header=False,
            new_columns=RAW_COLUMNS,
            schema_overrides={"cell_id": pl.Int64, "time_ms": pl.Int64, "country_code": pl.Int64}
            | {m: pl.Float64 for m in METRICS},
        )
        for p in paths
    ]
    return (
        pl.concat(frames)
        .with_columns(pl.from_epoch("time_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("ts"))
        .group_by("cell_id", "ts")
        .agg([pl.col(m).sum().alias(m) for m in METRICS])
        .sort("cell_id", "ts")
    )


def load_milan(paths: list[Path] | Path) -> pl.DataFrame:
    return scan_milan(paths).collect()
