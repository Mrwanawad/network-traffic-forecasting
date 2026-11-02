"""Raw ingestion: any source -> parquet partitioned by (UTC) date.

Date partitioning gives cheap incremental loads (append a new day, re-read only
the window you need) and predicate pushdown on `date`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import polars as pl


def write_partitioned(df: pl.DataFrame | pd.DataFrame, root: Path, overwrite: bool = False) -> list[Path]:
    """Write one parquet file per UTC date.

    overwrite=True removes the whole dataset first. Batch runs need it: a run with a
    shorter date range would otherwise leave stale partitions from an older run
    behind, and the reader would silently mix two datasets.
    """
    if overwrite and root.exists():
        shutil.rmtree(root)
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    df = df.with_columns(pl.col("ts").dt.convert_time_zone("UTC").dt.date().alias("date"))
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (day,), part in df.partition_by("date", as_dict=True, maintain_order=True).items():
        out = root / f"date={day.isoformat()}" / "part-0.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.drop("date").write_parquet(out)
        written.append(out)
    return written


def scan_partitioned(root: Path, start: str | None = None, end: str | None = None) -> pl.LazyFrame:
    """Lazy scan with optional inclusive date bounds (YYYY-MM-DD)."""
    lf = pl.scan_parquet(root / "**" / "*.parquet", hive_partitioning=True)
    if start:
        lf = lf.filter(pl.col("date") >= pl.lit(start).str.to_date())
    if end:
        lf = lf.filter(pl.col("date") <= pl.lit(end).str.to_date())
    return lf
