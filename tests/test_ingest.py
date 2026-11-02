from __future__ import annotations

import pandas as pd
import polars as pl

from nettraffic.data.ingest import scan_partitioned, write_partitioned
from nettraffic.data.milan import load_milan


def test_milan_loader_sums_country_rows(tmp_path) -> None:
    t0 = 1383260400000  # 2013-11-01 00:00 Europe/Rome in ms (UTC epoch)
    rows = [
        f"1\t{t0}\t39\t1.5\t\t2.0\t\t10.0",
        f"1\t{t0}\t0\t0.5\t1.0\t\t\t5.0",
        f"2\t{t0 + 600_000}\t39\t\t\t\t\t3.0",
    ]
    (tmp_path / "sms-call-internet-mi-2013-11-01.txt").write_text("\n".join(rows) + "\n")
    df = load_milan(tmp_path).sort("cell_id")
    first = df.row(0, named=True)
    assert first["sms_in"] == 2.0 and first["internet"] == 15.0 and first["sms_out"] == 1.0
    assert str(df["ts"].dtype.time_zone) == "UTC"
    assert df.height == 2


def test_partitioned_roundtrip(tmp_path) -> None:
    ts = pd.date_range("2024-01-01 22:00", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"cell_id": 1, "ts": ts, "internet": range(6)})
    written = write_partitioned(df, tmp_path)
    assert {p.parent.name for p in written} == {"date=2024-01-01", "date=2024-01-02"}
    back = scan_partitioned(tmp_path, start="2024-01-02").collect()
    assert back.height == 4 and back["internet"].to_list() == [2, 3, 4, 5]
    assert isinstance(back, pl.DataFrame)


def test_overwrite_drops_stale_partitions(tmp_path) -> None:
    old = pd.DataFrame(
        {"cell_id": 1, "ts": pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC"), "internet": 1}
    )
    new = pd.DataFrame(
        {"cell_id": 2, "ts": pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC"), "internet": 2}
    )
    write_partitioned(old, tmp_path)
    write_partitioned(new, tmp_path, overwrite=True)
    back = scan_partitioned(tmp_path).collect()
    assert back["cell_id"].unique().to_list() == [2] and back.height == 24
