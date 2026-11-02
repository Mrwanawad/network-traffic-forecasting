"""Dense (cell x time x metric) representation of a regular time-series panel.

Telecom traffic is a *regular* grid: every cell reports every interval. Keeping
it as a dense array makes seasonal lags, rolling windows and backtest masking
plain array slicing instead of groupby/shift gymnastics on long frames.
Missing observations are NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Panel:
    values: np.ndarray  # float32 (n_cells, n_times, n_metrics), NaN = missing
    cells: np.ndarray  # int64 (n_cells,)
    index: pd.DatetimeIndex  # tz-aware UTC, regular
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        c, t, m = self.values.shape
        if c != len(self.cells) or t != len(self.index) or m != len(self.metrics):
            raise ValueError(
                f"shape {self.values.shape} does not match cells={len(self.cells)}, "
                f"times={len(self.index)}, metrics={len(self.metrics)}"
            )
        if self.index.tz is None:
            raise ValueError("Panel index must be tz-aware (UTC)")

    # ------------------------------------------------------------------ access
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_times(self) -> int:
        return len(self.index)

    def metric(self, name: str) -> np.ndarray:
        """(n_cells, n_times) view for one metric."""
        return self.values[:, :, self.metrics.index(name)]

    def cell_pos(self, cell_id: int) -> int:
        pos = np.flatnonzero(self.cells == cell_id)
        if pos.size == 0:
            raise KeyError(f"unknown cell_id {cell_id}")
        return int(pos[0])

    def time_pos(self, ts: pd.Timestamp) -> int:
        return int(self.index.get_loc(ts))

    def slice_time(self, start: int, stop: int) -> Panel:
        return Panel(
            values=self.values[:, start:stop, :],
            cells=self.cells,
            index=self.index[start:stop],
            metrics=self.metrics,
        )

    def subset_cells(self, cell_ids: np.ndarray) -> Panel:
        pos = np.array([self.cell_pos(int(c)) for c in cell_ids])
        return Panel(self.values[pos], self.cells[pos], self.index, self.metrics)

    # --------------------------------------------------------------- convert
    def to_long(self) -> pd.DataFrame:
        c, t, _ = self.values.shape
        df = pd.DataFrame(
            {
                "cell_id": np.repeat(self.cells, t),
                "ts": np.tile(self.index, c),
            }
        )
        for i, m in enumerate(self.metrics):
            df[m] = self.values[:, :, i].reshape(-1)
        return df

    @classmethod
    def from_long(cls, df: pd.DataFrame, metrics: tuple[str, ...], freq: str = "1h") -> Panel:
        cells = np.sort(df["cell_id"].unique()).astype(np.int64)
        ts = pd.to_datetime(df["ts"], utc=True)
        index = pd.date_range(ts.min(), ts.max(), freq=freq, tz="UTC")
        values = np.full((len(cells), len(index), len(metrics)), np.nan, dtype=np.float32)
        ci = np.searchsorted(cells, df["cell_id"].to_numpy())
        ti = index.get_indexer(ts)
        if (ti < 0).any():
            raise ValueError("timestamps not aligned to the panel frequency")
        for k, m in enumerate(metrics):
            values[ci, ti, k] = df[m].to_numpy(dtype=np.float32)
        return cls(values, cells, index, metrics)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_long().to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path, metrics: tuple[str, ...]) -> Panel:
        return cls.from_long(pd.read_parquet(path), metrics)
