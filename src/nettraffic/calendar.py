"""Calendar features computed in *local* time.

Data is stored in UTC (no DST ambiguity, regular grid); human behaviour follows
local wall-clock time, so every seasonal feature is derived after converting to
the operator's timezone. This keeps the 08:00 commute peak at hour=8 on both
sides of a DST switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd

EVE_DAYS = ("12-24", "12-31")


@dataclass(frozen=True)
class ScheduledEvent:
    """A *known* event (match, concert, fair). It is a feature, not an anomaly."""

    name: str
    start: pd.Timestamp  # UTC
    hours: int
    cell_ids: tuple[int, ...]


@dataclass
class CalendarFeatures:
    hour: np.ndarray
    dow: np.ndarray
    hour_of_week: np.ndarray
    is_weekend: np.ndarray
    is_holiday: np.ndarray
    is_eve: np.ndarray  # Christmas Eve / New Year's Eve: atypical evenings, not public holidays
    local_date: np.ndarray = field(repr=False)

    @property
    def is_offday(self) -> np.ndarray:
        return self.is_weekend | self.is_holiday


def holiday_dates(years: list[int], country: str) -> set[pd.Timestamp]:
    cal = holidays.country_holidays(country, years=years)
    return {pd.Timestamp(d) for d in cal}


def calendar_features(index: pd.DatetimeIndex, tz: str, country: str) -> CalendarFeatures:
    local = index.tz_convert(tz)
    years = sorted({*local.year.tolist()})
    hol = holiday_dates(years, country)
    local_date = local.normalize().tz_localize(None)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    return CalendarFeatures(
        hour=hour,
        dow=dow,
        hour_of_week=dow * 24 + hour,
        is_weekend=dow >= 5,
        is_holiday=np.isin(local_date, list(hol)) if hol else np.zeros(len(index), bool),
        is_eve=np.isin(local.strftime("%m-%d"), list(EVE_DAYS)),
        local_date=local_date.to_numpy(),
    )


def event_matrix(index: pd.DatetimeIndex, cells: np.ndarray, events: list[ScheduledEvent]) -> np.ndarray:
    """(n_cells, n_times) bool: cell is inside a scheduled event window."""
    out = np.zeros((len(cells), len(index)), dtype=bool)
    pos = {int(c): i for i, c in enumerate(cells)}
    for ev in events:
        end = ev.start + ev.hours * pd.Timedelta(hours=1)
        tmask = (index >= ev.start) & (index < end)
        for c in ev.cell_ids:
            if c in pos:
                out[pos[c], tmask] = True
    return out


def events_to_frame(events: list[ScheduledEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"name": e.name, "start": e.start, "hours": e.hours, "cell_ids": list(e.cell_ids)} for e in events]
    )


def events_from_frame(df: pd.DataFrame) -> list[ScheduledEvent]:
    return [
        ScheduledEvent(
            name=str(r["name"]),
            start=pd.Timestamp(r["start"]).tz_convert("UTC"),
            hours=int(r["hours"]),
            cell_ids=tuple(int(c) for c in r["cell_ids"]),
        )
        for _, r in df.iterrows()
    ]
