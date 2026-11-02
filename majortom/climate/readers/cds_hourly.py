"""Climate Data Store hourly ERA5-Land, cut to an area: the fill south of 57.1°S (docs/CLIMATE.md 6.1).

    field(var, hour) -> the span holding `hour`: its week (Monday-Sunday) cut at month ends, because the store
      requests every combination of the year, month and day lists (a week across two months would ask for both
      months' days) -> one request per span and variable group (source.options["groups"]): at most
      11 variables x 7 days x 24 h = 1,848 fields, the largest size the store accepted (8,184 was refused)
      -> <cache>/hourly/<first day>_<days>d_g<group>_<area>.nc
      -> the hour's band, placed on source.grid
    Accumulated variables are totals since 00 UTC, restarting at 01 UTC (as Earth Data Hub's copy): field()
    returns the hour's amount, this hour minus the previous, except at 01 UTC.
    stream() reads hours in time order, so when a span is fetched, the group's span files that end more than a
    day before it starts are removed: one group holds two spans on disk (a week of group 1: 1.29 GB), not the
    whole record.
"""
from __future__ import annotations

import calendar
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache_dir, read_netcdf
from .queue import _fetch

_lock = threading.Lock()


def group_of(source, variable: str) -> int:
    """The variable group (one request) that holds `variable`."""
    return next(g for g, names in source.options["groups"].items() if variable in names)


def span(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day of the request holding `day`: its Monday-Sunday week, cut at the month's ends."""
    day = pd.Timestamp(day).normalize()
    monday = day - pd.Timedelta(days=day.weekday())
    first_of_month = day.replace(day=1)
    last_of_month = day.replace(day=calendar.monthrange(day.year, day.month)[1])
    return max(monday, first_of_month), min(monday + pd.Timedelta(days=6), last_of_month)


def _tag(source) -> str:
    return "_".join(f"{v:g}" for v in source.options["area"])


def request(source, first: pd.Timestamp, last: pd.Timestamp, group: int) -> dict:
    """The store request for one span and group (one month, its days, every hour, cut to the area)."""
    names = source.options["groups"][group]
    return {"variable": [source.variables[v] for v in names], "year": f"{first:%Y}", "month": f"{first:%m}",
            "day": [f"{d:%d}" for d in pd.date_range(first, last, freq="D")],
            "time": [f"{h:02d}:00" for h in range(24)], "area": list(source.options["area"]),
            "data_format": "netcdf", "download_format": "unarchived"}


def _span_file(source, day: pd.Timestamp, group: int) -> Path:
    """The cached file of the span holding `day` for `group`, fetched on first use; older spans are removed."""
    first, last = span(day)
    folder = cache_dir(source.name) / "hourly"
    target = folder / f"{first:%Y-%m-%d}_{(last - first).days + 1}d_g{group}_{_tag(source)}.nc"
    if not target.exists():
        folder.mkdir(parents=True, exist_ok=True)
        _fetch(source, request(source, first, last, group), target)
        _remove_before(folder, group, first - pd.Timedelta(days=1), source)
    return target


def _remove_before(folder: Path, group: int, day: pd.Timestamp, source) -> None:
    """Remove the group's span files that end before `day` (the previous span stays: 00 UTC needs 23 UTC)."""
    with _lock:
        for f in folder.glob(f"*_g{group}_{_tag(source)}.nc"):
            first = pd.Timestamp(f.name[:10])
            days = int(f.name[11:].split("d_", 1)[0])
            if first + pd.Timedelta(days=days) <= day:
                f.unlink(missing_ok=True)


def _at(source, variable: str, hour: pd.Timestamp) -> np.ndarray:
    path = _span_file(source, hour.normalize(), group_of(source, variable))
    return read_netcdf(path, source.options["files"][variable], hour.to_datetime64().astype("datetime64[ns]"),
                       source.grid)


def field(source, variable: str, hour) -> np.ndarray:
    """One field on source.grid (NaN at sea) at `hour`; accumulated variables: the amount over the hour ending at
    `hour` (the total since 00 UTC, less the previous hour's, except at 01 UTC where the total restarts)."""
    hour = pd.Timestamp(hour)
    f = _at(source, variable, hour)
    if variable in source.accumulated and hour.hour != 1:
        f = f - _at(source, variable, hour - pd.Timedelta(hours=1))
    return f


def valid_range(source) -> tuple[pd.Timestamp, pd.Timestamp]:
    """ERA5-Land hourly starts in 1950; the store publishes with a delay, so it ends with the last complete month."""
    last = (pd.Timestamp.now().normalize() - pd.DateOffset(months=1)).to_period("M").end_time.floor("h")
    return pd.Timestamp("1950-01-01"), last
