"""Climate Data Store and Atmosphere Data Store requests: ERA5-Land daily statistics, CAMS EAC4.

    one request per month (ERA5-Land: per month and UTC-offset group, cut to the cells' area; CAMS: global,
    surface-level fields and model level 60 as two requests) -> <cache>/raw/<request>.nc, or a folder of .nc
    files when the store sends a zip -> field() reads one band from the cached file and places it on the grid

The stores run one request at a time per user for these products (measured for ERA5-Land: ~43 s per tiny
request in sequence; month requests 3-8 min), so a month is fetched once and every field is read from disk.
Token: majortom.build.credentials.cds_token() (CDS_TOKEN, or `python -m majortom.build.credentials cds`).
"""
from __future__ import annotations

import calendar
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache_dir, read_netcdf

H = np.timedelta64(1, "h")
_locks: dict[Path, threading.Lock] = {}
_guard = threading.Lock()


def _client(url: str):
    from ecmwf.datastores import Client

    from ...build.credentials import cds_token
    return Client(url=url, key=cds_token(), progress=False)


def _fetch(source, request: dict, target: Path) -> Path:
    """Run one request into `target` (unless it is there); a zip becomes a folder of its files."""
    with _guard:
        lock = _locks.setdefault(target, threading.Lock())
    with lock:
        folder = target.with_suffix("")
        if target.exists() or folder.is_dir():
            return folder if folder.is_dir() else target
        partial = target.with_name(target.name + ".part")
        _client(source.options["url"]).retrieve(source.store, request, str(partial))
        if zipfile.is_zipfile(partial):
            with zipfile.ZipFile(partial) as z:
                z.extractall(folder)
            partial.unlink()
            return folder
        partial.rename(target)
        return target


# ERA5-Land daily statistics: days centred on each acquisition by a UTC offset (period "day") ---------------

def groups(source, acquired: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """UTC-offset group of each acquisition: 12 - its UTC hour, rounded to the source's offset step (3 h)."""
    hour = (acquired - acquired.astype("datetime64[D]")) / H
    step = source.options.get("offset_step", 1)
    return np.clip(step * np.round((12 - hour) / step), -12, 12).astype(int)


def days(source, centres: np.ndarray, lon: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The local date (at the group's offset) of each window centre: that day is centred on it within ±1.5 h."""
    return (centres + group * H).astype("datetime64[D]").astype(np.int64)


def period_start(source, day: np.ndarray, group: np.ndarray) -> np.ndarray:
    """UTC start of a group's day: local midnight at offset `group`, i.e. 00:00 - group hours."""
    return np.asarray(day).astype("datetime64[D]") - np.asarray(group) * H


def _month_file(source, month: pd.Timestamp, group: int) -> Path:
    return cache_dir(source.name) / "raw" / f"{month:%Y-%m}_utc{group:+03d}.nc"


def prepare(source, month: pd.Timestamp, group: int, bbox: tuple[float, float, float, float]) -> None:
    """Fetch one month of daily means for one offset group, cut to `bbox` (north, west, south, east)."""
    target = _month_file(source, month, group)
    target.parent.mkdir(parents=True, exist_ok=True)
    north, west, south, east = (round(v, 1) for v in bbox)
    request = {"variable": list(source.variables.values()), "year": f"{month:%Y}", "month": f"{month:%m}",
               "day": [f"{d:02d}" for d in range(1, calendar.monthrange(month.year, month.month)[1] + 1)],
               "daily_statistic": "daily_mean", "time_zone": f"utc{group:+03d}:00", "frequency": "1_hourly",
               "area": [north, west, south, east]}
    _fetch(source, request, target)


def field(source, variable: str, key) -> np.ndarray:
    """ERA5-Land: `key` is (day, group) -> that day's mean; CAMS: `key` is a time -> that 3-hourly step."""
    if source.period == "day":
        day, group = key
        date = np.datetime64(int(day), "D")
        path = _month_file(source, pd.Timestamp(date), group)
        found = path.with_suffix("") if path.with_suffix("").is_dir() else path
        if not found.exists():
            raise FileNotFoundError(f"{path} not fetched: run prepare() for its month and group first")
        return read_netcdf(found, source.options["files"][variable], date.astype("datetime64[ns]"), source.grid)
    return _cams(source, variable, pd.Timestamp(key))


# CAMS EAC4: 3-hourly fields (period "hour"), fetched a month at a time on first use ------------------------

def _cams(source, variable: str, time: pd.Timestamp) -> np.ndarray:
    surface = variable in source.options["surface"]
    month = time.to_period("M").start_time
    names = [v for v in source.variables if (v in source.options["surface"]) == surface]
    last = calendar.monthrange(month.year, month.month)[1]
    request = {"variable": [source.variables[v] for v in names], "date": f"{month:%Y-%m}-01/{month:%Y-%m}-{last:02d}",
               "time": [f"{h:02d}:00" for h in range(0, 24, 3)], "data_format": "netcdf_zip"}
    if surface:
        request["model_level"] = [source.options["model_level"]]
    target = cache_dir(source.name) / "raw" / f"{month:%Y-%m}_{'ml' if surface else 'sl'}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    path = _fetch(source, request, target)
    return read_netcdf(path, source.options["files"][variable], time.to_datetime64().astype("datetime64[ns]"), source.grid)
