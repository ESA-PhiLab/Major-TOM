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

def groups(source, acquired: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
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


def _month_file(source, month: pd.Timestamp, group: int, bbox: tuple | None = None) -> Path:
    """The cached month for a group; requests are cut to an area, so the area is part of the name."""
    folder = cache_dir(source.name) / "raw"
    if bbox is None:                                       # reading: the largest area fetched for this month
        found = _cached(folder, month, group)
        if not found:
            raise FileNotFoundError(f"no cached file for {month:%Y-%m} utc{group:+03d} in {folder}: run prepare()")
        return max(found, key=lambda fa: (fa[1][0] - fa[1][2]) * (fa[1][3] - fa[1][1]))[0]
    area = "_".join(f"{v:g}" for v in bbox)
    return folder / f"{month:%Y-%m}_utc{group:+03d}_{area}.nc"


def _cached(folder: Path, month: pd.Timestamp, group: int) -> list[tuple[Path, tuple[float, ...]]]:
    """Cached files (or unzipped folders) for a month and group, with their areas (north, west, south, east)."""
    out = []
    for f in folder.glob(f"{month:%Y-%m}_utc{group:+03d}_*"):
        if f.name.endswith(".part"):
            continue
        area = f.name.removesuffix(".nc").split("_", 2)[2]
        out.append((f, tuple(float(v) for v in area.split("_"))))
    return out


def prepare(source, month: pd.Timestamp, group: int, bbox: tuple[float, float, float, float]) -> None:
    """Fetch one month of daily means for one offset group, cut to `bbox` (north, west, south, east)."""
    north, west, south, east = (round(v, 1) for v in bbox)
    for _, (n, w, s, e) in _cached(cache_dir(source.name) / "raw", month, group):
        if n >= north and w <= west and s <= south and e >= east:
            return                                           # an earlier run fetched an area that covers this one
    target = _month_file(source, month, group, (north, west, south, east))
    target.parent.mkdir(parents=True, exist_ok=True)
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
        found = _month_file(source, pd.Timestamp(date), group)
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


# Monthly means (period "month"): ERA5 and ERA5-Land, one request per year ------------------------------------

def valid_months(source) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First month published, and the last complete month assumed published (the stores lag by 2-3 months)."""
    last = (pd.Timestamp.now().normalize() - pd.DateOffset(months=3)).to_period("M").start_time
    return pd.Timestamp(source.options["first_month"]), last


def month_field(source, variable: str, month: pd.Timestamp) -> np.ndarray:
    """One monthly mean (global, on source.grid) from that year's cached request.

    CAMS: surface gases (model level 60) and single-level fields come as two requests per year.
    """
    year = {"year": [f"{month:%Y}"], "month": [f"{m:02d}" for m in range(1, 13)]}
    if "surface" in source.options:
        surface = variable in source.options["surface"]
        names = [v for v in source.variables if (v in source.options["surface"]) == surface]
        request = {"product_type": [source.options["product_type"]], "variable": [source.variables[v] for v in names],
                   **year, "data_format": "netcdf_zip"}
        if surface:
            request["model_level"] = [source.options["model_level"]]
        target = cache_dir(source.name) / "monthly" / f"{source.store}_{month:%Y}_{'ml' if surface else 'sl'}.zip"
    else:
        request = {"product_type": [source.options["product_type"]], "variable": list(source.variables.values()),
                   **year, "time": ["00:00"], "data_format": "netcdf", "download_format": "unarchived"}
        target = cache_dir(source.name) / "monthly" / f"{source.store}_{month:%Y}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    path = _fetch(source, request, target)
    when = pd.Timestamp(month).to_datetime64().astype("datetime64[ns]")
    return read_netcdf(path, source.options["files"][variable], when, source.grid)


def land_pixels(source) -> np.ndarray | None:
    """ERA5 monthly means share ERA5's 0.25° grid: its land mask comes from the ARCO store (see readers/zarr.py)."""
    if "land_from" not in source.options:
        return None
    from ..sources import SOURCES
    from . import zarr
    return zarr.land_pixels(SOURCES[source.options["land_from"]])
