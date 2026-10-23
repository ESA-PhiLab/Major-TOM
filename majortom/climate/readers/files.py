"""Sources published as files: NOAA CPC Global Unified (yearly netCDF), GPM IMERG (half-hourly HDF5).

    a field needs a file -> <cache>/<file> (downloaded once, resumed if cut) -> one band -> the global grid

NOAA CPC (period "day"): a day's 24 hours end at a different UTC hour per country (CPC's end-of-day map); each
cell's group is its day start in hours from 00Z of the labelled date, and a daily step takes the CPC day holding
the step's window centre, i.e. the day overlapping its 24 hours most.
IMERG (period "hour"): precipitation in mm/hr per half hour; the hour ending at H is the mean of the two
half-hours [H - 1 h, H - 30 min) and [H - 30 min, H), in mm. Needs the Earthdata token, with the GES DISC
application approved on the Earthdata profile.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine

from . import cache_dir, place, read_netcdf

H = np.timedelta64(1, "h")
_locks: dict[Path, threading.Lock] = {}
_guard = threading.Lock()


def _download(url: str, target: Path, headers: dict | None = None) -> Path:
    """`url` into `target` once; a cut download resumes from its .part file."""
    import requests

    with _guard:
        lock = _locks.setdefault(target, threading.Lock())
    with lock:
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        for _ in range(5):
            have = partial.stat().st_size if partial.exists() else 0
            ranged = {**(headers or {}), **({"Range": f"bytes={have}-"} if have else {})}
            try:
                with requests.get(url, headers=ranged, stream=True, timeout=120) as r:
                    if r.status_code == 416:                       # nothing left to fetch
                        break
                    r.raise_for_status()
                    with open(partial, "ab" if have and r.status_code == 206 else "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                break
            except requests.RequestException:
                continue
        partial.rename(target)
        return target


# NOAA CPC Global Unified: daily, yearly files ------------------------------------------------------------------

def end_of_day(source) -> np.ndarray:
    """CPC's map of the hour (UTC, from 00Z of the labelled date) at which each 0.5° box's daily total ends.

    National gauge networks report at different hours: 12 = 12Z on the date, 30 = 06Z the next day; -999 where
    unknown. 360 x 720 float32, south row first, as published (README: "based on very limited and UNCHECKED
    information"). https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/DOCU/
    """
    target = cache_dir(source.name) / "PRCP_CU_GAUGE_V1.0GLB_0.50deg_EOD.lnx"
    _download(source.options["end_of_day"], target)
    return np.fromfile(target, "<f4").reshape(360, 720)


def groups(source, acquired: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """CPC: where each cell's day starts, in hours from 00Z of the labelled date (end of day - 24); else 0."""
    if "end_of_day" not in source.options:
        return np.zeros(len(acquired), dtype=int)
    eod = end_of_day(source)
    rows = np.clip(((np.asarray(lat) + 90) // 0.5).astype(int), 0, 359)
    cols = ((np.asarray(lon) % 360) // 0.5).astype(int) % 720
    hours = eod[rows, cols]
    return np.where(hours > -999, hours - 24, 0).astype(int)


def days(source, centres: np.ndarray, lon: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The CPC day whose 24 hours hold the window centre: day D covers D 00Z + group h to D 00Z + group h + 24 h."""
    return (centres - group * H).astype("datetime64[D]").astype(np.int64)


def period_start(source, day: np.ndarray, group: np.ndarray) -> np.ndarray:
    return np.asarray(day).astype("datetime64[D]") + np.asarray(group) * H


def field(source, variable: str, key) -> np.ndarray:
    if source.period == "day":
        day, _ = key
        date = np.datetime64(int(day), "D")
        year = pd.Timestamp(date).year
        path = source.variables[variable]                             # e.g. cpc_global_precip/precip
        target = cache_dir(source.name) / f"{Path(path).name}.{year}.nc"
        _download(f"{source.store}/{path}.{year}.nc", target)
        return read_netcdf(target, Path(path).name, date.astype("datetime64[ns]"), source.grid)
    return _imerg(source, pd.Timestamp(key))


# GPM IMERG half-hourly ------------------------------------------------------------------------------------------

def _imerg_file(source, start: pd.Timestamp) -> Path:
    from ...build.credentials import earthdata_token

    minute = start.hour * 60 + start.minute
    end = start + pd.Timedelta(minutes=29, seconds=59)
    name = f"3B-HHR.MS.MRG.3IMERG.{start:%Y%m%d}-S{start:%H%M%S}-E{end:%H%M%S}.{minute:04d}.V07B.HDF5"
    url = f"{source.store}/{start:%Y}/{start.dayofyear:03d}/{name}"
    target = cache_dir(source.name) / f"{start:%Y}" / f"{start.dayofyear:03d}" / name
    return _download(url, target, {"Authorization": f"Bearer {earthdata_token()}"})


def _imerg_half_hour(path: Path) -> np.ndarray:
    """precipitation (mm/hr) as lat x lon, row 0 north; the file stores it as lon x lat, south first."""
    import rasterio

    with rasterio.open(f'HDF5:"{path}"://Grid/precipitation') as src:
        data = src.read(1).astype(np.float32)
    if data.shape == (3600, 1800):                                  # lon x lat
        data = data.T
    data = data[::-1]                                               # lat ascending -> row 0 north
    data[data < 0] = np.nan                                         # fill value -9999.9
    return data


def _imerg(source, hour: pd.Timestamp) -> np.ndarray:
    halves = [_imerg_half_hour(_imerg_file(source, hour - pd.Timedelta(minutes=m))) for m in (60, 30)]
    hourly = (halves[0] + halves[1]) / 2                            # mean rate over the hour = mm in that hour
    return place(hourly, Affine(0.1, 0, -180, 0, -0.1, 90), source.grid)
