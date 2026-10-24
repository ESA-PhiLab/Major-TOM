"""How each climate source is read: one module per access method (docs/CLIMATE.md section 6).

    zarr.py         ERA5 from ARCO-ERA5 (anonymous Zarr), one field per hour
    queue.py        Climate / Atmosphere Data Store requests (ERA5-Land daily statistics, CAMS), cached per month
    files.py        downloadable files (NOAA CPC yearly netCDF, IMERG half-hourly HDF5), cached per file
    earthengine.py  Sentinel-5P daily mosaics on a 0.05° grid (computePixels)

Hourly sources ("hour" period) provide field(source, variable, time). Daily sources ("day" period) provide
field(source, variable, key) for a source day, and groups / days / period_start, which say which source day
fits each acquisition's daily window (see majortom/climate/days.py). Every field is global, on source.grid,
row 0 north, NaN where missing.
"""
from __future__ import annotations

import os
import threading
from functools import cache
from pathlib import Path

import numpy as np
from affine import Affine

from ..weights import Grid


def cache_dir(source_name: str) -> Path:
    """Where downloaded files are kept: $MAJORTOM_CLIMATE_CACHE (default ~/.cache/majortom/climate) / source."""
    root = Path(os.environ.get("MAJORTOM_CLIMATE_CACHE", "~/.cache/majortom/climate")).expanduser()
    path = root / source_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def place(data: np.ndarray, transform: Affine, grid: Grid) -> np.ndarray:
    """A (possibly regional, possibly south-up) lat/lon array placed on the global `grid`, NaN elsewhere.

    Longitudes wrap, so 0-360° and -180-180° files land in the same columns.
    """
    data = np.asarray(data, dtype=np.float32)
    north = transform.f
    if transform.e > 0:                                      # south-up: flip so row 0 is north
        data, north = data[::-1], transform.f + transform.e * data.shape[0]
    out = np.full((grid.height, grid.width), np.nan, np.float32)
    row0 = int(round((grid.north - north) / grid.step))
    cols = (int(round((transform.c - grid.west) / grid.step)) + np.arange(data.shape[1])) % round(360 / grid.step)
    rows = row0 + np.arange(data.shape[0])
    keep_r = (rows >= 0) & (rows < grid.height)
    keep_c = cols < grid.width
    out[np.ix_(rows[keep_r], cols[keep_c])] = data[np.ix_(keep_r, keep_c)]
    return out


def netcdf_times(tags: dict, dimension: str) -> np.ndarray:
    """The values of a netCDF time dimension as datetime64, from GDAL's tags ("<unit> since <origin>")."""
    import pandas as pd

    units = tags[f"{dimension}#units"]
    unit, origin = units.split(" since ")
    values = [float(v) for v in tags[f"NETCDF_DIM_{dimension}_VALUES"].strip("{}").split(",")]
    step = {"days": "D", "hours": "h", "minutes": "min", "seconds": "s"}[unit.strip()]
    return (pd.Timestamp(origin.strip()) + pd.to_timedelta(values, unit=step)).to_numpy()


_netcdf = threading.Lock()                          # GDAL's netCDF driver is not thread-safe: one access at a time


@cache
def netcdf_variables(path: str) -> frozenset[str]:
    """The variables a netCDF file holds (from its subdataset names)."""
    import rasterio

    with _netcdf, rasterio.open(path) as src:
        if src.subdatasets:
            return frozenset(name.rsplit(":", 1)[-1] for name in src.subdatasets)
        return frozenset([src.tags(1).get("NETCDF_VARNAME", "")]) if src.count else frozenset()   # one variable


@cache
def band_times(path: str, variable: str) -> tuple[np.ndarray, tuple]:
    """Times of each band of `variable` in a netCDF file, and the file's transform."""
    import rasterio

    with _netcdf, rasterio.open(f'netcdf:"{path}":{variable}') as src:
        tags = src.tags()
        dim = next(k.removeprefix("NETCDF_DIM_").removesuffix("_VALUES") for k in tags
                   if k.startswith("NETCDF_DIM_") and k.endswith("_VALUES") and "time" in k)
        return netcdf_times(tags, dim), tuple(src.transform)[:6]


def read_netcdf(path: Path, variable: str, when: np.datetime64, grid: Grid, until: np.datetime64 | None = None) -> np.ndarray:
    """The band of `variable` at `when` (or, with `until`, the band stamped in [when, until)) from a cached file
    or a folder of files, on the global grid. Monthly means are stamped differently per product (CAMS: 21:00 on
    the first day), so monthly readers match the month."""
    import rasterio

    files = [path] if path.is_file() else sorted(path.glob("*.nc"))
    holding = [f for f in files if variable in netcdf_variables(str(f))]
    if not holding:
        raise KeyError(f"no file in {path} holds {variable}")
    f = holding[0]
    times, transform = band_times(str(f), variable)
    band = np.flatnonzero(times == when) if until is None else np.flatnonzero((times >= when) & (times < until))
    if not len(band):
        raise KeyError(f"{variable} has no band for {when} in {f}")
    with _netcdf, rasterio.open(f'netcdf:"{f}":{variable}') as src:
        raw = src.read(int(band[0]) + 1)
        data = raw.astype(np.float32) * (src.scales[band[0]] or 1) + (src.offsets[band[0]] or 0)
        if src.nodata is not None:
            data[raw == src.nodata] = np.nan
    return place(data, Affine(*transform), grid)
