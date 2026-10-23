"""ERA5 from ARCO-ERA5 (Google): anonymous Zarr, one chunk per variable and hour.

Bandwidth-bound: ~29 fields/s (~64 MB/s) measured on cirrus with 32 threads; processes are no faster.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd


@cache
def _open(store: str):
    import xarray as xr                  # optional dependency (extra "climate"), imported when first used
    return xr.open_zarr(store, chunks=None, storage_options={"token": "anon"})


def valid_range(source) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last time with final data. ARCO-ERA5's time axis runs 1900-2050 with empty slots;
    its metadata names the range that holds data (preliminary ERA5T values after it are left out)."""
    attrs = _open(source.store).attrs
    stop = pd.Timestamp(attrs["valid_time_stop"]) + pd.Timedelta(days=1) - pd.Timedelta(source.step)
    return pd.Timestamp(attrs["valid_time_start"]), stop


def field(source, variable: str, time: pd.Timestamp) -> np.ndarray:
    """One global field (height x width) of `variable` at `time` (UTC, on the source's time step)."""
    start, stop = valid_range(source)
    time = pd.Timestamp(time).tz_localize(None) if pd.Timestamp(time).tzinfo else pd.Timestamp(time)
    if not start <= time <= stop:
        raise ValueError(f"{source.name} has final data from {start} to {stop}, not {time}")
    data = _open(source.store)[source.variables[variable]].sel(time=time).to_numpy()
    return data.astype(np.float32, copy=False)


def land_pixels(source) -> np.ndarray | None:
    """1 where a source pixel counts as land, else 0 (height x width).

    ERA5's land fraction is blended along coasts by the regridding to 0.25° (a pixel 40% land holds soil
    water diluted by sea), so a pixel counts as land when its fraction is at least 0.5, the usual ERA5
    convention. Land-only variables average over these pixels; a cell with none gets no value.
    """
    if source.land_mask is None:
        return None
    return _mask(source.store, source.land_mask, valid_range(source)[0])


@cache
def _mask(store: str, name: str, time: pd.Timestamp) -> np.ndarray:
    return (_open(store)[name].sel(time=time).to_numpy() >= 0.5).astype(np.float32)
