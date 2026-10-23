"""Climate sources: where each lives, its grid, variables and units (docs/CLIMATE.md section 2).

    SOURCES[name]  (store, Grid, our variable name -> the source's, units, time step)
      -> field(source, variable, time)   one global array, row 0 north, NaN where missing
      -> weights.cell_means              value and valid share per cell

Values keep the units the producer published; they are recorded per variable, never converted.
Values are not altered either: ERA5 accumulations hold tiny negative values (about -2e-9 m) from its
storage precision, and they stay as published.
Only ERA5 exists so far.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from .weights import Grid


@dataclass(frozen=True)
class Source:
    name: str                    # column prefix: era5_t2m
    store: str                   # where the data are
    grid: Grid
    variables: dict[str, str]    # our short name -> the source's name
    units: dict[str, str]        # as stored
    step: str                    # time step, as a pandas frequency: "1h", "3h", "1D"
    accumulated: tuple[str, ...] = ()   # totals over the step ending at the time stamp, not instantaneous values


ERA5_VARIABLES = {
    "t2m": ("2m_temperature", "K"),
    "d2m": ("2m_dewpoint_temperature", "K"),
    "tp": ("total_precipitation", "m"),
    "u10": ("10m_u_component_of_wind", "m s**-1"),
    "v10": ("10m_v_component_of_wind", "m s**-1"),
    "sp": ("surface_pressure", "Pa"),
    "ssrd": ("surface_solar_radiation_downwards", "J m**-2"),
    "sf": ("snowfall", "m of water equivalent"),
    "sd": ("snow_depth", "m of water equivalent"),
    "swvl1": ("volumetric_soil_water_layer_1", "m**3 m**-3"),
}

SOURCES = {
    # ARCO-ERA5 (Google): Zarr format 2, one chunk per variable and hour; latitude 90 -> -90, longitude 0 -> 359.75
    "era5": Source("era5", "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                   Grid(west=-0.125, north=90.125, step=0.25, width=1440, height=721),
                   {k: v[0] for k, v in ERA5_VARIABLES.items()}, {k: v[1] for k, v in ERA5_VARIABLES.items()},
                   step="1h", accumulated=("tp", "ssrd", "sf")),
}


@cache
def _open(store: str):
    import xarray as xr                  # optional dependency (extra "climate"), imported when first used
    return xr.open_zarr(store, chunks=None, storage_options={"token": "anon"})


def valid_range(source: Source) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last time with final data. ARCO-ERA5's time axis runs 1900-2050 with empty slots;
    its metadata names the range that holds data (preliminary ERA5T values after it are left out)."""
    attrs = _open(source.store).attrs
    stop = pd.Timestamp(attrs["valid_time_stop"]) + pd.Timedelta(days=1) - pd.Timedelta(source.step)
    return pd.Timestamp(attrs["valid_time_start"]), stop


def field(source: Source, variable: str, time: pd.Timestamp) -> np.ndarray:
    """One global field (height x width) of `variable` at `time` (UTC, on the source's time step)."""
    start, stop = valid_range(source)
    time = pd.Timestamp(time).tz_localize(None) if pd.Timestamp(time).tzinfo else pd.Timestamp(time)
    if not start <= time <= stop:
        raise ValueError(f"{source.name} has final data from {start} to {stop}, not {time}")
    data = _open(source.store)[source.variables[variable]].sel(time=time).to_numpy()
    return data.astype(np.float32, copy=False)
