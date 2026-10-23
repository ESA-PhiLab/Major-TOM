"""Weather around each acquisition: the hourly and daily tables of Core-Weather (docs/CLIMATE.md section 5).

    samples (grid_cell, grid_row_u, grid_col_r, timestamp)
      -> acquisitions(): one row per (grid_cell, acquired), UTC
      -> hours_needed(): union of every acquisition's windows, as whole hours
      -> cell_hours(): per needed hour and variable, field -> cell means      -> [hour, cell, variable]
      -> hourly_table(): steps -24..+24 h from the nearest full hour
      -> daily_table(): steps -30..+30 days; step d is the 24 stamped hours H with
                        acquired + 24d - 12h < H <= acquired + 24d + 12h
      -> (hourly, daily) DataFrames

An accumulated value stamped H is the total over the hour ending at H, so a daily window's sums cover
its 24 hours shifted by up to one hour. This version holds every needed hour for every distinct cell:
fine for thousands of acquisitions, not for all of Core at once (that needs time blocks).
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .sources import SOURCES, Source, field, land_pixels
from .weights import cell_means, weights

HOUR = np.timedelta64(1, "h")
EXTREMES = ("t2m",)                  # daily min and max as well as the mean


def acquisitions(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per (grid_cell, acquired): samples of one acquisition in several datasets share a row."""
    acquired = pd.to_datetime(samples["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    acq = samples.assign(acquired=acquired).drop_duplicates(["grid_cell", "acquired"])
    return acq[["grid_cell", "grid_row_u", "grid_col_r", "acquired"]].sort_values("acquired").reset_index(drop=True)


def _hourly_times(acquired: np.ndarray, hourly: int) -> np.ndarray:
    """acquisitions x steps: the nearest full hour plus each step."""
    nearest = (acquired + 30 * np.timedelta64(1, "m")).astype("datetime64[h]")
    return nearest[:, None] + np.arange(-hourly, hourly + 1) * HOUR


def _daily_times(acquired: np.ndarray, daily: int) -> np.ndarray:
    """acquisitions x steps x 24: the stamped hours in each 24 h window centred on acquired + 24 h x step."""
    low = acquired[:, None] + (np.arange(-daily, daily + 1) * 24 - 12) * HOUR
    first = low.astype("datetime64[h]") + HOUR                      # first full hour after the window opens
    return first[:, :, None] + np.arange(24) * HOUR


def hours_needed(acquired: np.ndarray, hourly: int, daily: int) -> pd.DatetimeIndex:
    times = np.concatenate([_hourly_times(acquired, hourly).ravel(), _daily_times(acquired, daily).ravel()])
    return pd.DatetimeIndex(np.unique(times).astype("datetime64[ns]"))


def cell_hours(cells: pd.DataFrame, hours: pd.DatetimeIndex, source: Source, variables: list[str],
               workers: int = 16, progress: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """values [hour, cell, variable] and valid [hour, cell]; reads run in threads (they wait on the network).
    With `progress`, prints the count, rate and time left every `progress` reads."""
    w = weights(cells, source.grid)
    land = land_pixels(source) if set(variables) & set(source.land_only) else None
    values = np.empty((len(hours), len(cells), len(variables)), np.float32)
    valid = np.empty((len(hours), len(cells)), np.float32)
    total, done, start = len(hours) * len(variables), itertools.count(1), time.monotonic()

    def one(i: int, j: int) -> None:
        mask = land if variables[j] in source.land_only else None
        values[i, :, j], ok = cell_means(w, field(source, variables[j], hours[i]), mask)
        if j == 0:
            valid[i] = ok
        n = next(done)
        if progress and n % progress == 0:
            rate = n / (time.monotonic() - start)
            print(f"{n:,}/{total:,} fields, {rate:.0f}/s, {(total - n) / rate / 60:.0f} min left", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda ij: one(*ij), itertools.product(range(len(hours)), range(len(variables)))))
    return values, valid


def _index(hours: pd.DatetimeIndex, times: np.ndarray) -> np.ndarray:
    index = hours.get_indexer(times.ravel().astype("datetime64[ns]")).reshape(times.shape)
    assert (index >= 0).all(), "a window hour was not read"
    return index


def hourly_table(acq: pd.DataFrame, cell_of: np.ndarray, hours, values, valid, source: Source,
                 variables: list[str], hourly: int) -> pd.DataFrame:
    steps = np.arange(-hourly, hourly + 1)
    h = _index(hours, _hourly_times(acq["acquired"].to_numpy(), hourly))
    c = cell_of[:, None]
    columns = {f"{source.name}_{v}": values[h, c, j].ravel() for j, v in enumerate(variables)}
    columns[f"{source.name}_valid"] = valid[h, c].ravel()
    return pd.DataFrame({"grid_cell": np.repeat(acq["grid_cell"].to_numpy(), len(steps)),
                         "acquired": np.repeat(acq["acquired"].to_numpy(), len(steps)),
                         "step_h": np.tile(steps, len(acq)), **columns})


def daily_table(acq: pd.DataFrame, cell_of: np.ndarray, hours, values, valid, source: Source,
                variables: list[str], daily: int) -> pd.DataFrame:
    """Per 24 h window: sums of accumulations, means of states, and min/max for EXTREMES."""
    steps = np.arange(-daily, daily + 1)
    h = _index(hours, _daily_times(acq["acquired"].to_numpy(), daily))      # acquisitions x steps x 24
    c = cell_of[:, None, None]
    columns = {}
    for j, v in enumerate(variables):
        window = values[h, c, j]
        if v in source.accumulated:
            columns[f"{source.name}_{v}_sum"] = window.sum(axis=2).ravel()
            continue
        columns[f"{source.name}_{v}_mean"] = window.mean(axis=2).ravel()
        if v in EXTREMES:
            columns[f"{source.name}_{v}_min"] = window.min(axis=2).ravel()
            columns[f"{source.name}_{v}_max"] = window.max(axis=2).ravel()
    columns[f"{source.name}_valid"] = valid[h, c].mean(axis=2).ravel()
    return pd.DataFrame({"grid_cell": np.repeat(acq["grid_cell"].to_numpy(), len(steps)),
                         "acquired": np.repeat(acq["acquired"].to_numpy(), len(steps)),
                         "step_d": np.tile(steps, len(acq)), **columns})


def series(samples: pd.DataFrame, source: str = "era5", variables: list[str] | None = None,
           hourly: int = 24, daily: int = 30, workers: int = 16, progress: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The hourly and daily tables for `samples` (columns grid_cell, grid_row_u, grid_col_r, timestamp)."""
    src = SOURCES[source]
    variables = list(variables or src.variables)
    acq = acquisitions(samples)
    cells = acq.drop_duplicates("grid_cell")[["grid_cell", "grid_row_u", "grid_col_r"]].reset_index(drop=True)
    cell_of = pd.Index(cells["grid_cell"]).get_indexer(acq["grid_cell"])
    hours = hours_needed(acq["acquired"].to_numpy(), hourly, daily)
    values, valid = cell_hours(cells, hours, src, variables, workers, progress)
    args = (acq, cell_of, hours, values, valid, src, variables)
    hourly_df, daily_df = hourly_table(*args, hourly), daily_table(*args, daily)
    land = land_pixels(src)
    if land is not None:                         # share of each cell in land pixels: what land-only variables cover
        share = cell_means(weights(cells, src.grid), land)[0][cell_of]
        hourly_df[f"{src.name}_land"] = np.repeat(share, 2 * hourly + 1)
        daily_df[f"{src.name}_land"] = np.repeat(share, 2 * daily + 1)
    return hourly_df, daily_df
