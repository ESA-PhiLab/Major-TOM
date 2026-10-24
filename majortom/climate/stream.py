"""Core-Weather for any number of acquisitions: each hour read once, memory bounded (docs/CLIMATE.md section 6).

    acquisitions (sorted by time); each needs the hours [first, last] of its daily window
      -> for each day of hours, in order:
           cells still needed that day -> weights for those cells (once per day)
           read the day's hours x variables in threads -> buffer[hour] = (cells, values, valid)
           acquisition days whose windows are complete -> hourly_table / daily_table (as in series.py)
             -> <out>/hourly/year=YYYY/<date>.parquet, <out>/daily/year=YYYY/<date>.parquet
           drop buffered hours that no remaining acquisition needs
A restart skips acquisition days whose daily file exists and re-reads only the remaining windows.
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .series import acquisitions, daily_table, hourly_table
from .sources import SOURCES, field, land_pixels, on_step
from .weights import ByCell, cell_means, weights

HOUR = pd.Timedelta(hours=1)


def _window_bounds(acquired: pd.Series, daily: int, hourly: int) -> tuple[pd.Series, pd.Series]:
    """First and last stamped hour each acquisition needs: the union of its daily and hourly windows."""
    first = (acquired - pd.Timedelta(hours=24 * daily + 12)).dt.floor("h") + HOUR
    last = first + pd.Timedelta(hours=24 * (2 * daily + 1) - 1)
    nearest = (acquired + pd.Timedelta(minutes=30)).dt.floor("h")
    return (pd.concat([first, nearest - hourly * HOUR], axis=1).min(axis=1),
            pd.concat([last, nearest + hourly * HOUR], axis=1).max(axis=1))


def _done(out: Path) -> set[pd.Timestamp]:
    return {pd.Timestamp(p.stem) for p in (out / "daily").glob("year=*/*.parquet")}


def _write(table: pd.DataFrame, out: Path, kind: str, day: pd.Timestamp) -> None:
    path = out / kind / f"year={day.year}" / f"{day.date()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def stream(samples: pd.DataFrame, out: Path | str, source: str = "era5", variables: list[str] | None = None,
           hourly: int = 24, daily: int = 30, workers: int = 32, progress: int = 0, max_fields: int | None = None) -> int:
    """Write the hourly and daily tables for `samples` to `out`, one file per acquisition day; returns fields read.

    `max_fields` stops after that many field reads (a quota budget); days finished so far are written, and a rerun
    continues from them.
    """
    out, src = Path(out), SOURCES[source]
    variables = list(variables or src.variables)
    acq = acquisitions(samples)
    acq["day"] = acq["acquired"].dt.normalize()
    acq = acq[~acq["day"].isin(_done(out))].reset_index(drop=True)
    if acq.empty:
        return 0
    cells = acq.drop_duplicates("grid_cell")[["grid_cell", "grid_row_u", "grid_col_r"]].reset_index(drop=True)
    cell_of = pd.Index(cells["grid_cell"]).get_indexer(acq["grid_cell"])
    w = weights(cells, src.grid)
    by_cell = ByCell(w)
    land = land_pixels(src)
    land_share = cell_means(w, land)[0] if land is not None else None
    land_vars = set(src.land_only) & set(variables)
    first, last = _window_bounds(acq["acquired"], daily, hourly)
    first_ns, last_ns = first.to_numpy(), last.to_numpy()
    day_last = last.groupby(acq["day"]).max()

    buffer: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    reads, start, pending = 0, time.monotonic(), list(acq["day"].unique())   # acquisition days, in order
    total = int(((last.max() - first.min()) / HOUR + 1) * len(variables))
    for day in pd.date_range(first.min().normalize(), last.max().normalize(), freq="D"):
        if max_fields is not None and reads >= max_fields:
            break
        hours = pd.date_range(day, periods=24, freq="h")
        lo = np.searchsorted(last_ns, hours[0].to_datetime64(), side="left")     # last >= first hour of the day
        hi = np.searchsorted(first_ns, hours[-1].to_datetime64(), side="right")  # first <= last hour of the day
        if lo < hi:
            day_cells = np.unique(cell_of[lo:hi])
            sub = by_cell.subset(day_cells)
            values = np.empty((24, len(day_cells), len(variables)), np.float32)
            valid = np.empty((24, len(day_cells)), np.float32)

            def one(i: int, j: int) -> None:
                if not on_step(src, hours[i]):                  # e.g. CAMS at 01:00: no field, left empty
                    values[i, :, j] = np.nan
                    if j == 0:
                        valid[i] = np.nan
                    return
                values[i, :, j], ok = cell_means(sub, field(src, variables[j], hours[i]),
                                                 land if variables[j] in land_vars else None)
                if j == 0:
                    valid[i] = ok

            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda ij: one(*ij), itertools.product(range(24), range(len(variables)))))
            for i, h in enumerate(hours):
                buffer[h] = (day_cells, values[i], valid[i])
            day_reads = sum(on_step(src, h) for h in hours) * len(variables)
            reads += day_reads
            if progress and reads // progress != (reads - day_reads) // progress:
                rate = reads / (time.monotonic() - start)
                print(f"{day.date()}: {reads:,}/~{total:,} fields, {rate:.0f}/s, "
                      f"{(total - reads) / rate / 60:.0f} min left", flush=True)
        while pending and day_last[pending[0]] <= hours[-1]:                 # this acquisition day is complete
            _emit(acq, pending.pop(0), cell_of, buffer, src, variables, hourly, daily, land_share, out)
        if pending:
            keep_from = first[acq["day"] == pending[0]].min()
            for h in [h for h in buffer if h < keep_from]:
                del buffer[h]
    return reads


def _emit(acq, day, cell_of, buffer, src, variables, hourly, daily, land_share, out) -> None:
    """Both tables for one acquisition day, from buffered hours."""
    rows = np.flatnonzero((acq["day"] == day).to_numpy())
    batch = acq.iloc[rows].reset_index(drop=True)
    batch_cells, local = np.unique(cell_of[rows], return_inverse=True)
    first, last = _window_bounds(batch["acquired"], daily, hourly)
    hours = pd.date_range(first.min(), last.max(), freq="h")
    values = np.full((len(hours), len(batch_cells), len(variables)), np.nan, np.float32)
    valid = np.full((len(hours), len(batch_cells)), np.nan, np.float32)
    for k, h in enumerate(hours):
        # a day's acquisitions are up to 24 h apart, so near the ends of these hours some cells are outside their
        # own window and were never stored; they stay NaN, and the tables only read each acquisition's own window
        ids, vals, ok = buffer[h]
        pos = np.minimum(np.searchsorted(ids, batch_cells), len(ids) - 1)
        stored = ids[pos] == batch_cells
        values[k, stored], valid[k, stored] = vals[pos[stored]], ok[pos[stored]]
    args = (batch, local, hours, values, valid, src, variables)
    hourly_df, daily_df = hourly_table(*args, hourly), daily_table(*args, daily)
    if land_share is not None:
        share = land_share[cell_of[rows]]
        hourly_df[f"{src.name}_land"] = np.repeat(share, 2 * hourly + 1)
        daily_df[f"{src.name}_land"] = np.repeat(share, 2 * daily + 1)
    _write(hourly_df, out, "hourly", day)
    _write(daily_df, out, "daily", day)                                     # last: its file marks the day done
