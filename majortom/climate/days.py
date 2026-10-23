"""Daily sources around each acquisition: the daily table for ERA5-Land, NOAA CPC, Sentinel-5P (docs/CLIMATE.md 5, 8).

    acquisitions -> per acquisition: a group (ERA5-Land: its UTC-offset group; else 0)
                 -> per daily step: the source day that fits the step's 24 h window (reader.days)
      -> pass 1, by month and group: reader.prepare (fetch a month of requests or a year file, cached)
                 then per source day: field -> cell means for every cell that may need that day
                 -> <cache>/means/<day>_<group>.npz        each source day read once; a rerun skips existing files
      -> pass 2, per acquisition day: each step's source-day means -> daily table
                 -> <out>/daily/year=YYYY/<date>.parquet   a rerun skips existing days

Columns: <name>_<variable><suffix> (ERA5-Land: _mean), <name>_valid (or <name>_<variable>_valid per variable when
the source's gaps differ by variable, as for Sentinel-5P), and <name>_start: the UTC start of the
source day used, so each value says which 24 hours it covers.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .readers import cache_dir
from .series import acquisitions
from .sources import SOURCES, field, reader
from .stream import _done, _write
from .weights import cell_means, weights

OFFSET = 50                               # key = day * 100 + group + OFFSET (groups run from -12 to +14)
DAY = np.timedelta64(1, "D")


def centre_lon(rows: np.ndarray, cols: np.ndarray, d: float = 10.0) -> np.ndarray:
    """Longitude of each cell's centre, from its grid row and column (one Cell per distinct row)."""
    from ..cells import Cell

    out = np.empty(len(rows))
    for row in np.unique(rows):
        cell = Cell(int(row), 0, d)
        at = rows == row
        out[at] = cols[at] * cell.dlon + cell.dlon / 2
    return out


ID = 10_000                                     # cell identity: row * ID + column (columns stay within +-ID/2)


def cell_id(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """A fixed identity per cell, the same in every run, so cached means can be shared between runs."""
    return np.asarray(rows, np.int64) * ID + np.asarray(cols, np.int64)


def _cells_of(ids: np.ndarray) -> pd.DataFrame:
    rows = np.round(ids / ID).astype(np.int64)
    return pd.DataFrame({"grid_row_u": rows, "grid_col_r": ids - rows * ID})


def centre_lat(rows: np.ndarray, d: float = 10.0) -> np.ndarray:
    """Latitude of each cell's centre, from its grid row."""
    from ..cells import Cell

    dlat = Cell(0, 0, d).dlat
    return rows * dlat + dlat / 2


def _keys(rdr, src, acquired: np.ndarray, lon: np.ndarray, group: np.ndarray, daily: int) -> np.ndarray:
    """acquisitions x steps: the source day (as a key) for each daily step."""
    centres = acquired[:, None] + np.arange(-daily, daily + 1) * DAY
    days = rdr.days(src, centres, lon[:, None], group[:, None])
    return days.astype(np.int64) * 100 + group[:, None] + OFFSET


def _split(key: int) -> tuple[int, int]:
    day, rest = divmod(int(key), 100)
    return day, rest - OFFSET


def stream_days(samples: pd.DataFrame, out: Path | str, source: str, variables: list[str] | None = None,
                daily: int = 30, workers: int = 8, progress: int = 0, max_fields: int | None = None) -> int:
    """Write the daily table of a daily source for `samples`; returns the number of fields read.

    `max_fields` stops reading after that many fields (a quota budget, e.g. Earth Engine's monthly EECU): only
    acquisition days whose source days are all cached are written, and the next run continues from the cache.
    """
    out, src = Path(out), SOURCES[source]
    rdr, variables = reader(src), list(variables or src.variables)
    acq = acquisitions(samples)
    acq["day"] = acq["acquired"].dt.normalize()
    acq = acq[~acq["day"].isin(_done(out))].reset_index(drop=True)
    if acq.empty:
        return 0
    acq_ids = cell_id(acq["grid_row_u"].to_numpy(), acq["grid_col_r"].to_numpy())
    ids = np.unique(acq_ids)                                       # this run's cells, sorted by identity
    cell_of = np.searchsorted(ids, acq_ids)
    cells = _cells_of(ids)
    acquired = acq["acquired"].to_numpy()
    lon = centre_lon(acq["grid_row_u"].to_numpy(), acq["grid_col_r"].to_numpy())
    lat = centre_lat(acq["grid_row_u"].to_numpy())
    group = rdr.groups(src, acquired, lon, lat).astype(np.int64)
    means = cache_dir(src.name) / "means"
    means.mkdir(exist_ok=True)

    keys = np.unique(np.concatenate([np.unique(_keys(rdr, src, acquired[i:i + 100_000], lon[i:i + 100_000],
                                                     group[i:i + 100_000], daily)) for i in range(0, len(acq), 100_000)]))
    reads, start = 0, time.monotonic()
    by_month = pd.DataFrame({"key": keys, "day": [_split(k)[0] for k in keys], "group": [_split(k)[1] for k in keys]})
    by_month["month"] = pd.to_datetime(by_month["day"], unit="D").dt.to_period("M")
    first, last = (np.datetime64(v, "D") for v in rdr.valid_range(src)) if hasattr(rdr, "valid_range") else (None, None)
    for (month, g), part in by_month.groupby(["month", "group"], sort=True):
        if max_fields is not None and reads >= max_fields:
            break
        reach = (daily + 2) * DAY                                     # acquisitions that may need this month
        prepared = False
        for key in part["key"]:
            if max_fields is not None and reads >= max_fields:
                break
            day, _ = _split(key)
            date = np.datetime64(day, "D")
            lo, hi = np.searchsorted(acquired, [date - reach, date + reach])
            need = ids[np.unique(cell_of[lo:hi][group[lo:hi] == g])]
            path = means / f"{key}.npz"
            have = np.load(path)["cells"] if path.exists() else np.array([], np.int64)
            if np.isin(need, have).all():
                continue                                              # cached for every cell this run needs
            need = np.union1d(need, have)                             # keep what other runs cached
            outside = first is not None and not (first <= date <= last)
            if not outside and not prepared and hasattr(rdr, "prepare"):
                near = np.arange(*np.searchsorted(acquired, [month.start_time.to_datetime64() - reach,
                                                             month.end_time.to_datetime64() + reach]))
                rdr.prepare(src, month.start_time, int(g), _bbox(cells, np.unique(cell_of[near[group[near] == g]])))
                prepared = True
            if outside:                                               # before or after the record: empty, no read
                empty = np.full((len(need), len(variables)), np.nan, np.float32)
                np.savez(path, cells=need, values=empty, valid=np.zeros_like(empty))
                continue
            sub = weights(_cells_of(need), src.grid)
            with ThreadPoolExecutor(min(workers, len(variables))) as pool:
                fields = list(pool.map(lambda v: field(src, v, (day, int(g))), variables))
            results = [cell_means(sub, f) for f in fields]
            np.savez(path, cells=need, valid=np.stack([r[1] for r in results], axis=1).astype(np.float32),
                     values=np.stack([r[0] for r in results], axis=1).astype(np.float32))
            reads += len(variables)
            if progress and reads % progress < len(variables):
                print(f"{date}: {reads:,} fields, {reads / (time.monotonic() - start):.1f}/s", flush=True)

    loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for day in acq["day"].unique():
        rows = np.flatnonzero((acq["day"] == day).to_numpy())
        keys_day = _keys(rdr, src, acquired[rows], lon[rows], group[rows], daily)        # rows x steps
        if not all((means / f"{k}.npz").exists() for k in np.unique(keys_day)):        # budget reached before it
            continue
        values = np.full(keys_day.shape + (len(variables),), np.nan, np.float32)
        valid = np.full(keys_day.shape + (len(variables),), np.nan, np.float32)
        for key in np.unique(keys_day):
            if key not in loaded:
                with np.load(means / f"{key}.npz") as f:
                    ok = f["valid"]
                    loaded[key] = (f["cells"], f["values"], ok[:, None] if ok.ndim == 1 else ok)
            have, vals, ok = loaded[key]
            r, s = np.nonzero(keys_day == key)
            want = acq_ids[rows][r]
            pos = np.minimum(np.searchsorted(have, want), len(have) - 1)
            if not (have[pos] == want).all():
                raise RuntimeError(f"cached source day {key} lacks cells this run needs: {means / f'{key}.npz'}")
            values[r, s], valid[r, s] = vals[pos], ok[pos]
        oldest = np.datetime64(day.date()) - (daily + 3) * DAY                           # free what is behind
        for key in [k for k in loaded if np.datetime64(_split(k)[0], "D") < oldest]:
            del loaded[key]
        _write(_table(acq.iloc[rows], keys_day, values, valid, src, variables, daily, rdr), out, "daily", day)
    return reads


def _bbox(cells: pd.DataFrame, which: np.ndarray) -> tuple[float, float, float, float]:
    """north, west, south, east around the given cells, in degrees, with half a degree of margin."""
    from ..cells import Cell

    part = cells.iloc[which]
    rows = part["grid_row_u"].to_numpy()
    south = np.array([Cell(int(r), 0).lat for r in rows])
    north = south + Cell(0, 0).dlat
    lons = centre_lon(rows, part["grid_col_r"].to_numpy())
    return (min(90.0, north.max() + 0.5), max(-180.0, lons.min() - 1.0),
            max(-90.0, south.min() - 0.5), min(180.0, lons.max() + 1.0))


def _table(acq: pd.DataFrame, keys: np.ndarray, values, valid, src, variables, daily: int, rdr) -> pd.DataFrame:
    steps = np.arange(-daily, daily + 1)
    suffix = src.options.get("suffix", "")
    days, groups = np.divmod(keys.ravel(), 100)
    starts = rdr.period_start(src, days, groups - OFFSET).astype("datetime64[s]")
    columns = {f"{src.name}_{v}{suffix}": values[:, :, j].ravel() for j, v in enumerate(variables)}
    if src.options.get("valid_per_variable"):
        columns.update({f"{src.name}_{v}_valid": valid[:, :, j].ravel() for j, v in enumerate(variables)})
    else:
        columns[f"{src.name}_valid"] = valid[:, :, 0].ravel()
    return pd.DataFrame({"grid_cell": np.repeat(acq["grid_cell"].to_numpy(), len(steps)),
                         "acquired": np.repeat(acq["acquired"].to_numpy(), len(steps)),
                         "step_d": np.tile(steps, len(acq)), **columns, f"{src.name}_start": starts})
