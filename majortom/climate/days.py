"""Daily sources around each acquisition: the daily table for ERA5-Land, NOAA CPC, Sentinel-5P (docs/CLIMATE.md 5, 8).

    acquisitions -> per acquisition: a group (ERA5-Land: its UTC-offset group; else 0)
                 -> per daily step: the source day that fits the step's 24 h window (reader.days)
      -> pass 1, by month and group: reader.prepare (fetch a month of requests or a year file, cached)
                 then per source day: field -> cell means for every cell that may need that day
                 -> <cache>/means/<day>_<group>.npz        each source day read once; a rerun skips existing files
      -> pass 2, per acquisition day: each step's source-day means -> daily table
                 -> <out>/daily/year=YYYY/<date>.parquet   a rerun skips existing days

Columns: <name>_<variable><suffix> (ERA5-Land: _mean), <name>_valid, and <name>_start: the UTC start of the
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
from .weights import ByCell, cell_means, weights

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


def _keys(rdr, src, acquired: np.ndarray, lon: np.ndarray, group: np.ndarray, daily: int) -> np.ndarray:
    """acquisitions x steps: the source day (as a key) for each daily step."""
    centres = acquired[:, None] + np.arange(-daily, daily + 1) * DAY
    days = rdr.days(src, centres, lon[:, None], group[:, None])
    return days.astype(np.int64) * 100 + group[:, None] + OFFSET


def _split(key: int) -> tuple[int, int]:
    day, rest = divmod(int(key), 100)
    return day, rest - OFFSET


def stream_days(samples: pd.DataFrame, out: Path | str, source: str, variables: list[str] | None = None,
                daily: int = 30, workers: int = 8, progress: int = 0) -> int:
    """Write the daily table of a daily source for `samples`; returns the number of fields read."""
    out, src = Path(out), SOURCES[source]
    rdr, variables = reader(src), list(variables or src.variables)
    acq = acquisitions(samples)
    acq["day"] = acq["acquired"].dt.normalize()
    acq = acq[~acq["day"].isin(_done(out))].reset_index(drop=True)
    if acq.empty:
        return 0
    cells = acq.drop_duplicates("grid_cell")[["grid_cell", "grid_row_u", "grid_col_r"]].reset_index(drop=True)
    cell_of = pd.Index(cells["grid_cell"]).get_indexer(acq["grid_cell"])
    acquired = acq["acquired"].to_numpy()
    lon = centre_lon(acq["grid_row_u"].to_numpy(), acq["grid_col_r"].to_numpy())
    group = rdr.groups(src, acquired, lon).astype(np.int64)
    by_cell = ByCell(weights(cells, src.grid))
    means = cache_dir(src.name) / "means"
    means.mkdir(exist_ok=True)

    keys = np.unique(np.concatenate([np.unique(_keys(rdr, src, acquired[i:i + 100_000], lon[i:i + 100_000],
                                                     group[i:i + 100_000], daily)) for i in range(0, len(acq), 100_000)]))
    reads, start = 0, time.monotonic()
    by_month = pd.DataFrame({"key": keys, "day": [_split(k)[0] for k in keys], "group": [_split(k)[1] for k in keys]})
    by_month["month"] = pd.to_datetime(by_month["day"], unit="D").dt.to_period("M")
    for (month, g), part in by_month.groupby(["month", "group"], sort=True):
        todo = [k for k in part["key"] if not (means / f"{k}.npz").exists()]
        if not todo:
            continue
        reach = (daily + 2) * DAY                                     # acquisitions that may need this month
        lo, hi = np.searchsorted(acquired, [month.start_time.to_datetime64() - reach,
                                            month.end_time.to_datetime64() + reach])
        near = np.arange(lo, hi)[group[lo:hi] == g]
        if hasattr(rdr, "prepare"):
            rdr.prepare(src, month.start_time, int(g), _bbox(cells, np.unique(cell_of[near])))
        for key in todo:
            day, _ = _split(key)
            date = np.datetime64(day, "D")
            lo, hi = np.searchsorted(acquired, [date - reach, date + reach])
            day_cells = np.unique(cell_of[lo:hi][group[lo:hi] == g])
            sub = by_cell.subset(day_cells)
            with ThreadPoolExecutor(min(workers, len(variables))) as pool:
                fields = list(pool.map(lambda v: field(src, v, (day, int(g))), variables))
            results = [cell_means(sub, f) for f in fields]
            np.savez(means / f"{key}.npz", cells=day_cells, valid=results[0][1].astype(np.float32),
                     values=np.stack([r[0] for r in results], axis=1).astype(np.float32))
            reads += len(variables)
            if progress and reads % progress < len(variables):
                print(f"{date}: {reads:,} fields, {reads / (time.monotonic() - start):.1f}/s", flush=True)

    loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for day in acq["day"].unique():
        rows = np.flatnonzero((acq["day"] == day).to_numpy())
        keys_day = _keys(rdr, src, acquired[rows], lon[rows], group[rows], daily)        # rows x steps
        values = np.full(keys_day.shape + (len(variables),), np.nan, np.float32)
        valid = np.full(keys_day.shape, np.nan, np.float32)
        for key in np.unique(keys_day):
            if key not in loaded:
                with np.load(means / f"{key}.npz") as f:
                    loaded[key] = (f["cells"], f["values"], f["valid"])
            ids, vals, ok = loaded[key]
            r, s = np.nonzero(keys_day == key)
            pos = np.searchsorted(ids, cell_of[rows][r])
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
    return pd.DataFrame({"grid_cell": np.repeat(acq["grid_cell"].to_numpy(), len(steps)),
                         "acquired": np.repeat(acq["acquired"].to_numpy(), len(steps)),
                         "step_d": np.tile(steps, len(acq)), **columns,
                         f"{src.name}_valid": valid.ravel(), f"{src.name}_start": starts})
