"""Earth Data Hub (Destination Earth): ERA5-Land hourly as zarr v3, read in 60-day chunks.

    store (https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr; key: credentials.edh_key)
      chunks of 1440 h x 50 x 100 pixels (60 days x 5° x 10°), 1 request each; the quota counts requests
    use_cells(cells)  -> the chunks those cells overlap (called once by stream)
    field(var, hour)  -> first hour of a 60-day block: fetch the block's chunks (16 at a time), decode once into
                         <cache>/blocks/<var>/<block>.f32 (hour-major, sparse file) -> one global field per hour
    Accumulated variables are stored as totals since 00 UTC (restarting at 01 UTC): field() returns the hour's
    amount, this hour minus the previous. Chunks that come back empty (open sea) are remembered in
    <cache>/empty.json and never requested again: ERA5-Land's land mask does not change.

Values are rounded by the store (bitround, 13 mantissa bits: ~1 part in 16,000).
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import cache_dir

T, Y, X = 1440, 50, 100                          # chunk shape: hours, rows, columns
START = np.datetime64("1950-01-01T00", "h")      # the store's first hour; blocks start at 00 UTC
KEEP = 2                                         # blocks kept per variable: the current one and the one before
_lock = threading.Lock()
_group = None
_needed: dict[str, set[tuple[int, int]]] = {}    # source name -> chunk (row, column) indices to fetch
_blocks: dict[tuple[str, int], tuple[np.memmap, np.ndarray]] = {}   # (variable, block) -> (data, filled pixels)
_block_locks: dict[tuple[str, int], threading.Lock] = {}
requests = {"chunks": 0, "empty": 0}


def _open():
    """The store's zarr group, opened once; the key goes as HTTP basic auth, never in a URL."""
    global _group
    with _lock:
        if _group is None:
            import aiohttp
            import fsspec
            import zarr

            from ...build.credentials import edh_key
            fs = fsspec.filesystem("https", asynchronous=True,
                                   client_kwargs={"auth": aiohttp.BasicAuth("edh", edh_key())})
            _group = zarr.open_group(zarr.storage.FsspecStore(fs, path=_store_url(), read_only=True), mode="r")
    return _group


def _store_url() -> str:
    from ...build.credentials import EDH_STORE
    return EDH_STORE


def _empty_file(source):
    return cache_dir(source.name) / "empty.json"


def _empty(source) -> set[tuple[int, int]]:
    f = _empty_file(source)
    return {tuple(c) for c in json.loads(f.read_text())} if f.exists() else set()


def use_cells(source, cells: pd.DataFrame) -> None:
    """Fetch only the chunks these cells overlap (cells: grid_row_u, grid_col_r), less those known to be empty."""
    from ..weights import weights

    w = weights(cells, source.grid)
    rows, cols = np.divmod(np.unique(w.pixel), source.grid.width)
    chunks = set(zip((rows // Y).tolist(), (cols // X).tolist()))
    _needed[source.name] = chunks - _empty(source)
    print(f"edh: {len(_needed[source.name])} chunks per variable and 60-day block "
          f"({len(chunks) - len(_needed[source.name])} known empty)", flush=True)


def _fetch(source, variable: str, block: int) -> tuple[np.memmap, np.ndarray]:
    """Fetch one block of one variable into a sparse hour-major file; returns (data, filled-pixel mask)."""
    grid = source.grid
    path = cache_dir(source.name) / "blocks" / variable / f"{block:04d}.f32"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.memmap(path, np.float32, "w+", shape=(T, grid.height, grid.width))
    filled = np.zeros((grid.height, grid.width), bool)
    array = _open()[variable]
    t0 = block * T
    empty = []

    def one(chunk: tuple[int, int]) -> None:
        cy, cx = chunk
        values = array[t0:t0 + T, cy * Y:(cy + 1) * Y, cx * X:(cx + 1) * X]
        with _lock:
            requests["chunks"] += 1
        if np.isnan(values).all():
            empty.append(chunk)
            return
        data[:values.shape[0], cy * Y:cy * Y + values.shape[1], cx * X:cx * X + values.shape[2]] = values
        filled[cy * Y:cy * Y + values.shape[1], cx * X:cx * X + values.shape[2]] = True

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, sorted(_needed[source.name])))
    if empty:
        with _lock:
            known = _empty(source) | set(empty)
            _empty_file(source).write_text(json.dumps(sorted(known)))
            _needed[source.name] -= set(empty)
            requests["empty"] += len(empty)
    data.flush()
    return data, filled


def _block(source, variable: str, block: int) -> tuple[np.memmap, np.ndarray]:
    key = (variable, block)
    with _lock:
        if key in _blocks:
            return _blocks[key]
        lock = _block_locks.setdefault(key, threading.Lock())
    with lock:
        if key not in _blocks:
            got = _fetch(source, variable, block)
            with _lock:
                _blocks[key] = got
                old = sorted(b for v, b in _blocks if v == variable and b <= block - KEEP)
                for b in old:                                      # stream moves forward: drop older blocks
                    del _blocks[(variable, b)]
                    (cache_dir(source.name) / "blocks" / variable / f"{b:04d}.f32").unlink(missing_ok=True)
                n = requests["chunks"]
            print(f"edh: {variable} block {block} ({START + block * T}) fetched; {n:,} chunk requests so far",
                  flush=True)
    return _blocks[key]


def _at(source, variable: str, index: int) -> np.ndarray:
    data, filled = _block(source, variable, index // T)
    f = np.array(data[index % T])
    f[~filled] = np.nan
    return f


def field(source, variable: str, hour) -> np.ndarray:
    """One global field (grid.height x grid.width, NaN at sea and south of 57.1°S) at `hour`.

    Accumulated variables: the amount over the hour ending at `hour` (the total since 00 UTC, less the
    previous hour's, except at 01 UTC where the total restarts).
    """
    hour = np.datetime64(pd.Timestamp(hour).to_datetime64(), "h")
    index = int((hour - START) / np.timedelta64(1, "h"))
    f = _at(source, variable, index)
    if variable in source.accumulated and pd.Timestamp(hour).hour != 1:
        f = f - _at(source, variable, index - 1)
    return f


def valid_range(source) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hour in the store (it is updated to the last closed month)."""
    times = _open()["valid_time"]
    first = pd.Timestamp(START.astype("datetime64[ns]"))
    return first, first + pd.Timedelta(hours=int(times.shape[0]) - 1)
