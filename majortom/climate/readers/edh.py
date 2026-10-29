"""Earth Data Hub (Destination Earth): ERA5-Land hourly as zarr v3, read in 60-day chunks.

    store (https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr; key: credentials.edh_key)
      chunks of 1440 h x 50 x 100 pixels (60 days x 5° x 10°), 1 request each; the quota counts requests
    use_cells(cells)  -> the chunks those cells overlap (called once by stream)
    field(var, hour)  -> first hour of a 60-day block: fetch the block's chunks (32 requests in flight over all
                         variables, each retried), each chunk written once into <cache>/blocks/<var>/<block>.t32
                         (chunk-tiled: whole chunks one after another, sparse; <block>.t32.filled.npy marks it
                         complete, so a rerun reuses it) -> one field per hour, read as one 20 kB piece per land
                         chunk. Nothing is mapped and written or read pages are dropped from the page cache, so a
                         job holds its decoded fields, not its 30 GB block files. Blocks written before 2026-10-28
                         (<block>.f32, hour-major) are still read, one 26 MB plane per hour.
    Accumulated variables are stored as totals since 00 UTC (restarting at 01 UTC): field() returns the hour's
    amount, this hour minus the previous. Chunks that come back empty (open sea) are remembered in
    <cache>/empty.json and never requested again: ERA5-Land's land mask does not change.

Values are rounded by the store (bitround, 13 mantissa bits: ~1 part in 16,000).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache_dir

T, Y, X = 1440, 50, 100                          # chunk shape: hours, rows, columns
START = np.datetime64("1950-01-01T00", "h")      # the store's first hour; blocks start at 00 UTC
KEEP = 2                                         # blocks kept per variable: the current one and the one before
IN_FLIGHT = 32                                   # chunk requests at once, over all variables
TRIES = 6                                        # per chunk, with 10 s, 20 s, 40 s ... between tries
_slots = threading.BoundedSemaphore(IN_FLIGHT)
_lock = threading.Lock()
_group = None
_needed: dict[str, set[tuple[int, int]]] = {}    # source name -> chunk (row, column) indices to fetch
_blocks: dict[tuple[str, int], tuple] = {}       # (variable, block) -> (block file, filled pixels, chunks or None)
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
                                   client_kwargs={"auth": aiohttp.BasicAuth("edh", edh_key()),
                                                  "timeout": aiohttp.ClientTimeout(total=300)})
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


def _read(array, sel: tuple) -> np.ndarray:
    """One chunk, retried on timeouts and connection or server errors; at most IN_FLIGHT requests at once.

    A refused key (401/403) or an exhausted quota (429) is not retried: the job stops and a rerun continues.
    """
    for attempt in range(TRIES):
        try:
            with _slots:
                values = array[sel]
            with _lock:
                requests["chunks"] += 1
            return values
        except Exception as error:                   # aiohttp/fsspec raise several types; the message decides
            text = f"{type(error).__name__}: {error}"
            # a status code as a whole number: "received 93429 of 3242240 bytes" (a cut download) is retried
            if re.search(r"\b(401|403|429)\b", text) or attempt == TRIES - 1:
                raise
            with _lock:
                requests["chunks"] += 1                  # a failed request may still count against the quota
            print(f"edh: retry {attempt + 1} after {text[:120]}", flush=True)
            time.sleep(10 * 2 ** attempt)


def _drop(path: Path, offset: int = 0, length: int = 0) -> None:
    """Drop a file's pages (all, or one range) from the page cache, so they do not count against the job."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _hour(path: Path, i: int, height: int, width: int) -> np.ndarray:
    """Hour `i` of a block file: one contiguous plane, read directly (no mapping), then dropped from the cache."""
    n = height * width
    with open(path, "rb") as f:
        plane = np.fromfile(f, np.float32, n, offset=i * n * 4).reshape(height, width)
    _drop(path, i * n * 4, n * 4)
    return plane


def _tile_offset(cy: int, cx: int, width: int) -> int:
    """Byte offset of chunk (cy, cx) in a chunk-tiled block file: whole chunks (T x Y x X), row-major by chunk."""
    return (cy * -(-width // X) + cx) * T * Y * X * 4


def _write_tile(fd: int, cy: int, cx: int, width: int, values: np.ndarray) -> None:
    """One chunk as one contiguous write (padded to T x Y x X), then let its pages go once written back."""
    tile = np.zeros((T, Y, X), np.float32)
    tile[:values.shape[0], :values.shape[1], :values.shape[2]] = values
    offset = _tile_offset(cy, cx, width)
    os.pwrite(fd, memoryview(tile).cast("B"), offset)
    os.posix_fadvise(fd, offset, tile.nbytes, os.POSIX_FADV_DONTNEED)   # starts writeback; drops clean pages


def _tiled_hour(path: Path, i: int, chunks: list[tuple[int, int]], height: int, width: int) -> np.ndarray:
    """Hour `i` of a chunk-tiled block file: one Y x X read per filled chunk, no read-ahead, nothing kept cached."""
    plane = np.zeros((height, width), np.float32)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        for cy, cx in chunks:
            raw = os.pread(fd, Y * X * 4, _tile_offset(cy, cx, width) + i * Y * X * 4)
            tile = np.frombuffer(raw, np.float32).reshape(Y, X)
            h, w = min(Y, height - cy * Y), min(X, width - cx * X)
            plane[cy * Y:cy * Y + h, cx * X:cx * X + w] = tile[:h, :w]
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return plane


def _chunks(filled: np.ndarray) -> list[tuple[int, int]]:
    """The chunks a block holds: a written chunk marks its whole extent filled."""
    return [(cy, cx) for cy in range(-(-filled.shape[0] // Y)) for cx in range(-(-filled.shape[1] // X))
            if filled[cy * Y, cx * X]]


def _fetch(source, variable: str, block: int) -> tuple[Path, np.ndarray, list[tuple[int, int]] | None]:
    """Fetch one block of one variable into a sparse chunk-tiled file; returns (file, filled-pixel mask, chunks).

    chunks is None for a block kept from before 2026-10-28 in the hour-major layout (<block>.f32), read by _hour.
    """
    grid = source.grid
    folder = cache_dir(source.name) / "blocks" / variable
    legacy = folder / f"{block:04d}.f32"
    if legacy.exists() and legacy.with_suffix(".filled.npy").exists():
        return legacy, np.load(legacy.with_suffix(".filled.npy")), None
    path = folder / f"{block:04d}.t32"
    done = folder / f"{block:04d}.t32.filled.npy"            # written last: the block is complete
    if path.exists() and done.exists():                      # a rerun reuses it, no requests
        filled = np.load(done)
        return path, filled, _chunks(filled)
    folder.mkdir(parents=True, exist_ok=True)
    filled = np.zeros((grid.height, grid.width), bool)
    array = _open()[variable]
    t0 = block * T
    empty = []
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o664)
    try:
        os.ftruncate(fd, _tile_offset(-(-grid.height // Y), 0, grid.width))   # sparse: unwritten chunks take no space

        def one(chunk: tuple[int, int]) -> None:
            cy, cx = chunk
            values = _read(array, (slice(t0, t0 + T), slice(cy * Y, (cy + 1) * Y), slice(cx * X, (cx + 1) * X)))
            if np.isnan(values).all():
                empty.append(chunk)
                return
            _write_tile(fd, cy, cx, grid.width, values)
            filled[cy * Y:cy * Y + values.shape[1], cx * X:cx * X + values.shape[2]] = True

        with ThreadPoolExecutor(16) as pool:
            list(pool.map(one, sorted(_needed[source.name])))
        os.fdatasync(fd)                                     # on disk before the block is marked complete
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    if empty:
        with _lock:
            known = _empty(source) | set(empty)
            _empty_file(source).write_text(json.dumps(sorted(known)))
            _needed[source.name] -= set(empty)
            requests["empty"] += len(empty)
    np.save(done, filled)
    return path, filled, _chunks(filled)


def _block(source, variable: str, block: int) -> tuple[Path, np.ndarray, list[tuple[int, int]] | None]:
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
                    folder = cache_dir(source.name) / "blocks" / variable
                    for name in (f"{b:04d}.filled.npy", f"{b:04d}.f32", f"{b:04d}.t32.filled.npy", f"{b:04d}.t32"):
                        (folder / name).unlink(missing_ok=True)
                n = requests["chunks"]
            print(f"edh: {variable} block {block} ({START + block * T}) fetched; {n:,} chunk requests so far",
                  flush=True)
    return _blocks[key]


def _at(source, variable: str, index: int) -> np.ndarray:
    path, filled, chunks = _block(source, variable, index // T)
    grid = source.grid
    f = (_hour(path, index % T, grid.height, grid.width) if chunks is None
         else _tiled_hour(path, index % T, chunks, grid.height, grid.width))
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
