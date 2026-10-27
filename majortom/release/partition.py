"""Partitions of about the same size for any modality, keeping neighbouring cells together (docs/RELEASE.md 7).

    samples (grid_cell, acquired, bytes as written)
      -> cell centroid -> position on a Hilbert curve over longitude x latitude (2^16 x 2^16)
      -> sort by (curve position, cell, acquisition time)
      -> walk the cells, closing a partition before it would pass `max_bytes`; a cell's samples stay together,
         unless the cell alone is larger than max_bytes: then its samples are cut in time order
      -> partition id per sample ("p0000", ...) and a summary per partition (size, cells, bounding box, times)
A run of consecutive cells on a Hilbert curve covers a compact area, so a regional query touches few partitions;
cutting on bytes written makes partitions nearly max_bytes in size whatever one sample weighs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..cells import cell_from_name

MAX_BYTES = 3 * 10**9          # partitions end up just under this (each cell is small next to it)


def hilbert(lon: np.ndarray, lat: np.ndarray, order: int = 16) -> np.ndarray:
    """Position along a Hilbert curve of 2^order x 2^order steps over longitude [-180, 180) x latitude [-90, 90]."""
    n = 1 << order
    x = np.clip(((np.asarray(lon, float) + 180) / 360 * n).astype(np.int64), 0, n - 1)
    y = np.clip(((np.asarray(lat, float) + 90) / 180 * n).astype(np.int64), 0, n - 1)
    d = np.zeros(x.shape, np.int64)
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = ~ry & rx                                   # rotate the quadrant so the curve stays continuous
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        swap = ~ry
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s //= 2
    return d


def partition(samples: pd.DataFrame, max_bytes: int = MAX_BYTES, d: float = 10.0) -> tuple[pd.Series, pd.DataFrame]:
    """Partition id per sample (index of `samples`), and a summary per partition.

    `samples` needs grid_cell, acquired and bytes (as written, after compression).
    """
    cells = pd.Series(samples["grid_cell"].unique())
    centre = np.array([cell_from_name(c, d).centroid for c in cells])
    position = pd.Series(hilbert(centre[:, 0], centre[:, 1]), index=cells)
    order = samples.assign(_h=samples["grid_cell"].map(position)).sort_values(["_h", "grid_cell", "acquired"])
    ids = np.empty(len(order), np.int64)
    part, size, i = 0, 0, 0
    sizes = order["bytes"].to_numpy()
    for _, n in _runs(order["grid_cell"].to_numpy()):
        cell_bytes = int(sizes[i:i + n].sum())
        if cell_bytes <= max_bytes:                           # the usual case: the whole cell goes in
            if size and size + cell_bytes > max_bytes:
                part, size = part + 1, 0
            ids[i:i + n] = part
            size += cell_bytes
        else:                                                 # one cell larger than a partition: cut by time
            for k in range(i, i + n):
                if size and size + sizes[k] > max_bytes:
                    part, size = part + 1, 0
                ids[k] = part
                size += int(sizes[k])
        i += n
    width = max(4, len(str(part)))
    names = pd.Series([f"p{p:0{width}d}" for p in ids], index=order.index).reindex(samples.index)
    lon = order["grid_cell"].map(dict(zip(cells, centre[:, 0])))
    lat = order["grid_cell"].map(dict(zip(cells, centre[:, 1])))
    summary = (order.assign(partition=names.loc[order.index], lon=lon, lat=lat)
               .groupby("partition", sort=True)
               .agg(samples=("bytes", "size"), cells=("grid_cell", "nunique"), bytes=("bytes", "sum"),
                    lon_min=("lon", "min"), lon_max=("lon", "max"), lat_min=("lat", "min"), lat_max=("lat", "max"),
                    first_cell=("grid_cell", "first"), last_cell=("grid_cell", "last"),
                    time_min=("acquired", "min"), time_max=("acquired", "max")))
    return names, summary


def _runs(values: np.ndarray):
    """(value, length) of each run of equal consecutive values."""
    if len(values) == 0:
        return
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(values)]
    for a, b in zip(starts, ends):
        yield values[a], int(b - a)
