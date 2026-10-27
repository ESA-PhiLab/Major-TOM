"""The order samples are written in: along a Hilbert curve, so that taco-eo's size cut keeps neighbours together.

    samples (grid_cell, acquired)
      -> each cell's centroid (lon, lat) -> its position on a Hilbert curve over the globe (2^16 x 2^16 steps)
      -> sort by (curve position, cell name, acquisition time)
      -> the writer adds samples in this order; taco-eo's partition_size closes a ZIP before it passes the size
         limit, so every partition is a run of consecutive samples in this order (docs/RELEASE.md section 7)

Why a Hilbert curve: it is one continuous line through the whole longitude x latitude rectangle that turns so any
stretch of it covers a compact patch, without jumps. Consecutive samples are therefore close on the ground, and
so is every run of them: a partition covers a compact area. Sorting by grid row instead gives long thin strips;
a Z-order (Morton) curve jumps between patches more often.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..cells import cell_from_name

ORDER_BITS = 16      # 65,536 steps per axis: ~0.0055° of longitude, ~0.0027° of latitude, finer than any cell


def hilbert(lon: np.ndarray, lat: np.ndarray, order: int = ORDER_BITS) -> np.ndarray:
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


def hilbert_order(samples: pd.DataFrame, d: float = 10.0) -> pd.Index:
    """The index of `samples` (columns grid_cell, acquired) in writing order.

    Cells by their centroid's position on the curve; ties (cells sharing a curve step, never at d >= 1 km) by name;
    a cell's samples by acquisition time, so a cell cut by a partition boundary is cut into two periods.
    """
    cells = pd.Series(samples["grid_cell"].unique())
    centre = np.array([cell_from_name(c, d).centroid for c in cells])
    position = pd.Series(hilbert(centre[:, 0], centre[:, 1]), index=cells.to_numpy())
    keyed = samples.assign(_curve=samples["grid_cell"].map(position))
    return keyed.sort_values(["_curve", "grid_cell", "acquired"], kind="stable").index
