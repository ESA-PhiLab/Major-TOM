"""Read one scene into a sample window, from any archive.

    layout      the window and pixel size per band, from each asset's grid (backend.header)
    read_asset  one asset stitched from a scene's tiles (backend.read_tile per tile): earlier tiles win where
                tiles overlap, and the covered pixels give the no-data share

Each backend puts a tile onto the window's pixel grid unchanged where the product is already on it (native),
else resampled with the collection's method.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pystac
from rasterio.crs import CRS

from ..cells import Cell
from ..spec import Window, window
from .backends import backend
from .collections import Collection, Source

if TYPE_CHECKING:
    from .archives import Archive


def layout(archive: Archive, cell: Cell, item: pystac.Item, collection: Collection, source: Source,
           assets: list[str], margin: float, crs: str) -> tuple[Window, dict[str, float], bool]:
    """The window in `crs` and the pixel size per asset (band name) for this cell and product.

    If the product is in `crs`, the lattice is its own coarsest pixel grid, so pixels are copied
    unchanged (native). Otherwise the lattice is the coarsest target pixel size and the data are resampled.
    """
    headers = {asset: backend(archive).header(archive, item, source.key(asset)) for asset in assets}
    native = all(c == CRS.from_user_input(crs) for c, _ in headers.values())
    if native:
        res = {a: abs(t.a) for a, (_, t) in headers.items()}
        coarse = max(res, key=res.get)
        origin = headers[coarse][1].c, headers[coarse][1].f
    else:
        res = {a: collection.target_res(a) or _metres(c, t) for a, (c, t) in headers.items()}
        coarse, origin = max(res, key=res.get), None
    return window(cell, res[coarse], origin, margin, crs), res, native


def _metres(crs: CRS, transform) -> float:
    if crs.is_geographic:
        raise ValueError("product is in latitude/longitude: set res_m for this collection")
    return abs(transform.a)


def read_asset(archive: Archive, tiles: list[pystac.Item], asset: str, win: Window, res: float,
               resampling: str) -> tuple[np.ndarray, np.ndarray]:
    """One asset (the archive's key) on the window grid, stitched from a scene's tiles.

    Returns (bands, covered pixels).
    """
    n = win.pixels(res)
    data, covered = None, np.zeros((n, n), bool)
    for item in tiles:
        values, alpha = backend(archive).read_tile(archive, item, asset, win, res, resampling)
        if data is None:
            data = np.zeros_like(values)
        new = alpha & ~covered                               # earlier tiles win where tiles overlap
        data[:, new] = values[:, new]
        covered |= alpha
        if covered.all():
            break
    return data, covered
