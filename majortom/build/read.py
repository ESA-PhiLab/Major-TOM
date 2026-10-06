"""Read one scene into a sample window.

Every read puts a file onto the window's pixel grid through a warped view (WarpedVRT):
- product already in the cell's CRS, window on its pixel grid: nearest neighbour, i.e. an exact copy;
- otherwise (latitude/longitude products, another UTM zone): resampled with the collection's method.
The view also says which window pixels the file covers, which is used to stitch tiles and count no-data.
"""
from __future__ import annotations

import numpy as np
import pystac
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from ..cells import Cell
from ..spec import Window, window
from .collections import Collection
from .search import retry

GDAL_OPTIONS = dict(                       # cloud-native reads: one request per header, HTTP/2, retries
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_INGESTED_BYTES_AT_OPEN="32768",
    GDAL_HTTP_VERSION="2", GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_TIMEOUT="30",
)


def layout(cell: Cell, item: pystac.Item, collection: Collection, assets: list[str],
           margin: float, crs: str) -> tuple[Window, dict[str, float], bool]:
    """The window in `crs` and the pixel size per asset for this cell and product.

    If the product is in `crs`, the lattice is its own coarsest pixel grid, so pixels are copied
    unchanged (native). Otherwise the lattice is the coarsest target pixel size and the data are resampled.
    """
    headers = {}
    for asset in assets:
        with rasterio.open(item.assets[asset].href) as src:
            headers[asset] = (src.crs, src.transform)
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


def read_asset(tiles: list[pystac.Item], asset: str, win: Window, res: float,
               resampling: str) -> tuple[np.ndarray, np.ndarray]:
    """One asset on the window grid, stitched from a scene's tiles. Returns (bands, covered pixels)."""
    n = win.pixels(res)
    data, covered = None, np.zeros((n, n), bool)
    for item in tiles:
        with rasterio.open(item.assets[asset].href) as src:
            same_crs = src.crs == CRS.from_user_input(win.crs)
            method = Resampling.nearest if same_crs else Resampling[resampling]
            with WarpedVRT(src, crs=win.crs, transform=win.transform(res), width=n, height=n,
                           resampling=method, add_alpha=True,
                           init_dest_nodata=False, INIT_DEST=0) as vrt:   # empty pixels 0; coverage from alpha
                tile = retry(vrt.read)                       # bands, then alpha (0 = not covered)
        values, alpha = tile[:-1], tile[-1] > 0
        if data is None:
            data = np.zeros_like(values)
        new = alpha & ~covered                               # earlier tiles win where tiles overlap
        data[:, new] = values[:, new]
        covered |= alpha
        if covered.all():
            break
    return data, covered
