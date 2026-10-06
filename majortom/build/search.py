"""Find scenes over a window in the Planetary Computer STAC catalogue.

A scene is all tiles of one acquisition (same time); tiles of one scene can be stitched together.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Callable, TypeVar

import pandas as pd
import planetary_computer
import pystac
import pystac_client
import requests
import shapely
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from ..cells import Cell
from ..spec import cell_crs, window
from .collections import Collection

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
T = TypeVar("T")


def retry(fn: Callable[[], T], attempts: int = 4) -> T:
    """Call `fn`, retrying with waits of 1, 2, 4 s (rate limits, dropped connections)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


@lru_cache(maxsize=1)
def catalog() -> pystac_client.Client:
    """The catalogue client; asset links come back signed, so no account is needed."""
    client = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace, timeout=30)
    client.add_conforms_to("SORT")                      # Planetary Computer sorts but does not declare it
    client._stac_io.session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))
    return client


def acquired(item: pystac.Item) -> pd.Timestamp:
    """Acquisition time in UTC."""
    when = pd.Timestamp(item.datetime or item.properties["start_datetime"])
    return when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")


def item_crs(item: pystac.Item) -> str | None:
    """The tile's CRS from its STAC metadata, without opening a file."""
    code, epsg = item.properties.get("proj:code"), item.properties.get("proj:epsg")
    return code or (f"EPSG:{epsg}" if epsg else None)


def scenes(name: str, collection: Collection, bbox: tuple[float, float, float, float],
           datetime: str | None, limit: int = 50) -> list[list[pystac.Item]]:
    """Scenes over a lon/lat box, best first: least cloudy, or newest for products without clouds.

    A scene is the list of tiles of one acquisition.
    """
    query = dict(collections=[name], bbox=bbox, max_items=limit,
                 sortby=[{"field": "eo:cloud_cover", "direction": "asc"}] if collection.clouds
                 else [{"field": "datetime", "direction": "desc"}])
    if collection.time and datetime:
        query["datetime"] = datetime
    needed = set(collection.assets) | ({collection.clouds[0]} if collection.clouds else set())
    items = [it for it in retry(lambda: list(catalog().search(**query).items())) if needed <= set(it.assets)]
    groups: dict[pd.Timestamp, list[pystac.Item]] = {}
    for item in items:                                  # dicts keep the catalogue's order
        groups.setdefault(acquired(item), []).append(item)
    return list(groups.values())


def tiles_for(scene: list[pystac.Item], cell: Cell, margin: float) -> tuple[str, list[pystac.Item]]:
    """The window's CRS and the tiles to read for one scene (docs/SPEC.md sections 3.1 and 6.4).

    The cell's UTM zone if that zone's tiles cover the whole window, else another projected zone whose
    tiles do: pixels stay native. Products in latitude/longitude use the cell's zone and are resampled.
    Tiles of one CRS only, those containing the whole window first.
    """
    target = cell_crs(cell)
    by_crs: dict[str | None, list[pystac.Item]] = {}
    for item in scene:
        by_crs.setdefault(item_crs(item), []).append(item)
    projected = sorted((c for c in by_crs if c and CRS.from_user_input(c).is_projected), key=lambda c: c != target)
    for crs in projected:
        need = shapely.box(*transform_bounds(crs, "EPSG:4326", *window(cell, 10, margin=margin, crs=crs).bounds,
                                             densify_pts=21))
        footprints = {it.id: shape(it.geometry) for it in by_crs[crs]}
        if shapely.union_all(list(footprints.values())).contains(need):
            return crs, sorted(by_crs[crs], key=lambda it: not footprints[it.id].contains(need))
    if projected:                                       # no zone covers it all: best effort, flagged by nodata
        return projected[0], by_crs[projected[0]]
    return target, scene
