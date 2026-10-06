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


ORDER = {                                               # catalogue sort for each preference
    "clearest": [{"field": "eo:cloud_cover", "direction": "asc"}],
    "cloudiest": [{"field": "eo:cloud_cover", "direction": "desc"}],
    "newest": [{"field": "datetime", "direction": "desc"}],
    "nearest": [{"field": "datetime", "direction": "desc"}],   # then re-sorted by distance to the date
}


def middle(datetime: str | None) -> pd.Timestamp:
    """The date a "nearest" search aims at: the given time, the middle of a range, or now."""
    if not datetime:
        return pd.Timestamp.now(tz="UTC")
    ends = [pd.Timestamp(t) for t in datetime.split("/")]
    ends = [t.tz_localize("UTC") if t.tzinfo is None else t for t in ends]
    return ends[0] + (ends[-1] - ends[0]) / 2


def scenes(name: str, collection: Collection, bbox: tuple[float, float, float, float],
           datetime: str | None, prefer: str = "clearest", limit: int = 100) -> tuple[list[list[pystac.Item]], str]:
    """Scenes over a lon/lat box in the order of `prefer`, and a note on which dates were used.

    A scene is the list of tiles of one acquisition. Products without clouds are ordered newest first
    for "clearest" and "cloudiest". Yearly products take the newest version if none is in `datetime`.
    """
    if not collection.clouds and prefer in ("clearest", "cloudiest"):
        prefer = "newest"
    needed = set(collection.assets) | ({collection.clouds[0]} if collection.clouds else set())

    def search(dates: str | None) -> list[pystac.Item]:
        query = dict(collections=[name], bbox=bbox, max_items=limit, sortby=ORDER[prefer])
        if dates:
            query["datetime"] = dates
        return [it for it in retry(lambda: list(catalog().search(**query).items())) if needed <= set(it.assets)]

    note = "dates ignored: one version" if collection.time == "static" else f"dates {datetime}"
    items = search(datetime if collection.time != "static" else None)
    if not items and collection.time == "yearly" and datetime:
        items, note = search(None), f"no version in {datetime}: newest used"
    if prefer == "nearest":
        target = middle(datetime)
        items.sort(key=lambda it: abs(acquired(it) - target))
    return passes(items), note


def passes(items: list[pystac.Item], gap: pd.Timedelta = pd.Timedelta(minutes=5)) -> list[list[pystac.Item]]:
    """Group tiles of one satellite pass: same platform, acquired within `gap` of each other.

    Tiles of one pass can carry sensing times seconds apart (HLS does), so exact times do not match.
    Groups keep the order of the input.
    """
    groups: list[list[pystac.Item]] = []
    for item in items:
        platform, when = item.properties.get("platform"), acquired(item)
        for group in groups:
            if group[0].properties.get("platform") == platform and abs(acquired(group[0]) - when) <= gap:
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def tiles_for(scene: list[pystac.Item], cell: Cell, margin: float) -> list[tuple[str, list[pystac.Item]]]:
    """Ways to read one scene, best first: (window CRS, tiles) (docs/SPEC.md sections 3.1 and 6.4).

    First the projected zones whose tiles cover the whole window, the cell's own zone first, so pixels
    stay native; then zones that cover it only partly. Products in latitude/longitude use the cell's zone
    and are resampled. Each option uses tiles of one CRS, those containing the whole window first.
    Footprints in the catalogue can overstate the data (some are whole tile squares), so the caller
    checks the no-data it actually reads and moves on to the next option if needed.
    """
    target = cell_crs(cell)
    by_crs: dict[str | None, list[pystac.Item]] = {}
    for item in scene:
        by_crs.setdefault(item_crs(item), []).append(item)
    projected = sorted((c for c in by_crs if c and CRS.from_user_input(c).is_projected), key=lambda c: c != target)
    covering, partial = [], []
    for crs in projected:
        need = shapely.box(*transform_bounds(crs, "EPSG:4326", *window(cell, 10, margin=margin, crs=crs).bounds,
                                             densify_pts=21))
        footprints = {it.id: shapely.make_valid(shape(it.geometry)) for it in by_crs[crs]}   # some are invalid
        tiles = sorted(by_crs[crs], key=lambda it: not footprints[it.id].contains(need))
        (covering if shapely.union_all(list(footprints.values())).contains(need) else partial).append((crs, tiles))
    return covering + partial if projected else [(target, scene)]
