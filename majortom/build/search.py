"""Find scenes over a window, in any archive: order, passes, zones and the same pass in another product.

A scene is all tiles of one acquisition (same time); tiles of one scene can be stitched together. The search
itself is the archive's backend (backends/); everything here works on the STAC items it returns.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import pystac
import shapely
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from ..cells import Cell
from ..spec import cell_crs, window
from .backends import backend
from .collections import Collection, Source

if TYPE_CHECKING:
    from .archives import Archive


def acquired(item: pystac.Item) -> pd.Timestamp:
    """Acquisition time in UTC."""
    when = pd.Timestamp(item.datetime or item.properties["start_datetime"])
    return when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")


MGRS_IN_ID = re.compile(r"(?:^|[._/])T(\d{2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})(?:[._]|$)")   # HLS.S30.T12TVK.2024...


def item_crs(item: pystac.Item) -> str | None:
    """The tile's CRS from its STAC metadata, without opening a file: the item's, else its assets' (STAC 1.1
    catalogues such as CDSE put proj:code on each asset), else the UTM zone of an MGRS tile in its id (NASA's
    HLS records give no CRS)."""
    for fields in [item.properties, *(asset.extra_fields for asset in item.assets.values())]:
        code, epsg = fields.get("proj:code"), fields.get("proj:epsg")
        if code or epsg:
            return code or f"EPSG:{epsg}"
    match = MGRS_IN_ID.search(item.id)
    if match:                                           # latitude bands N and later are north of the equator
        return f"EPSG:{(32600 if match.group(2) >= 'N' else 32700) + int(match.group(1))}"
    return None


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


def scenes(archive: Archive, source: Source, collection: Collection, bbox: tuple[float, float, float, float],
           datetime: str | None, prefer: str = "clearest", limit: int = 100) -> tuple[list[list[pystac.Item]], str]:
    """Scenes over a lon/lat box in the order of `prefer`, and a note on which dates were used.

    A scene is the list of tiles of one acquisition. Products without clouds are ordered newest first
    for "clearest" and "cloudiest". Yearly products take the newest version if none is in `datetime`.
    """
    if not (collection.clouds or collection.clouds_from) and prefer in ("clearest", "cloudiest"):
        prefer = "newest"
    bands = set(collection.assets) | ({collection.clouds[0]} if collection.clouds else set())
    needed = {source.asset(band) for band in bands}                  # this archive's asset names

    def search(dates: str | None) -> list[pystac.Item]:
        query = dict(bbox=bbox, max_items=limit, sortby=ORDER[prefer])
        if dates:
            query["datetime"] = dates
        return [it for it in backend(archive).search(archive, source.collection, **query) if needed <= set(it.assets)]

    note = "dates ignored: one version" if collection.time == "static" else f"dates {datetime}"
    items = search(datetime if collection.time != "static" else None)
    if not items and collection.time == "yearly" and datetime:
        items, note = search(None), f"no version in {datetime}: newest used"
    if prefer == "nearest":
        target = middle(datetime)
        items.sort(key=lambda it: abs(acquired(it) - target))
    return passes(items), note


def tile_code(item: pystac.Item) -> str | None:
    """The product tile an item covers (Sentinel-2: its MGRS square)."""
    code = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile")
    match = None if code else MGRS_IN_ID.search(item.id)
    return f"MGRS-{''.join(match.groups())}" if match else code


def same_pass(archive: Archive, source: Source, scene: list[pystac.Item], bbox: tuple[float, float, float, float],
              slack: pd.Timedelta = pd.Timedelta(seconds=60)) -> list[pystac.Item]:
    """Tiles of another product from the same pass as `scene`: same tile and time (within `slack`), and the same
    platform where both name one (Dynamic World does not)."""
    times = [acquired(it) for it in scene]
    dates = f"{(min(times) - slack).isoformat()}/{(max(times) + slack).isoformat()}"
    found = backend(archive).search(archive, source.collection, bbox=bbox, datetime=dates, max_items=50)
    want = {(tile_code(it), it.properties.get("platform")) for it in scene}
    tiles = {tile for tile, _ in want}
    if any(platform is None for _, platform in want):                  # match on tile alone
        return [it for it in found if tile_code(it) in tiles]
    return [it for it in found if (tile_code(it), it.properties.get("platform")) in want]


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


def near_zone(crs: str | None, target: str, zones: int = 2) -> bool:
    """Whether a UTM CRS is within `zones` zones of the cell's (other CRSs: always). A tile in a far zone holds
    no window of this cell; catalogues return such tiles when a footprint crossing 180° covers the globe."""
    def zone(code):
        epsg = int(code.split(":")[1]) if code and code.startswith("EPSG:") else 0
        return epsg % 100 if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760 else None
    a, b = zone(crs), zone(target)
    return a is None or b is None or min(abs(a - b), 60 - abs(a - b)) <= zones


def tiles_for(scene: list[pystac.Item], cell: Cell, margin: float) -> list[tuple[str, list[pystac.Item]]]:
    """Ways to read one scene, best first: (window CRS, tiles) (docs/SPEC.md sections 3.1 and 6.4).

    First the projected zones whose tiles cover the whole window, the cell's own zone first, so pixels
    stay native; then zones that cover it only partly. Products in latitude/longitude use the cell's zone
    and are resampled. Each option uses tiles of one CRS, those containing the whole window first, then
    by tile code.
    Footprints in the catalogue can overstate the data (some are whole tile squares), so the caller
    checks the no-data it actually reads and moves on to the next option if needed.
    """
    target = cell_crs(cell)
    by_crs: dict[str | None, list[pystac.Item]] = {}
    for item in scene:
        crs = item_crs(item)
        if not near_zone(crs, target):                  # broken footprints (across 180°) match everywhere
            continue
        by_crs.setdefault(crs, []).append(item)
    projected = sorted((c for c in by_crs if c and CRS.from_user_input(c).is_projected), key=lambda c: c != target)
    covering, partial = [], []
    for crs in projected:
        need = shapely.box(*transform_bounds(crs, "EPSG:4326", *window(cell, 10, margin=margin, crs=crs).bounds,
                                             densify_pts=21))
        footprints = {it.id: shapely.make_valid(shape(it.geometry)) for it in by_crs[crs]}   # some are invalid
        # tiles containing the window first, then by tile code: the same order from every archive, which
        # matters where tiles overlap (the first tile's pixels are kept, and L1C tiles differ slightly there)
        tiles = sorted(by_crs[crs], key=lambda it: (not footprints[it.id].contains(need), tile_code(it) or it.id))
        (covering if shapely.union_all(list(footprints.values())).contains(need) else partial).append((crs, tiles))
    return covering + partial if projected else [(target, scene)]
