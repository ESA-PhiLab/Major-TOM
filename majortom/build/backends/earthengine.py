"""Google Earth Engine (needs `pip install 'majortom[build,gee]'`).

    search     ImageCollection(s).filterBounds.filterDate.sort.limit -> one getInfo -> STAC-like items
    header     CRS and grid of a band, from the image's band list (no file to open)
    read_tile  ee.data.computePixels onto the window grid: values and a coverage mask in one request;
               nearest neighbour (Earth Engine's default) where the band is in the window's CRS, else resampled
    session    signs in once per process (service-account key file, or your own login) on the high-volume API

Each image becomes a pystac.Item whose assets are the bands we save, under our names, with links
"ee://<image id>#<Earth Engine band>" and the band's proj:code and proj:transform. Naming per collection (bands,
cloud, satellite and tile properties, scaling) is in collections.py EARTH_ENGINE. A merged search is named
"id+id+...".
"""
from __future__ import annotations

import datetime as dt
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd
import pystac
from rasterio.crs import CRS
from rasterio.transform import Affine

from ..collections import EARTH_ENGINE, EarthEngine
from ..credentials import earth_engine_settings
from . import retry

if TYPE_CHECKING:
    from ...spec import Window
    from ..archives import Archive

HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"
KNOWN_CRS = {"SR-ORG:6974": "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m +no_defs"}   # MODIS
_signed_in = threading.Lock()
_ready: list[bool] = []


def _ee():
    import ee                                            # optional dependency, imported when first used
    return ee


@contextmanager
def session(archive: Archive) -> Iterator[None]:
    """Sign in to Earth Engine once per process; later sessions reuse it."""
    project, key_file = earth_engine_settings()
    with _signed_in:
        if not _ready:
            ee = _ee()
            credentials = None
            if key_file:
                import json
                from pathlib import Path
                email = json.loads(Path(key_file).read_text())["client_email"]
                credentials = ee.ServiceAccountCredentials(email, key_file)
            ee.Initialize(credentials, project=project, opt_url=HIGH_VOLUME)
            _ready.append(True)
    yield


def _dates(datetime: str) -> tuple[str, str]:
    """A STAC datetime ("start/end", or one time) as Earth Engine's [start, end) with whole end days included."""
    ends = [t for t in datetime.split("/") if t not in ("", "..")]
    start, end = pd.Timestamp(ends[0]), pd.Timestamp(ends[-1])
    if len(ends[-1]) <= 10:                               # a date: include the whole day
        end += pd.Timedelta(days=1)
    else:
        end += pd.Timedelta(milliseconds=1)
    return start.isoformat(), end.isoformat()


def _platform(props: dict, naming: EarthEngine) -> str | None:
    value = props.get(naming.platform) if naming.platform else None
    return None if value is None else (naming.prefix + str(value)).lower().replace("_", "-")


def _tile(props: dict, naming: EarthEngine) -> str | None:
    if isinstance(naming.tile, tuple):
        path, row = (props.get(p) for p in naming.tile)
        return None if path is None else f"WRS2-{int(path):03d}{int(row):03d}"
    return f"MGRS-{props[naming.tile]}" if naming.tile and props.get(naming.tile) else None


def to_item(image: dict, collection_id: str) -> pystac.Item:
    """One image of an ImageCollection getInfo as an item: properties the scene choice reads, one asset per band."""
    naming, props = EARTH_ENGINE[collection_id], image["properties"]
    footprint = props.get("system:footprint") or {}
    geometry = {"type": "Polygon", "coordinates": [footprint["coordinates"]]} if footprint.get("coordinates") else None
    when = dt.datetime.fromtimestamp(props["system:time_start"] / 1000, tz=dt.timezone.utc)
    bands = {b["id"]: b for b in image["bands"]}
    first = next(iter(bands.values()))["crs"] if bands else None
    item = pystac.Item(image["id"], geometry, None, when, {
        "eo:cloud_cover": props.get(naming.cloud) if naming.cloud else None,
        "platform": _platform(props, naming), "grid:code": _tile(props, naming),
        # only EPSG codes count as the scene's CRS: others (MODIS sinusoidal) are resampled into the cell's zone,
        # as from Planetary Computer, which declares no code for them
        "proj:code": first if first and first.startswith("EPSG:") else None,
        "s2:processing_baseline": props.get("PROCESSING_BASELINE"), "ee:collection": collection_id})
    for ours, theirs in naming.bands.items():
        if theirs in bands:
            crs = bands[theirs]["crs"]
            extra = {"proj:code": crs if crs.startswith("EPSG:") else None, "ee:crs": crs,
                     "proj:transform": bands[theirs].get("crs_transform")}
            if ours in naming.scaling:                   # declared like STAC raster:bands, read the same way
                scale, offset = naming.scaling[ours]
                extra["raster:bands"] = [{"scale": scale, "offset": offset}]
            item.add_asset(ours, pystac.Asset(f"ee://{image['id']}#{theirs}", extra_fields=extra))
    return item


def search(archive: Archive, collection: str, sortby: list[dict] | None = None, bbox=None, datetime: str | None = None,
           max_items: int = 100) -> list[pystac.Item]:
    """One Earth Engine request: the collection(s) filtered, sorted and limited, as items."""
    ee = _ee()
    ids = collection.split("+")
    with session(archive):                               # signed in before any Earth Engine object is made
        images = None
        for cid in ids:
            part = ee.ImageCollection(cid).map(lambda image, cid=cid: image.set("majortom:collection", cid))
            if EARTH_ENGINE[cid].link:                   # properties of the image with the same id elsewhere
                linked, prop = EARTH_ENGINE[cid].link
                part = part.linkCollection(ee.ImageCollection(linked), [], [prop])
            images = part if images is None else images.merge(part)
        if bbox is not None:
            images = images.filterBounds(ee.Geometry.Rectangle(list(bbox)))
            if all(EARTH_ENGINE[cid].zone_index for cid in ids):
                zone = int((bbox[0] + bbox[2]) / 2 + 180) // 6 + 1
                near = [(zone - 1 + k) % 60 + 1 for k in (-1, 0, 1)]
                images = images.filter(ee.Filter.Or(*[ee.Filter.stringStartsWith("system:index", f"T{z:02d}")
                                                      for z in near]))
        if datetime:
            images = images.filterDate(*_dates(datetime))
        for order in (sortby or [])[:1]:
            prop = "system:time_start" if order["field"] == "datetime" else EARTH_ENGINE[ids[0]].cloud
            if prop:
                images = images.sort(prop, order.get("direction", "asc") == "asc")
        info = retry(lambda: images.limit(max_items).getInfo(), archive.retries)
    return [to_item(image, image["properties"]["majortom:collection"]) for image in info["features"]]


def _parse(href: str) -> tuple[str, str]:
    image_id, band = href.removeprefix("ee://").rsplit("#", 1)
    return image_id, band


def header(archive: Archive, item: pystac.Item, key: str) -> tuple[CRS, Affine]:
    """CRS and grid of one band, as Earth Engine listed it."""
    asset = item.assets[key]
    code = asset.extra_fields["ee:crs"]
    return CRS.from_user_input(KNOWN_CRS.get(code, code)), Affine(*asset.extra_fields["proj:transform"][:6])


def read_tile(archive: Archive, item: pystac.Item, key: str, win: Window, res: float,
              method: str) -> tuple[np.ndarray, np.ndarray]:
    """One band onto the window grid: (bands, covered pixels). Pixels without data are 0."""
    ee = _ee()
    image_id, band = _parse(item.assets[key].href)
    n = win.pixels(res)
    grid = {"crsCode": win.crs, "dimensions": {"width": n, "height": n},
            "affineTransform": {"scaleX": res, "shearX": 0, "translateX": win.left,
                                "shearY": 0, "scaleY": -res, "translateY": win.top}}
    with session(archive):
        image = ee.Image(image_id).select(band)
        if item.assets[key].extra_fields["ee:crs"] != win.crs and method != "nearest":
            image = image.resample(method)               # Earth Engine's default is nearest neighbour
        request = {"expression": image.unmask(0).addBands(image.mask().rename("majortom_covered")),
                   "fileFormat": "NUMPY_NDARRAY", "grid": grid}
        pixels = retry(lambda: ee.data.computePixels(request))
    values = pixels[band]
    stored = EARTH_ENGINE[item.properties["ee:collection"]].stored
    if stored and values.dtype.kind == "f":              # reflectance back to the integers other archives store;
        values = np.round(values * stored[0]).astype(stored[1])   # integer bands (HLS Fmask) are as they are
    return values[np.newaxis], pixels["majortom_covered"] > 0
