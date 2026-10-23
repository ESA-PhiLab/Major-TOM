"""Sentinel-5P OFFL L3 on Earth Engine: one UTC-day mosaic per product, served on a 0.05° grid.

    source day (UTC date) -> ImageCollection(product).filterDate(day, day + 1).mean()    (orbits averaged)
      -> 32 tiles of 900 x 900 at 0.05° (computePixels, in threads) -> one global field, NaN where no data

Cost measured 2026-10-23 (docs/evidence/s5p_cost.py): ~1.5 EECU-s per tile, ~50 per day and product.
A daily step takes the UTC day of the overpass (about 13:30 local solar time) on the local day of the step's
window centre. Project and key: majortom.build.credentials.earth_engine_settings (EE_PROJECT, e.g.
major-tom-climate; EE_SERVICE_ACCOUNT_KEY).
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

H = np.timedelta64(1, "h")
TILE = 900
_signed_in: list[bool] = []
_lock = threading.Lock()


def _ee():
    import ee

    from ...build.credentials import earth_engine_settings
    with _lock:
        if not _signed_in:
            project, key_file = earth_engine_settings()
            email = json.loads(Path(key_file).read_text())["client_email"] if key_file else None
            ee.Initialize(ee.ServiceAccountCredentials(email, key_file) if key_file else None, project=project)
            _signed_in.append(True)
    return ee


def valid_range(source) -> tuple[str, str]:
    """Days with data: before the first OFFL products (2018-06-28) a day is empty without asking Earth Engine."""
    return source.options["first_day"], "2100-01-01"


def groups(source, acquired: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.zeros(len(acquired), dtype=int)


def days(source, centres: np.ndarray, lon: np.ndarray, group: np.ndarray) -> np.ndarray:
    """UTC date of the overpass on the local day of each window centre."""
    shift = (lon / 15 * 3600).astype("timedelta64[s]")              # local solar time - UTC
    local_day = (centres + shift).astype("datetime64[D]")
    overpass = local_day + np.timedelta64(int(source.options["overpass_local"] * 3600), "s") - shift
    return overpass.astype("datetime64[D]").astype(np.int64)


def period_start(source, day: np.ndarray, group: np.ndarray) -> np.ndarray:
    return np.asarray(day).astype("datetime64[D]").astype("datetime64[s]")


def field(source, variable: str, key) -> np.ndarray:
    """The UTC-day mosaic of one product on the global 0.05° grid."""
    ee = _ee()
    day, _ = key
    date = str(np.datetime64(int(day), "D"))
    start = ee.Date(date)
    band = source.variables[variable]
    day_images = ee.ImageCollection(source.options["collections"][variable]).filterDate(start, start.advance(1, "day"))
    image = ee.Image(ee.Algorithms.If(day_images.size().gt(0), day_images.select(band).mean(),
                                      ee.Image.constant(-9999).rename(band))).unmask(-9999)     # no orbits: empty
    grid = source.grid
    out = np.full((grid.height, grid.width), np.nan, np.float32)

    def tile(r: int, c: int) -> None:
        block = ee.data.computePixels({
            "expression": image, "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": TILE, "height": TILE}, "crsCode": "EPSG:4326",
                     "affineTransform": {"scaleX": grid.step, "shearX": 0, "translateX": grid.west + c * TILE * grid.step,
                                         "shearY": 0, "scaleY": -grid.step, "translateY": grid.north - r * TILE * grid.step}}})
        values = block[block.dtype.names[0]] if block.dtype.names else block
        values = values.astype(np.float32)
        values[values == -9999] = np.nan
        out[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = values

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda rc: tile(*rc), [(r, c) for r in range(grid.height // TILE) for c in range(grid.width // TILE)]))
    return out
