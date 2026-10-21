"""Products the builder knows: bands, cloud masks, how to resample them, and where each archive keeps them.

A product's band names are the same from every archive (Sentinel-2: B01..B12, SCL); a Source maps them to
the archive's collection and asset keys. Collections and cloud masks follow asterisk-labs/taco
examples/workshop.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pystac

MaskFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]


def s2_clouds(scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sentinel-2 scene classes: (cloud shadow, medium and high cloud, cirrus), (no data)."""
    return np.isin(scl, [3, 8, 9, 10]), scl == 0


def landsat_clouds(qa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Landsat QA_PIXEL: (bits 1-4: dilated cloud, cirrus, cloud, shadow), (bit 0: fill)."""
    return (qa & 0b11110) > 0, (qa & 1) > 0


def hls_clouds(fmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HLS Fmask: (bits 1-3: cloud, adjacent to cloud, shadow), (255: fill)."""
    return (fmask & 0b1110) > 0, fmask == 255


def declared_or_baseline_offset(item: pystac.Item, key: str) -> float | None:
    """Sentinel-2 reflectance offset in stored values: as declared by the archive, else from the baseline.

    Declared: raster:bands offset and scale (Earth Search: offset -0.1 at scale 0.0001, i.e. +1000).
    Not declared (Planetary Computer): processing baseline 04.00 or later adds 1000 (from 2022-01-25).
    """
    bands = item.assets[key].extra_fields.get("raster:bands") or [{}]
    offset, scale = bands[0].get("offset"), bands[0].get("scale")
    if offset is not None and scale:
        return float(round(-offset / scale))
    baseline = item.properties.get("s2:processing_baseline")
    return None if baseline is None else (1000.0 if baseline >= "04.00" else 0.0)


@dataclass(frozen=True)
class Source:
    """One product in one archive: its collection id, and the asset key of each band (default: the band name)."""
    collection: str
    assets: dict[str, str] = field(default_factory=dict)

    def key(self, band: str) -> str:
        return self.assets.get(band, band)


@dataclass(frozen=True)
class Collection:
    """What the builder needs to know about one collection.

    assets      bands to save
    time        "scene": dates select acquisitions; "yearly": dates select the version (the newest if none
                falls in them); "static": one version, dates ignored
    clouds      (mask asset, function giving cloud and no-data pixels) used to rank scenes; not saved
    resampling  method when the data are not already on the cell's pixel grid: "nearest" for classes,
                "bilinear" for continuous values
    res_m       pixel size in metres when the product is in latitude/longitude: one value, or per asset
    sources     archive name -> Source; None: Planetary Computer, under the product's own name
    offset      bands whose stored values carry an offset, and the function giving it (item, asset key)
    """
    assets: tuple[str, ...]
    time: str = "scene"
    clouds: tuple[str, MaskFn] | None = None
    resampling: str = "bilinear"
    res_m: float | dict[str, float] | None = None
    sources: dict[str, Source] | None = None
    offset: tuple[tuple[str, ...], Callable[[pystac.Item, str], float | None]] | None = None

    def source(self, name: str, archive: str) -> Source | None:
        """Where `archive` keeps this product (`name` is the product's name), or None."""
        return {"planetary-computer": Source(name)}.get(archive) if self.sources is None else self.sources.get(archive)

    def archives(self) -> list[str]:
        return ["planetary-computer"] if self.sources is None else sorted(self.sources)

    def target_res(self, asset: str) -> float | None:
        return self.res_m.get(asset) if isinstance(self.res_m, dict) else self.res_m


S2 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_L1C = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
S2_EARTH_SEARCH = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
                   "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
                   "B10": "cirrus", "B11": "swir16", "B12": "swir22", "SCL": "scl"}
COLLECTIONS = {
    "sentinel-2-l2a": Collection(S2, clouds=("SCL", s2_clouds), offset=(S2, declared_or_baseline_offset),
                                 sources={"planetary-computer": Source("sentinel-2-l2a"),
                                          "earth-search": Source("sentinel-2-c1-l2a", S2_EARTH_SEARCH)}),
    "sentinel-2-l1c": Collection(S2_L1C, offset=(S2_L1C, declared_or_baseline_offset),
                                 sources={"earth-search": Source("sentinel-2-l1c", S2_EARTH_SEARCH)}),
    "cop-dem-glo-30": Collection(("DEM",), time="static", res_m=30,
                                 sources={"planetary-computer": Source("cop-dem-glo-30", {"DEM": "data"}),
                                          "earth-search": Source("cop-dem-glo-30", {"DEM": "data"})}),
    "landsat-c2-l2": Collection(("coastal", "blue", "green", "red", "nir08", "swir16", "swir22"),
                                clouds=("qa_pixel", landsat_clouds)),
    "hls2-s30": Collection(("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"),
                           clouds=("Fmask", hls_clouds)),
    "hls2-l30": Collection(("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10", "B11"),
                           clouds=("Fmask", hls_clouds)),
    "alos-palsar-mosaic": Collection(("HH", "HV"), time="yearly", res_m=25),
    "nasadem": Collection(("elevation",), time="static", res_m=30),
    "alos-dem": Collection(("data",), time="static", res_m=30),
    "hgb": Collection(("aboveground", "belowground", "aboveground_uncertainty", "belowground_uncertainty"),
                      time="static", res_m=300),
    "alos-fnf-mosaic": Collection(("C",), time="yearly", resampling="nearest", res_m=25),
    "esa-worldcover": Collection(("map", "input_quality"), time="yearly", resampling="nearest",
                                 res_m={"map": 10, "input_quality": 60}),
    "io-lulc-annual-v02": Collection(("data",), time="yearly", resampling="nearest", res_m=10),
    "esa-cci-lc": Collection(("lccs_class", "change_count", "observation_count", "current_pixel_state",
                              "processed_flag"), time="yearly", resampling="nearest", res_m=300),
    "jrc-gsw": Collection(("occurrence", "change", "seasonality", "recurrence", "transitions", "extent"),
                          time="static", resampling="nearest", res_m=30),
}
