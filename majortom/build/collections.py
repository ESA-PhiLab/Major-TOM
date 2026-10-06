"""Planetary Computer collections the builder knows: assets, cloud masks and how to resample them.

Collections and cloud masks follow asterisk-labs/taco examples/workshop.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

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
    """
    assets: tuple[str, ...]
    time: str = "scene"
    clouds: tuple[str, MaskFn] | None = None
    resampling: str = "bilinear"
    res_m: float | dict[str, float] | None = None

    def target_res(self, asset: str) -> float | None:
        return self.res_m.get(asset) if isinstance(self.res_m, dict) else self.res_m


S2 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
COLLECTIONS = {
    "sentinel-2-l2a": Collection(S2, clouds=("SCL", s2_clouds)),
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
