"""From one built sample to its raster groups: the bands that share a pixel size (docs/RELEASE.md section 1).

    <build>/<product>/<index>.npz (one array per band) + .json sidecar
      -> pixel size per band: the window side (window_m) / the band's width in pixels
      -> groups by pixel size, finest first: "10m", "20m", "60m" (any product, not only Sentinel-2)
      -> per group: data (bands, y, x), transform at that pixel size, CRS, band names, nodata,
         the stored-value offset and declared scaling as the build recorded them
Transforms: a v2 window has one corner on every grid (the sidecar's left, top); a Core v1 window (margin "v1")
is rounded to each pixel size's own grid, so it is rebuilt with legacy_window (docs/SPEC.md 7.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine

from ..cells import cell_from_name
from ..spec import LEGACY_V1, legacy_window


@dataclass(frozen=True)
class RasterGroup:
    name: str                       # "10m": the file is rumi/10m.rumi (or cog/10m.tif)
    data: np.ndarray                # (bands, y, x)
    transform: Affine
    crs: str
    bands: list[str]
    nodata: float | None
    stored_offset: float | None     # shift carried by stored values (Sentinel-2 from baseline 04.00: +1000)
    scaling: dict | None            # declared by the archive: physical = stored x scale + offset


def _shared(values: dict | None, bands: list[str], what: str):
    """The one value the group's bands share (None if none recorded); an error if they differ."""
    if not values:
        return None
    found = {repr(values.get(b)) for b in bands}
    if len(found) > 1:
        raise ValueError(f"bands {bands} have different {what}: {sorted(found)}")
    return values.get(bands[0])


def groups(sample: str | Path) -> tuple[list[RasterGroup], dict]:
    """The raster groups of one built sample (finest pixel size first), and its sidecar."""
    sample = Path(sample)
    side = json.loads(sample.with_suffix(".json").read_text())
    arrays = np.load(sample)
    by_res: dict[float, list[str]] = {}
    for band in side["bands"]:
        res = round(side["window_m"] / arrays[band].shape[-1], 6)
        by_res.setdefault(res, []).append(band)
    legacy = side.get("query", {}).get("margin") == LEGACY_V1
    win = legacy_window(cell_from_name(side["grid_cell"], side["grid_km"]), side["crs"]) if legacy else None
    out = []
    for res, bands in sorted(by_res.items()):
        transform = win.transform(res) if legacy else Affine(res, 0, side["left"], 0, -res, side["top"])
        if res == side["res"] and (transform.c, transform.f) != (side["left"], side["top"]):
            raise ValueError(f"{sample.name}: rebuilt corner {transform.c, transform.f} differs from the sidecar's")
        out.append(RasterGroup(f"{res:g}m", np.stack([arrays[b] for b in bands]), transform, side["crs"], bands,
                               side.get("nodata"), _shared(side.get("offset"), bands, "offsets"),
                               _shared(side.get("scaling"), bands, "scaling")))
    return out, side
