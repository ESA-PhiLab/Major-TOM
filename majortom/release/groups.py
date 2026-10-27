"""From one built sample to its raster groups: the bands that share a pixel size (docs/RELEASE.md section 1).

    <build>/<product>/<index>.npz (one array per band) + .json sidecar
      -> pixel size per band: the window side (window_m) / the band's width in pixels
      -> groups by pixel size and data type, finest first: "10m", "20m", "60m"; a pixel size holding
         several data types gets one group per type, named "300m-uint16", "300m-uint8" (never cast)
      -> per group: data (bands, y, x), transform at that pixel size, CRS, band names, nodata,
         the stored-value offset and declared scaling as the build recorded them, and whether the values are
         categorical (resampled with "nearest": classes, codes, masks), which must stay lossless
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
    categorical: bool               # classes or codes: only lossless profiles apply


def _shared(values: dict | None, bands: list[str], what: str):
    """The one value the group's bands share (None if none recorded); an error if they differ."""
    if not values:
        return None
    found = {repr(values.get(b)) for b in bands}
    if len(found) > 1:
        raise ValueError(f"bands {bands} have different {what}: {sorted(found)}")
    return values.get(bands[0])


def _categorical(side: dict) -> bool:
    """Resampled with "nearest" (recorded by the build since 2026-10-27; older samples: the collection's setting)."""
    method = side.get("resampling")
    if method is None:
        try:
            from ..build.collections import COLLECTIONS
            method = COLLECTIONS[side["collection"]].resampling
        except (ImportError, KeyError) as error:
            raise ValueError(f"cannot tell whether {side.get('collection')} is categorical: no 'resampling' in the "
                             f"sidecar and the build collections are unavailable ({error})") from error
    return method == "nearest"


def groups(sample: str | Path) -> tuple[list[RasterGroup], dict]:
    """The raster groups of one built sample (finest pixel size first), and its sidecar."""
    sample = Path(sample)
    side = json.loads(sample.with_suffix(".json").read_text())
    arrays = np.load(sample)
    by_key: dict[tuple[float, str], list[str]] = {}
    for band in side["bands"]:
        res = round(side["window_m"] / arrays[band].shape[-1], 6)
        by_key.setdefault((res, str(arrays[band].dtype)), []).append(band)
    types_at = {}
    for res, dtype in by_key:
        types_at[res] = types_at.get(res, 0) + 1
    categorical = _categorical(side)
    legacy = side.get("query", {}).get("margin") == LEGACY_V1
    win = legacy_window(cell_from_name(side["grid_cell"], side["grid_km"]), side["crs"]) if legacy else None
    out = []
    for (res, dtype), bands in sorted(by_key.items()):
        transform = win.transform(res) if legacy else Affine(res, 0, side["left"], 0, -res, side["top"])
        if res == side["res"] and (transform.c, transform.f) != (side["left"], side["top"]):
            raise ValueError(f"{sample.name}: rebuilt corner {transform.c, transform.f} differs from the sidecar's")
        name = f"{res:g}m" if types_at[res] == 1 else f"{res:g}m-{dtype}"
        out.append(RasterGroup(name, np.stack([arrays[b] for b in bands]), transform, side["crs"], bands,
                               side.get("nodata"), _shared(side.get("offset"), bands, "offsets"),
                               _shared(side.get("scaling"), bands, "scaling"), categorical))
    return out, side
