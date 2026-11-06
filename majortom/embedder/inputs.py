"""Inputs for MajorTOM_Embedder: one Image from a Core row, a majortom.build sample, or a TACO record.

    Core image Parquet row (GeoTIFF bytes per band) --from_core_row--+
    majortom.build sample (.npz + .json sidecar) ------from_sample---+--> Image --> MajorTOM_Embedder.embed(image)
    TACO record (rumi files per pixel-size group) -----from_taco-----+

An Image holds the bands one model needs, stacked at the finest pixel size (coarser bands upsampled by nearest
neighbour, as for Core), with the georeferencing of that grid and the ids written into each embedding row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from affine import Affine


@dataclass
class Image:
    """Bands of one sample, ready to fragment and embed."""

    data: np.ndarray            # (height, width, bands), float32, in the order the model asks for
    transform: Affine           # of the finest band's pixel grid
    crs: str                    # e.g. "EPSG:32630"
    product_id: str             # source product(s); ";"-joined when a sample uses several
    timestamp: str              # acquisition time as YYYYMMDDTHHMMSS (the Core format)
    grid_cell: str              # e.g. "624U_29L"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in the image CRS."""
        h, w = self.data.shape[:2]
        left, top = self.transform.c, self.transform.f
        return left, top + h * self.transform.e, left + w * self.transform.a, top


def stack(arrays: list[np.ndarray]) -> np.ndarray:
    """2-D bands -> (height, width, bands) float32; coarser bands upsampled (nearest) to the largest shape."""
    import cv2

    h, w = max(a.shape[0] for a in arrays), max(a.shape[1] for a in arrays)
    same = [a if a.shape == (h, w) else cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST) for a in arrays]
    return np.stack(same, -1).astype(np.float32)


def core_time(value: str) -> str:
    """An ISO 8601 time (as in majortom.build sidecars and TACO catalogues) -> YYYYMMDDTHHMMSS."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%S")


def from_core_row(row, row_meta, bands: list[str]) -> Image:
    """A Core image Parquet row group (pyarrow table of one row) and its metadata row (one-row DataFrame)."""
    from rasterio.io import MemoryFile

    arrays, finest = [], None
    for band in bands:
        with MemoryFile(row[band][0].as_py()) as mem, mem.open(driver="GTiff") as f:
            arrays.append(f.read(1))
            if finest is None or f.res[0] < finest[0].a:
                finest = (f.transform, f.crs.to_string())
    return Image(stack(arrays), finest[0], finest[1], row_meta.product_id.item(), row_meta.timestamp.item(),
                 row_meta.grid_cell.item())


def from_sample(sample: str | Path, bands: list[str]) -> Image:
    """A majortom.build sample: <index>.npz (one array per band) and its <index>.json sidecar."""
    sample = Path(sample)
    side = json.loads(sample.with_suffix(".json").read_text())
    arrays = np.load(sample)
    missing = [b for b in bands if b not in arrays]
    if missing:
        raise KeyError(f"{sample.name} ({side['collection']}) has no bands {missing}; it has {sorted(arrays)}")
    data = stack([arrays[b] for b in bands])
    res = side["window_m"] / data.shape[0]
    return Image(data, Affine(res, 0, side["left"], 0, -res, side["top"]), side["crs"], ";".join(side["items"]),
                 core_time(side["datetime"]), side["grid_cell"])


def from_taco(record: dict, bands: list[str]) -> Image:
    """A TACO record (one catalogue row as a dict) of a majortom.release dataset.

    Every `<group>.rumi::location` / `::header` pair is opened; rumi records each file's bands, transform and CRS.
    """
    import rumi
    from majortom.release import read

    found: dict[str, np.ndarray] = {}
    finest = None
    for key in record:
        if not key.endswith(".rumi::location") or record[key] is None:
            continue
        location, header = record[key], record[key.replace("::location", "::header")]
        info = rumi.info(source=location, header=header)
        wanted = [b for b in info.bands if b in bands]
        if not wanted:
            continue
        array = read(location, header)                                  # (bands, y, x)
        for b in wanted:
            found[b] = array[list(info.bands).index(b)]
        transform = Affine(*info.transform)
        if finest is None or transform.a < finest[0].a:
            finest = (transform, f"EPSG:{info.crs}")
    missing = [b for b in bands if b not in found]
    if missing:
        raise KeyError(f"{record.get('id')} has no bands {missing}")
    items = record.get("source:items")
    product_id = ";".join(items) if items is not None and len(items) else str(record.get("id"))
    cell = str(record["majortom:code"]).split("_", 1)[1]                # MT10km_0624U_0029L -> 0624U_0029L
    row, col = cell.split("_")
    return Image(stack([found[b] for b in bands]), finest[0], finest[1], product_id,
                 core_time(record.get("source:acquired") or record["stac:datetime"]),
                 f"{int(row[:-1])}{row[-1]}_{int(col[:-1])}{col[-1]}")
