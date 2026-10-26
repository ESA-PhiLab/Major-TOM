"""The Major TOM COG profile (docs/RELEASE.md section 3): one definition, used for releases and conversions.

    array (bands, y, x) + transform + CRS + band names [+ nodata, scale, offset]
      -> write_cog(dest)  dest: a path, or a rasterio MemoryFile
      -> COG with 176 px tiles, tile interleave, ZSTD + predictor (lossless) or LERC (near-lossless), no overviews
      -> read back: fails if GDAL dropped tile interleave (GDAL < 3.11 writes pixel interleave without an error)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile

COG_PROFILE = dict(driver="COG", BLOCKSIZE=176, INTERLEAVE="TILE", OVERVIEWS="NONE")   # 1056 = 6 x 176 (3.1)


def write_cog(dest: str | Path | MemoryFile, data: np.ndarray, transform: Affine, crs: str, bands: list[str],
              nodata: float | None = None, scale: float | None = None, offset: float | None = None,
              max_error: float | None = None) -> None:
    """Write one raster group as a COG with the Major TOM profile.

    Lossless by default: ZSTD with predictor 2 (integers) or 3 (floats). `max_error` > 0 writes LERC with that
    absolute bound instead (near-lossless; plain LERC, not LERC_ZSTD: docs/RELEASE.md section 4).
    """
    if max_error:
        codec = dict(COMPRESS="LERC", MAX_Z_ERROR=max_error)
    else:
        codec = dict(COMPRESS="ZSTD", PREDICTOR=3 if np.issubdtype(data.dtype, np.floating) else 2)
    profile = dict(COG_PROFILE, width=data.shape[2], height=data.shape[1], count=data.shape[0], dtype=data.dtype,
                   crs=crs, transform=transform, nodata=nodata, **codec)
    opener = dest.open if isinstance(dest, MemoryFile) else lambda **kw: rasterio.open(dest, "w", **kw)
    with opener(**profile) as f:
        f.write(data)
        f.descriptions = tuple(bands)
        if scale is not None or offset is not None:
            f.scales = (1.0 if scale is None else scale,) * data.shape[0]
            f.offsets = (0.0 if offset is None else offset,) * data.shape[0]
    reopen = dest.open if isinstance(dest, MemoryFile) else lambda: rasterio.open(dest)
    with reopen() as f:
        layout = f.tags(ns="IMAGE_STRUCTURE").get("INTERLEAVE")
    if layout != "TILE":
        raise RuntimeError(f"GDAL {rasterio.__gdal_version__} wrote INTERLEAVE={layout}; the Major TOM COG profile "
                           "needs GDAL >= 3.11 (rasterio >= 1.5)")
