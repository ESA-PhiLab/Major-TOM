"""Reading released rumi samples as arrays or as COGs (docs/RELEASE.md section 5).

    location (path, URI or bytes) + header (from the TACO table)
      -> read(as_="array")  rumi.read -> (bands, y, x)
      -> read(as_="cog")    + rumi.info (bands, CRS, transform from the file) -> write_cog into memory -> MemoryFile
      -> to_cog(items, out) the same, one file per item: <out>/<name>.tif

Nodata, scale and offset are not stored in rumi files: pass them from the sample's metadata.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

from .cog import write_cog


def _array(location, header) -> np.ndarray:
    """(bands, y, x): rumi returns a leading time axis for single-time files on some layouts; drop it."""
    import rumi

    data = np.asarray(rumi.read(location, header))
    return data[0] if data.ndim == 4 and data.shape[0] == 1 else data


def read(location, header: bytes, as_: str = "array", nodata: float | None = None, scale: float | None = None,
         offset: float | None = None):
    """One rumi sample file as a numpy array (bands, y, x), or as an in-memory COG (`as_="cog"`).

    The COG is a rasterio MemoryFile with the Major TOM profile; open it with `.open()` and use any rasterio or
    GDAL code. Close the MemoryFile when done (or use it as a context manager).
    """
    if as_ == "array":
        return _array(location, header)
    if as_ != "cog":
        raise ValueError(f"as_ must be 'array' or 'cog', not {as_!r}")
    import rumi

    meta = rumi.info(source=location, header=header)
    mem = MemoryFile()
    write_cog(mem, _array(location, header), Affine(*meta.transform), f"EPSG:{meta.crs}", list(meta.bands),
              nodata=nodata, scale=scale, offset=offset)
    return mem


def to_cog(items: Iterable[tuple], out: str | Path) -> list[Path]:
    """Write COG files for rumi samples: items of (name, location, header) or (name, location, header, metadata).

    `metadata` may hold nodata, scale and offset. Files: <out>/<name>.tif; returns their paths.
    """
    import rumi

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for item in items:
        name, location, header = item[:3]
        extra = item[3] if len(item) > 3 else {}
        meta = rumi.info(source=location, header=header)
        path = out / f"{name}.tif"
        write_cog(path, _array(location, header), Affine(*meta.transform), f"EPSG:{meta.crs}", list(meta.bands),
                  **{k: extra.get(k) for k in ("nodata", "scale", "offset")})
        written.append(path)
    return written
