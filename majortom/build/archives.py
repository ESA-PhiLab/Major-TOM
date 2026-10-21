"""Archives the builder searches and reads from.

    Archive  ->  catalog(archive)       STAC client: address, link signing, sorting
             ->  archive.gdal_env()     GDAL settings for reading its files, credentials included

An Archive holds no credentials: gdal_env() fetches them when a read starts, and the reads use them only
inside a rasterio.Env block, so they never reach os.environ, sample metadata or error messages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import planetary_computer

from .read import GDAL_OPTIONS


def public_env() -> dict:
    """GDAL settings for archives whose files need no credentials."""
    return dict(GDAL_OPTIONS)


def unsigned_s3_env() -> dict:
    """GDAL settings for public S3 buckets: s3:// links are read without signing (AWS Open Data).

    JPEG 2000 files (Sentinel-2 L1C) are decoded in one thread: GDAL's decoder threads do not see settings
    made by rasterio.Env in a worker thread, so their unsigned requests fail. Samples still run in parallel.
    """
    return dict(GDAL_OPTIONS, AWS_NO_SIGN_REQUEST="YES", GDAL_NUM_THREADS="1")


@dataclass(frozen=True)
class Archive:
    """Where samples are searched and read.

    name        recorded in every sample
    stac_url    STAC API address
    modifier    applied to every item found, e.g. signing its links
    sorts       the API sorts results (sortby), whether or not it declares it
    sort_prefix put before sort fields (Earth Search sorts on "properties.eo:cloud_cover")
    gdal_env    GDAL settings for reading, fetched per read
    workers     parallel samples the archive allows
    """
    name: str
    stac_url: str
    modifier: Callable | None = None
    sorts: bool = True
    sort_prefix: str = ""
    gdal_env: Callable[[], dict] = public_env
    workers: int = 16


ARCHIVES = {
    "planetary-computer": Archive("planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1",
                                  modifier=planetary_computer.sign_inplace),
    # Element 84's catalogue of AWS Open Data buckets; some links are s3://, some https://
    "earth-search": Archive("earth-search", "https://earth-search.aws.element84.com/v1", sort_prefix="properties.",
                            gdal_env=unsigned_s3_env),
}
