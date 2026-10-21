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


@dataclass(frozen=True)
class Archive:
    """Where samples are searched and read.

    name        recorded in every sample
    stac_url    STAC API address
    modifier    applied to every item found, e.g. signing its links
    sorts       the API sorts results (sortby), whether or not it declares it
    gdal_env    GDAL settings for reading, fetched per read
    workers     parallel samples the archive allows
    """
    name: str
    stac_url: str
    modifier: Callable | None = None
    sorts: bool = True
    gdal_env: Callable[[], dict] = public_env
    workers: int = 16


ARCHIVES = {
    "planetary-computer": Archive("planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1",
                                  modifier=planetary_computer.sign_inplace),
}
