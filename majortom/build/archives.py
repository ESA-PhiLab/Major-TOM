"""Archives the builder searches and reads from: who they are, how they are reached, what they allow.

    Archive  ->  backend           access method, a module in backends/ (stac: STAC API + GDAL)
             ->  stac_url, ...     that method's settings: address, link signing, sorting, GDAL settings
             ->  workers, ...      what the archive allows: parallel samples, retries, search spacing

An Archive holds no credentials: they are looked up (credentials.py) when a backend session starts, and live
only inside it. Set them up once with `python -m majortom.build.credentials cdse` (or `earthdata`,
`earth-engine`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import planetary_computer

from .backends.stac import cdse_env, earthdata_env, public_env, unsigned_s3_env


@dataclass(frozen=True)
class Archive:
    """Where samples are searched and read.

    name        recorded in every sample
    backend     access method: a module in backends/ ("stac", "earthengine")
    stac_url    STAC API address
    modifier    applied to every item found, e.g. signing its links
    sorts       the API sorts results (sortby), whether or not it declares it
    sort_prefix put before sort fields (Earth Search sorts on "properties.eo:cloud_cover")
    gdal_env    GDAL settings for reading, fetched when a session starts
    workers     parallel samples the archive allows
    retries     attempts per catalogue search (waits 1, 2, 4, ... s; for rate limits)
    search_gap  least seconds between two catalogue searches, across threads (rate limits)
    refused     added to a sample's error when reading is refused (HTTP 401/403), e.g. how to fix the keys
    page        the archive's web page for a collection id, for docs/ARCHIVES.md
    """
    name: str
    stac_url: str = ""
    backend: str = "stac"
    modifier: Callable | None = None
    sorts: bool = True
    sort_prefix: str = ""
    gdal_env: Callable[[], dict] = public_env
    workers: int = 16
    retries: int = 4
    search_gap: float = 0.0
    refused: str = ""
    page: Callable[[str], str] | None = None


ARCHIVES = {
    "planetary-computer": Archive("planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1",
                                  modifier=planetary_computer.sign_inplace,
                                  page=lambda c: f"https://planetarycomputer.microsoft.com/dataset/{c}"),
    # Element 84's catalogue of AWS Open Data buckets; some links are s3://, some https://
    "earth-search": Archive("earth-search", "https://earth-search.aws.element84.com/v1", sort_prefix="properties.",
                            gdal_env=unsigned_s3_env, page=lambda c: "https://radiantearth.github.io/stac-browser/#/"
                            f"external/earth-search.aws.element84.com/v1/collections/{c}"),
    # ESA's Copernicus Data Space: public catalogue, S3 keys for the files, 4 connections per account
    "cdse": Archive("cdse", "https://stac.dataspace.copernicus.eu/v1", gdal_env=cdse_env, workers=4, retries=6,
                    search_gap=1.0, refused="CDSE refused the S3 keys (expired or wrong?); see docs/ARCHIVES.md",
                    page=lambda c: f"https://browser.stac.dataspace.copernicus.eu/collections/{c}"),
    # NASA Earthdata, LP DAAC's catalogue (HLS); public search, a token for the files
    "earthdata": Archive("earthdata", "https://cmr.earthdata.nasa.gov/stac/LPCLOUD", gdal_env=earthdata_env,
                         workers=8, refused="Earthdata refused the token (tokens expire after 60 days); generate "
                                            "a new one and run `python -m majortom.build.credentials earthdata`",
                         page=lambda c: "https://www.earthdata.nasa.gov/data/catalog/lpcloud-"
                         + c.lower().replace("_", "-")),
    # Google Earth Engine: a registered Cloud project; reads with computePixels, so no files or STAC
    "earth-engine": Archive("earth-engine", backend="earthengine", workers=8,
                            refused="Earth Engine refused the request (project not registered, or quota?)",
                            page=lambda c: "https://developers.google.com/earth-engine/datasets/catalog/"
                            + c.split("+")[-1].replace("/", "_")),
}
