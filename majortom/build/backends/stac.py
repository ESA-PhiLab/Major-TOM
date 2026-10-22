"""STAC API + GDAL: Planetary Computer, Earth Search, CDSE, NASA Earthdata.

    search     STAC item search: one request per search (page size), retries, spacing for rate limits
    header     opens the asset's file header with GDAL
    read_tile  the file through a warped view (WarpedVRT) onto the window grid: nearest neighbour where the
               file is already in the window's CRS (an exact copy), else the collection's resampling
    session    rasterio.Env with the archive's GDAL settings: credentials exist only inside it

The GDAL settings per archive (public, unsigned S3, CDSE keys, Earthdata token) are defined here and named by
each Archive's gdal_env. They never reach os.environ, sample metadata or error messages.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator

import numpy as np
import pystac
import pystac_client
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.session import Session
from rasterio.vrt import WarpedVRT

from ..credentials import cdse_keys, earthdata_token
from . import retry

if TYPE_CHECKING:
    from ...spec import Window
    from ..archives import Archive

GDAL_OPTIONS = dict(                       # cloud-native reads: one request per header, HTTP/2, retries
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_INGESTED_BYTES_AT_OPEN="32768",
    GDAL_HTTP_VERSION="2", GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="1",
    # give up on stalled transfers, not slow ones: a limit on total time (GDAL_HTTP_TIMEOUT) cut 1.4 MB tiles
    # from a distant bucket short, and GDAL decoded the partial tile (TIFFFillTile "got ... bytes")
    GDAL_HTTP_CONNECTTIMEOUT="30", GDAL_HTTP_LOW_SPEED_TIME="60", GDAL_HTTP_LOW_SPEED_LIMIT="1024",
)
PAGE = 100                                              # items per page: one request per search (CDSE pages by 10)
_gates: dict[str, tuple[threading.Lock, list[float]]] = {}
_gates_lock = threading.Lock()


# GDAL settings per archive ------------------------------------------------------------------------------------

def public_env() -> dict:
    """GDAL settings for archives whose files need no credentials."""
    return dict(GDAL_OPTIONS)


def unsigned_s3_env() -> dict:
    """GDAL settings for public S3 buckets: s3:// links are read without signing (AWS Open Data).

    JPEG 2000 files (Sentinel-2 L1C) are decoded in one thread: GDAL's decoder threads do not see settings
    made by rasterio.Env in a worker thread, so their unsigned requests fail. Samples still run in parallel.
    """
    return dict(GDAL_OPTIONS, AWS_NO_SIGN_REQUEST="YES", GDAL_NUM_THREADS="1")


class S3Keys(Session):
    """S3 keys for rasterio.Env(session=...): rasterio refuses AWS keys as plain settings, and its own
    AWSSession needs boto3. GDAL gets the keys only while the Env block is open; repr hides them."""

    def __init__(self, key: str, secret: str):
        self._options = {"AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret, "AWS_SESSION_TOKEN": ""}

    @classmethod
    def hascreds(cls, config: dict) -> bool:
        return "AWS_ACCESS_KEY_ID" in config and "AWS_SECRET_ACCESS_KEY" in config

    def get_credential_options(self) -> dict:
        return dict(self._options)

    def __repr__(self) -> str:
        return "S3Keys(<hidden>)"


def cdse_env() -> dict:
    """GDAL settings for CDSE's S3 storage, keys passed as a session. A session token from the environment is
    blanked so it cannot mix in. JPEG 2000 decoding stays in one thread, as for Earth Search."""
    return dict(GDAL_OPTIONS, session=S3Keys(*cdse_keys()), AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu",
                AWS_VIRTUAL_HOSTING="FALSE", AWS_HTTPS="YES", AWS_NO_SIGN_REQUEST="NO", GDAL_NUM_THREADS="1")


def earthdata_env() -> dict:
    """GDAL settings for NASA Earthdata files: the token as bearer authentication, only while the Env is open.

    No HEAD request: Earthdata answers a HEAD with a link signed for HEAD only, and GDAL would reuse it for the
    data reads (they then fail as "not recognized as being in a supported file format").
    """
    return dict(GDAL_OPTIONS, GDAL_HTTP_AUTH="BEARER", GDAL_HTTP_BEARER=earthdata_token(),
                CPL_VSIL_CURL_USE_HEAD="NO")


# the four backend functions -----------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def catalog(archive: Archive) -> pystac_client.Client:
    """The archive's catalogue client, one per archive (Planetary Computer signs the asset links)."""
    client = pystac_client.Client.open(archive.stac_url, modifier=archive.modifier, timeout=30)
    if archive.sorts:
        client.add_conforms_to("SORT")                  # Planetary Computer sorts but does not declare it
    client._stac_io.session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))
    return client


def search(archive: Archive, collection: str, sortby: list[dict] | None = None, **query) -> list[pystac.Item]:
    """One catalogue search with retries, at most one per `archive.search_gap` seconds across threads."""
    if sortby:
        query["sortby"] = [dict(order, field=archive.sort_prefix + order["field"]) for order in sortby]

    def run() -> list[pystac.Item]:
        if archive.search_gap:
            with _gates_lock:
                gate = _gates.setdefault(archive.name, (threading.Lock(), [0.0]))
            with gate[0]:                                # one search at a time, spaced by the gap
                wait = gate[1][0] + archive.search_gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                gate[1][0] = time.monotonic()
        limit = min(PAGE, query.get("max_items") or PAGE)
        return list(catalog(archive).search(collections=[collection], limit=limit, **query).items())
    return retry(run, archive.retries)


def _split(key: str) -> tuple[str, int | None]:
    """An asset key, or "ASSET#n" for band n (1-based) of a multi-band file (ASTER's VNIR, SWIR, TIR)."""
    asset, _, band = key.partition("#")
    return asset, (int(band) if band else None)


def header(archive: Archive, item: pystac.Item, key: str) -> tuple[CRS, rasterio.Affine]:
    """CRS and geotransform of one asset, from its file header."""
    with rasterio.open(item.assets[_split(key)[0]].href) as src:
        return src.crs, src.transform


def read_tile(archive: Archive, item: pystac.Item, key: str, win: Window, res: float,
              method: str) -> tuple[np.ndarray, np.ndarray]:
    """One file onto the window grid: (bands, covered pixels). Pixels outside the file are 0."""
    n = win.pixels(res)
    asset, band = _split(key)
    with rasterio.open(item.assets[asset].href) as src:
        same_crs = src.crs == CRS.from_user_input(win.crs)
        resampling = Resampling.nearest if same_crs else Resampling[method]
        with WarpedVRT(src, crs=win.crs, transform=win.transform(res), width=n, height=n,
                       resampling=resampling, add_alpha=True,
                       init_dest_nodata=False, INIT_DEST=0) as vrt:   # empty pixels 0; coverage from alpha
            indexes = None if band is None else [band, vrt.count]   # one band and the alpha band
            tile = retry(lambda: vrt.read(indexes))      # bands, then alpha (0 = not covered)
    return tile[:-1], tile[-1] > 0


@contextmanager
def session(archive: Archive) -> Iterator[None]:
    """rasterio.Env with the archive's GDAL settings; credentials, if any, live only inside this block."""
    with rasterio.Env(**archive.gdal_env()):
        yield
