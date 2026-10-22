"""STAC API + GDAL: Planetary Computer, Earth Search, CDSE, NASA Earthdata.

    search     STAC item search: one request per search (page size), retries, spacing for rate limits
    header     opens the asset's file header with GDAL
    read_tile  the file through a warped view (WarpedVRT) onto the window grid: nearest neighbour where the
               file is already in the window's CRS (an exact copy), else the collection's resampling; files
               with ground control points instead of a grid (raw Sentinel-1 GRD) are placed by those
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
from rasterio.warp import reproject

from ..collections import find_asset
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
    data reads (they then fail as "not recognized as being in a supported file format"). The token is kept on
    redirects: ASF sends reads on to a second host (cumulus.asf.alaska.edu) that needs it too.
    """
    return dict(GDAL_OPTIONS, GDAL_HTTP_AUTH="BEARER", GDAL_HTTP_BEARER=earthdata_token(),
                CPL_VSIL_CURL_USE_HEAD="NO", CPL_VSIL_CURL_AUTHORIZATION_HEADER_ALLOWED_IF_REDIRECT="YES")


# the four backend functions -----------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def catalog(archive: Archive, provider: str | None = None) -> pystac_client.Client:
    """The archive's catalogue client, one per archive and provider (Planetary Computer signs the asset links)."""
    url = f"{archive.stac_url}/{provider}" if provider else archive.stac_url
    client = pystac_client.Client.open(url, modifier=archive.modifier, timeout=30)
    if archive.sorts:
        client.add_conforms_to("SORT")                  # Planetary Computer sorts but does not declare it
    client._stac_io.session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))
    return client


def search(archive: Archive, collection: str, sortby: list[dict] | None = None, **query) -> list[pystac.Item]:
    """One catalogue search with retries, at most one per `archive.search_gap` seconds across threads."""
    if sortby:
        query["sortby"] = [dict(order, field=archive.sort_prefix + order["field"]) for order in sortby]
    provider, collection = collection.split("/", 1) if archive.providers else (None, collection)

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
        return list(catalog(archive, provider).search(collections=[collection], limit=limit, **query).items())
    return retry(run, archive.retries)


def _split(key: str) -> tuple[str, int | None, str | None]:
    """An asset key; "ASSET#n" for band n (1-based) of a multi-band file (ASTER's VNIR, SWIR, TIR);
    "ASSET:VARIABLE" for a variable of a netCDF swath placed by latitude and longitude (Sentinel-3)."""
    key, _, variable = key.partition(":")
    asset, _, band = key.partition("#")
    return asset, (int(band) if band else None), (variable or None)


def header(archive: Archive, item: pystac.Item, key: str) -> tuple[CRS, rasterio.Affine]:
    """CRS and geotransform of one asset, from its file header (a swath: latitude/longitude, no grid)."""
    asset, _, variable = _split(key)
    if variable:
        return CRS.from_epsg(4326), rasterio.Affine.identity()
    with rasterio.open(item.assets[find_asset(item, asset)].href) as src:
        return src.crs, src.transform


def read_tile(archive: Archive, item: pystac.Item, key: str, win: Window, res: float,
              method: str) -> tuple[np.ndarray, np.ndarray]:
    """One file onto the window grid: (bands, covered pixels). Pixels outside the file are 0."""
    n = win.pixels(res)
    asset, band, variable = _split(key)
    if variable:
        return _read_swath(item, asset, variable, win, res, n, Resampling[method])
    with rasterio.open(item.assets[find_asset(item, asset)].href) as src:
        if src.crs is None and src.gcps[0]:
            return _read_by_gcps(src, band, win, res, n, Resampling[method])
        same_crs = src.crs == CRS.from_user_input(win.crs)
        resampling = Resampling.nearest if same_crs else Resampling[method]
        with WarpedVRT(src, crs=win.crs, transform=win.transform(res), width=n, height=n,
                       resampling=resampling, add_alpha=True,
                       init_dest_nodata=False, INIT_DEST=0) as vrt:   # empty pixels 0; coverage from alpha
            indexes = None if band is None else [band, vrt.count]   # one band and the alpha band
            tile = retry(lambda: vrt.read(indexes))      # bands, then alpha (0 = not covered)
    return tile[:-1], tile[-1] > 0


def _read_by_gcps(src, band: int | None, win: Window, res: float, n: int,
                  resampling: Resampling) -> tuple[np.ndarray, np.ndarray]:
    """A file placed by ground control points (raw Sentinel-1 GRD) onto the window grid: no terrain correction.

    GDAL reads only the part of the file the window needs. 0 is "no data" in these files, so it marks coverage.
    """
    gcps, gcps_crs = src.gcps
    bands = [band] if band else list(range(1, src.count + 1))
    out = np.zeros((len(bands), n, n), src.dtypes[0])
    for i, b in enumerate(bands):
        retry(lambda: reproject(rasterio.band(src, b), out[i], gcps=gcps, src_crs=gcps_crs, src_nodata=0,
                                dst_transform=win.transform(res), dst_crs=win.crs, dst_nodata=0,
                                resampling=resampling))
    return out, (out != 0).any(axis=0)


def _gdal_path(href: str, variable: str) -> str:
    """GDAL's name for one variable of a remote netCDF-4 (HDF5) file."""
    path = "/vsis3/" + href[5:] if href.startswith("s3://") else f"/vsicurl/{href}"
    return f'HDF5:"{path}"://{variable}'


@lru_cache(maxsize=32)                                 # the bands of one sample share the scene's coordinates
def _read_scaled(href: str, variable: str, block: tuple[int, int, int, int] | None = None
                 ) -> tuple[np.ndarray, float | None]:
    """A variable, or a block of it (col, row, width, height), with its CF scale and offset applied (fill: NaN)."""
    from rasterio.windows import Window as Block

    with rasterio.open(_gdal_path(href, variable)) as src:
        raw = retry(lambda: src.read(1, window=Block(*block) if block else None))
        fill = src.nodata
        values = raw.astype("float32") * src.scales[0] + src.offsets[0]
        if fill is not None:
            values[raw == fill] = np.nan
    return values, fill


def _read_swath(item: pystac.Item, asset: str, variable: str, win: Window, res: float, n: int,
                resampling: Resampling) -> tuple[np.ndarray, np.ndarray]:
    """A Sentinel-3 swath variable onto the window grid, placed by its latitude and longitude arrays.

    The tie-point coordinates (every 64th column) locate the block of rows and columns around the window; only
    that block of the variable and of the full-resolution coordinates is read. Values are physical (CF scale and
    offset applied); pixels without data, or outside the swath, are NaN and not covered.
    """
    from rasterio.warp import transform_bounds

    west, south, east, north = transform_bounds(win.crs, "EPSG:4326", *win.bounds, densify_pts=21)
    pad = 0.05
    tie = item.assets[find_asset(item, "tie-geo-coordinates")].href
    tie_lat, _ = _read_scaled(tie, "latitude")
    tie_lon, _ = _read_scaled(tie, "longitude")
    rows, cols = np.nonzero((tie_lat > south - pad) & (tie_lat < north + pad) &
                            (tie_lon > west - pad) & (tie_lon < east + pad))
    out = np.full((1, n, n), np.nan, "float32")
    if not rows.size:
        return out, np.zeros((n, n), bool)
    geo, data = item.assets[find_asset(item, "geo-coordinates")].href, item.assets[find_asset(item, asset)].href
    with rasterio.open(_gdal_path(geo, "latitude")) as src:
        height, width = src.height, src.width
    step = width // (tie_lat.shape[1] - 1) if tie_lat.shape[1] > 1 else 1
    r0, r1 = max(rows.min() - 2, 0), min(rows.max() + 3, height)
    c0, c1 = max((cols.min() - 1) * step, 0), min((cols.max() + 2) * step, width)
    block = (int(c0), int(r0), int(c1 - c0), int(r1 - r0))
    lat, _ = _read_scaled(geo, "latitude", block)
    lon, _ = _read_scaled(geo, "longitude", block)
    values, _ = _read_scaled(data, variable, block)
    located = np.stack([lon, lat]).astype("float64")    # GDAL needs float64 coordinates, longitude first
    retry(lambda: reproject(values, out[0], src_geoloc_array=located, src_crs=CRS.from_epsg(4326),
                            src_nodata=np.nan, dst_transform=win.transform(res), dst_crs=CRS.from_user_input(win.crs),
                            dst_nodata=np.nan, resampling=resampling))
    return out, np.isfinite(out[0])


@contextmanager
def session(archive: Archive) -> Iterator[None]:
    """rasterio.Env with the archive's GDAL settings; credentials, if any, live only inside this block."""
    with rasterio.Env(**archive.gdal_env()):
        yield
