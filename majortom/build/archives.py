"""Archives the builder searches and reads from.

    Archive  ->  catalog(archive)       STAC client: address, link signing, sorting
             ->  archive.gdal_env()     GDAL settings for reading its files, credentials included

An Archive holds no credentials: gdal_env() fetches them when a read starts, and the reads use them only
inside a rasterio.Env block, so they never reach os.environ, sample metadata or error messages.

CDSE keys are looked up in this order: CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY, then the [cdse] section of
the AWS credentials file (~/.aws/credentials, or AWS_SHARED_CREDENTIALS_FILE). The NASA Earthdata token:
EARTHDATA_TOKEN, then ~/.config/majortom/earthdata-token. Set either up once with
`python -m majortom.build.credentials cdse` (or `earthdata`).
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import planetary_computer
from rasterio.session import Session

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


CDSE_SETUP = ("CDSE needs S3 keys (not your login). Create them at "
              "https://eodata-s3keysmanager.dataspace.copernicus.eu, then run "
              "`python -m majortom.build.credentials cdse`, or set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY")


def credentials_file() -> Path:
    """The AWS-style credentials file: AWS_SHARED_CREDENTIALS_FILE, else ~/.aws/credentials."""
    return Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials")).expanduser()


def profile_keys(section: str) -> tuple[str | None, str | None]:
    """(access key, secret key) from one section of the credentials file, or (None, None)."""
    config = configparser.ConfigParser()
    config.read(credentials_file())
    if not config.has_section(section):
        return None, None
    return config[section].get("aws_access_key_id"), config[section].get("aws_secret_access_key")


def cdse_keys() -> tuple[str, str]:
    """CDSE S3 keys: from the environment, else the [cdse] profile. Never logged or stored."""
    key, secret = os.environ.get("CDSE_S3_ACCESS_KEY"), os.environ.get("CDSE_S3_SECRET_KEY")
    if not (key and secret):
        key, secret = profile_keys("cdse")
    if not (key and secret):
        raise PermissionError(CDSE_SETUP)
    return key, secret


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


EARTHDATA_SETUP = ("NASA Earthdata needs a token (not your password). Generate one at "
                   "https://urs.earthdata.nasa.gov (Generate Token; it lasts 60 days), then run "
                   "`python -m majortom.build.credentials earthdata`, or set EARTHDATA_TOKEN")


def earthdata_token_file() -> Path:
    return Path(os.environ.get("MAJORTOM_CONFIG", "~/.config/majortom")).expanduser() / "earthdata-token"


def earthdata_token() -> str:
    """The Earthdata token: from the environment, else the owner-only token file. Never logged or stored."""
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token and earthdata_token_file().exists():
        token = earthdata_token_file().read_text().strip()
    if not token:
        raise PermissionError(EARTHDATA_SETUP)
    return token


def earthdata_env() -> dict:
    """GDAL settings for NASA Earthdata files: the token as bearer authentication, only while the Env is open.

    No HEAD request: Earthdata answers a HEAD with a link signed for HEAD only, and GDAL would reuse it for the
    data reads (they then fail as "not recognized as being in a supported file format").
    """
    return dict(GDAL_OPTIONS, GDAL_HTTP_AUTH="BEARER", GDAL_HTTP_BEARER=earthdata_token(),
                CPL_VSIL_CURL_USE_HEAD="NO")


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
    retries     attempts per catalogue search (waits 1, 2, 4, ... s; for rate limits)
    search_gap  least seconds between two catalogue searches, across threads (rate limits)
    refused     added to a sample's error when reading is refused (HTTP 403), e.g. how to fix the keys
    """
    name: str
    stac_url: str
    modifier: Callable | None = None
    sorts: bool = True
    sort_prefix: str = ""
    gdal_env: Callable[[], dict] = public_env
    workers: int = 16
    retries: int = 4
    search_gap: float = 0.0
    refused: str = ""


ARCHIVES = {
    "planetary-computer": Archive("planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1",
                                  modifier=planetary_computer.sign_inplace),
    # Element 84's catalogue of AWS Open Data buckets; some links are s3://, some https://
    "earth-search": Archive("earth-search", "https://earth-search.aws.element84.com/v1", sort_prefix="properties.",
                            gdal_env=unsigned_s3_env),
    # ESA's Copernicus Data Space: public catalogue, S3 keys for the files, 4 connections per account
    "cdse": Archive("cdse", "https://stac.dataspace.copernicus.eu/v1", gdal_env=cdse_env, workers=4, retries=6,
                    search_gap=1.0, refused="CDSE refused the S3 keys (expired or wrong?); see docs/ARCHIVES.md"),
    # NASA Earthdata, LP DAAC's catalogue (HLS); public search, a token for the files
    "earthdata": Archive("earthdata", "https://cmr.earthdata.nasa.gov/stac/LPCLOUD", gdal_env=earthdata_env,
                         workers=8, refused="Earthdata refused the token (tokens expire after 60 days); generate "
                                            "a new one and run `python -m majortom.build.credentials earthdata`"),
}
