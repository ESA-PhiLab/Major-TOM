"""Archives the builder searches and reads from.

    Archive  ->  catalog(archive)       STAC client: address, link signing, sorting
             ->  archive.gdal_env()     GDAL settings for reading its files, credentials included

An Archive holds no credentials: gdal_env() fetches them when a read starts, and the reads use them only
inside a rasterio.Env block, so they never reach os.environ, sample metadata or error messages.

CDSE keys are looked up in this order: CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY, then the [cdse] section of
the AWS credentials file (~/.aws/credentials, or AWS_SHARED_CREDENTIALS_FILE). Set them up once with
`python -m majortom.build.credentials cdse`.
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
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


def cdse_env() -> dict:
    """GDAL settings for CDSE's S3 storage. Other AWS settings are blanked so none mix in (e.g. a session token).

    JPEG 2000 decoding stays in one thread, as for Earth Search: GDAL's decoder threads would not see the keys.
    """
    key, secret = cdse_keys()
    return dict(GDAL_OPTIONS, AWS_ACCESS_KEY_ID=key, AWS_SECRET_ACCESS_KEY=secret, AWS_SESSION_TOKEN="",
                AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu", AWS_VIRTUAL_HOSTING="FALSE", AWS_HTTPS="YES",
                AWS_NO_SIGN_REQUEST="NO", GDAL_NUM_THREADS="1")


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
    refused: str = ""


ARCHIVES = {
    "planetary-computer": Archive("planetary-computer", "https://planetarycomputer.microsoft.com/api/stac/v1",
                                  modifier=planetary_computer.sign_inplace),
    # Element 84's catalogue of AWS Open Data buckets; some links are s3://, some https://
    "earth-search": Archive("earth-search", "https://earth-search.aws.element84.com/v1", sort_prefix="properties.",
                            gdal_env=unsigned_s3_env),
    # ESA's Copernicus Data Space: public catalogue, S3 keys for the files, 4 connections per account
    "cdse": Archive("cdse", "https://stac.dataspace.copernicus.eu/v1", gdal_env=cdse_env, workers=4, retries=6,
                    refused="CDSE refused the S3 keys (expired or wrong?); see docs/ARCHIVES.md"),
}
