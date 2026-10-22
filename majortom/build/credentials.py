"""Set up archive credentials once: `python -m majortom.build.credentials cdse` (or `earthdata`).

    cdse       prompt (hidden)  ->  [cdse] section of the AWS credentials file (owner-only, 600)  ->  one test read
    earthdata  prompt (hidden)  ->  ~/.config/majortom/earthdata-token (owner-only, 600)          ->  one test read

Nothing is printed back. The environment (CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY, EARTHDATA_TOKEN) still takes
precedence, e.g. in Slurm jobs or Colab. On shared machines file permissions keep out other users, not
administrators: use these revocable keys and tokens, never a main password. See docs/ARCHIVES.md.
"""
from __future__ import annotations

import argparse
import configparser
import getpass
import os
import sys

from .archives import ARCHIVES, credentials_file, earthdata_token_file

SNOWBIRD = dict(bbox=(-111.7, 40.55, -111.6, 40.6), max_items=1, datetime="2024-08-29T00:00:00Z/2024-08-29T23:59:59Z")
TEST_SEARCH = dict(collections=["sentinel-2-l2a"], **SNOWBIRD)


def save_cdse(key: str, secret: str) -> None:
    """Write the [cdse] section, keeping the file's other sections, and make the file owner-only."""
    path = credentials_file()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config.read(path)
    config["cdse"] = {"aws_access_key_id": key, "aws_secret_access_key": secret}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        config.write(handle)
    os.chmod(path, 0o600)                                # also if the file existed with wider permissions


def save_earthdata(token: str) -> None:
    """Write the token to its owner-only file, in an owner-only folder."""
    path = earthdata_token_file()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    os.chmod(path, 0o600)


def check(archive_name: str, collection: str, asset: str) -> str:
    """Open one file header of `collection` with the saved credentials; returns what was read."""
    import rasterio

    from .search import search_items

    archive = ARCHIVES[archive_name]
    item = search_items(archive, collections=[collection], **dict(SNOWBIRD, datetime="2024-08-01/2024-08-31"))[0]
    with rasterio.Env(**archive.gdal_env()), rasterio.open(item.assets[asset].href) as src:
        return f"{item.id}: {src.width} x {src.height} px"


def check_cdse() -> str:
    """Open one CDSE file header with the saved keys; returns what was read."""
    import rasterio

    from .search import catalog

    archive = ARCHIVES["cdse"]
    item = next(catalog(archive).search(**TEST_SEARCH).items())
    with rasterio.Env(**archive.gdal_env()), rasterio.open(item.assets["SCL_60m"].href) as src:
        return f"{item.id}: {src.width} x {src.height} px"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", choices=["cdse", "earthdata"])
    parser.add_argument("--no-check", action="store_true", help="skip the test read")
    args = parser.parse_args(argv)
    if args.archive == "cdse":
        print("CDSE S3 keys (create them at https://eodata-s3keysmanager.dataspace.copernicus.eu)")
        key = getpass.getpass("access key: ").strip()
        secret = getpass.getpass("secret key: ").strip()
        if not (key and secret):
            print("both keys are needed; nothing saved", file=sys.stderr)
            return 1
        save_cdse(key, secret)
        print(f"saved to the [cdse] section of {credentials_file()} (owner-only)")
        test = check_cdse
    else:
        print("NASA Earthdata token (https://urs.earthdata.nasa.gov > Generate Token; it lasts 60 days)")
        token = getpass.getpass("token: ").strip()
        if not token:
            print("no token given; nothing saved", file=sys.stderr)
            return 1
        save_earthdata(token)
        print(f"saved to {earthdata_token_file()} (owner-only)")
        test = lambda: check("earthdata", "HLSS30_2.0", "B04")    # noqa: E731
    if not args.no_check:
        try:
            print("test read ok:", test())
        except Exception as error:                     # the message names the file, never the keys
            print(f"test read failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
