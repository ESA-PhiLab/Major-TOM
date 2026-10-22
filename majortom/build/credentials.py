"""Set up archive credentials once: `python -m majortom.build.credentials cdse`.

    prompt (hidden input)  ->  [cdse] section of the AWS credentials file (owner-only, 600)  ->  one test read

Keys are never printed. The environment (CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY) still takes precedence,
e.g. in Slurm jobs or Colab; see docs/ARCHIVES.md.
"""
from __future__ import annotations

import argparse
import configparser
import getpass
import os
import sys

from .archives import ARCHIVES, credentials_file

TEST_SEARCH = dict(collections=["sentinel-2-l2a"], bbox=(-111.7, 40.55, -111.6, 40.6), max_items=1,
                   datetime="2024-08-29T00:00:00Z/2024-08-29T23:59:59Z")   # Snowbird, one clear day


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
    parser.add_argument("archive", choices=["cdse"])
    parser.add_argument("--no-check", action="store_true", help="skip the test read")
    args = parser.parse_args(argv)
    print("CDSE S3 keys (create them at https://eodata-s3keysmanager.dataspace.copernicus.eu)")
    key = getpass.getpass("access key: ").strip()
    secret = getpass.getpass("secret key: ").strip()
    if not (key and secret):
        print("both keys are needed; nothing saved", file=sys.stderr)
        return 1
    save_cdse(key, secret)
    print(f"saved to the [cdse] section of {credentials_file()} (owner-only)")
    if not args.no_check:
        try:
            print("test read ok:", check_cdse())
        except Exception as error:                     # the message names the file, never the keys
            print(f"test read failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
