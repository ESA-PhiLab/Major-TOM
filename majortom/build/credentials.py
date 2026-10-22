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
from pathlib import Path

CDSE_SETUP = ("CDSE needs S3 keys (not your login). Create them at "
              "https://eodata-s3keysmanager.dataspace.copernicus.eu, then run "
              "`python -m majortom.build.credentials cdse`, or set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY")
EARTHDATA_SETUP = ("NASA Earthdata needs a token (not your password). Generate one at "
                   "https://urs.earthdata.nasa.gov (Generate Token; it lasts 60 days), then run "
                   "`python -m majortom.build.credentials earthdata`, or set EARTHDATA_TOKEN")


# looking credentials up (backends call these when a session starts) -------------------------------------------

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


# setting them up once ----------------------------------------------------------------------------------------

SNOWBIRD = dict(bbox=(-111.7, 40.55, -111.6, 40.6), max_items=1)


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
    from .archives import ARCHIVES
    from .backends import backend

    archive = ARCHIVES[archive_name]
    method = backend(archive)
    item = method.search(archive, collection, **dict(SNOWBIRD, datetime="2024-08-01/2024-08-31"))[0]
    with method.session(archive):
        crs, transform = method.header(archive, item, asset)
    return f"{item.id}: {crs}, {abs(transform.a):g} m pixels"


def check_cdse() -> str:
    """Open one CDSE file header with the saved keys; returns what was read."""
    return check("cdse", "sentinel-2-l2a", "SCL_60m")


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
