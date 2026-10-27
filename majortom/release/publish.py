"""Uploading a release folder: source.coop by default, Hugging Face as an option (docs/RELEASE.md section 8).

    release folder (<name>_part*.zip or <name>.zip, .tacocat/)
      -> local files: size and S3 ETag (MD5, or MD5 of 64 MB parts for multipart uploads; cached in
         <folder>/.publish-sums.json with the sha256)
      -> remote objects under the destination: size and ETag, from the listing (source.coop's proxy keeps no
         custom metadata, but returns ETags)
      -> plan: upload what is missing or different; partitions first (largest first), the TACOCAT last
      -> upload (multipart for large files) -> verify every file against the remote listing

Destinations:
    source.coop://<account>/<repository>[/<prefix>]   S3, two credential routes (docs.source.coop):
        A. data proxy, endpoint https://data.source.coop: bucket <account>, keys <repository>/<prefix>/...
           with a repository API key (saved once with `python -m majortom.build.credentials source-coop`), or the
           temporary credentials the repository page gives for the proxy (SOURCE_COOP_ACCESS_KEY_ID,
           SOURCE_COOP_SECRET_ACCESS_KEY, SOURCE_COOP_SESSION_TOKEN in the environment; never saved)
        B. temporary credentials from the repository page ("View Credentials": AWS_ACCESS_KEY_ID,
           AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN in the environment): bucket <region>.opendata.source.coop
           (the product's data connection; SOURCE_COOP_REGION, default eu-central-1), keys
           <account>/<repository>/<prefix>/...
    hf://<owner>/<dataset>[/<path>]                     Hugging Face dataset repository (huggingface_hub login)
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_COOP_ENDPOINT = "https://data.source.coop"
REGION = "eu-central-1"          # source.coop data connection of Major TOM releases (Frankfurt; Miko, 2026-10-27)


def _bucket(region: str) -> str:
    """Route B's bucket for an AWS Open Data connection, e.g. eu-central-1.opendata.source.coop."""
    return f"{region}.opendata.source.coop"
CHUNK = 64 * 1024**2


@dataclass
class PublishReport:
    destination: str
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    verified: bool = False

    def __str__(self) -> str:
        return (f"{self.destination}: {len(self.uploaded)} uploaded, {len(self.skipped)} already there, "
                f"verified {self.verified}")


def _files(folder: Path, include: list[str] | None = None) -> list[Path]:
    """Release files: partitions (largest first), then the TACOCAT; nothing else (no stage, no sums cache).

    `include`: glob patterns instead (e.g. README.md and figures/*.png at a repository's root), in that order.
    """
    if include is not None:
        found = [p for pattern in include for p in sorted(folder.glob(pattern)) if p.is_file()]
        if not found:
            raise FileNotFoundError(f"nothing matches {include} in {folder}")
        return found
    parts = sorted((p for p in folder.glob("*.zip") if p.is_file()), key=lambda p: -p.stat().st_size)
    catalogue = sorted(p for p in (folder / ".tacocat").rglob("*") if p.is_file()) if (folder / ".tacocat").exists() else []
    if not parts:
        raise FileNotFoundError(f"no release files (*.zip) in {folder}")
    return parts + catalogue


def _etag(path: Path) -> str:
    """The ETag S3 gives an upload of `path` with our settings: MD5, or for multipart uploads (larger than CHUNK)
    the MD5 of the parts' MD5s, a dash and the number of parts."""
    parts = []
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            parts.append(hashlib.md5(block).digest())
    if len(parts) <= 1:
        return (parts[0] if parts else hashlib.md5(b"").digest()).hex()
    return f"{hashlib.md5(b''.join(parts)).hexdigest()}-{len(parts)}"


def _sums(folder: Path, files: list[Path]) -> dict[str, tuple[int, str]]:
    """relative path -> (size, ETag); cached by size and modification time (with the sha256, for the record)."""
    cache_path = folder / ".publish-sums.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out = {}
    for f in files:
        rel = f.relative_to(folder).as_posix()
        stat = f.stat()
        hit = cache.get(rel)
        if hit and hit["size"] == stat.st_size and hit["mtime"] == stat.st_mtime and "etag" in hit:
            out[rel] = (stat.st_size, hit["etag"])
            continue
        digest = hashlib.sha256()
        with open(f, "rb") as handle:
            while block := handle.read(CHUNK):
                digest.update(block)
        out[rel] = (stat.st_size, _etag(f))
        cache[rel] = dict(size=stat.st_size, mtime=stat.st_mtime, etag=out[rel][1], sha256=digest.hexdigest())
    cache_path.write_text(json.dumps(cache))
    return out


def _source_coop(destination: str):
    """(S3 client, bucket, key prefix) for a source.coop destination, by credential route (module docstring)."""
    import boto3

    from ..build.credentials import source_coop_keys

    account, _, rest = destination.removeprefix("source.coop://").partition("/")
    if not account or not rest:
        raise ValueError(f"expected source.coop://<account>/<repository>[/<prefix>], got {destination!r}")
    keys = source_coop_keys()
    if keys is not None:                                   # route A: repository API key, data proxy
        key, secret, endpoint = keys
        client = boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret, endpoint_url=endpoint,
                              aws_session_token=os.environ.get("SOURCE_COOP_SESSION_TOKEN"),
                              region_name=os.environ.get("SOURCE_COOP_REGION", REGION))
        return client, account, rest.strip("/")
    if os.environ.get("AWS_SESSION_TOKEN"):                # route B: temporary credentials from the repository page
        region = os.environ.get("SOURCE_COOP_REGION", REGION)
        return boto3.client("s3", region_name=region), _bucket(region), f"{account}/{rest.strip('/')}"
    raise PermissionError("no source.coop credentials: save a repository API key with "
                          "`python -m majortom.build.credentials source-coop`, or paste the temporary credentials "
                          "from the repository page (View Credentials) as AWS_* environment variables")


def _remote(client, bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
    """relative key -> (size, ETag without quotes) for every object under the prefix (one listing, no HEADs)."""
    found = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            found[obj["Key"][len(prefix) + 1:]] = (obj["Size"], obj.get("ETag", "").strip('"'))
    return found


def publish(folder: str | Path, destination: str, *, dry_run: bool = False, workers: int = 4,
            include: list[str] | None = None, force: bool = False, client=None) -> PublishReport:
    """Upload a release folder to `destination` (module docstring); returns what was uploaded and skipped.

    Resumable: files already there with the same size and sha256 are skipped. `dry_run` only plans.
    `include`: glob patterns of other files to upload instead of a release (README.md, figures/*.png).
    `force`: upload even files already there (e.g. to set a content type). Each file gets the content type of its
    extension (image/png, text/markdown, ...), so browsers show README images.
    `client` (S3) is for tests; normally it is made from the saved credentials.
    """
    folder = Path(folder)
    files = _files(folder, include)
    sums = _sums(folder, files)
    report = PublishReport(destination)
    if destination.startswith("hf://"):
        return _hugging_face(folder, destination, files, report, dry_run)
    if client is None:
        client, bucket, prefix = _source_coop(destination)
    else:
        bucket, _, prefix = destination.removeprefix("s3://").partition("/")
    from boto3.s3.transfer import TransferConfig

    remote = {rel: v for rel, v in _remote(client, bucket, prefix).items()
              if any(f.relative_to(folder).as_posix() == rel for f in files)}
    config = TransferConfig(multipart_threshold=CHUNK, multipart_chunksize=CHUNK, max_concurrency=workers)
    for f in files:
        rel = f.relative_to(folder).as_posix()
        size, sha = sums[rel]
        if remote.get(rel) == (size, sha) and not force:
            report.skipped.append(rel)
            continue
        if not dry_run:
            kind = mimetypes.guess_type(f.name)[0] or ("text/markdown" if f.suffix == ".md" else "application/octet-stream")
            client.upload_file(str(f), bucket, f"{prefix}/{rel}", Config=config, ExtraArgs={"ContentType": kind})
        report.uploaded.append(rel)
    if not dry_run:
        remote = _remote(client, bucket, prefix)
        report.verified = all(remote.get(rel) == sums[rel] for rel in sums)
    return report


def _hugging_face(folder: Path, destination: str, files: list[Path], report: PublishReport, dry_run: bool):
    """Hugging Face: upload_folder of the release files (it skips files whose content is already there)."""
    from huggingface_hub import HfApi

    parts = destination.removeprefix("hf://").split("/")
    repo_id, path = "/".join(parts[:2]), "/".join(parts[2:]) or None
    rels = [f.relative_to(folder).as_posix() for f in files]
    if not dry_run:
        HfApi().upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder), path_in_repo=path,
                              allow_patterns=rels)
        report.verified = True
    report.uploaded = rels
    return report
