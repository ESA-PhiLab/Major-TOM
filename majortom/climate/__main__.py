"""Core-Weather-style tables for any dataset's samples (docs/CLIMATE.md section 6).

    python -m majortom.climate stream --samples <metadata.parquet> [more ...] --source era5 --out <dir>
    python -m majortom.climate monthly --source era5-monthly --years 1940 2025 --out <dir>
    python -m majortom.climate climatology --out <dir> --period 1991 2020
    python -m majortom.climate merge --root <Core-Weather> --sources era5 cams cpc s5p --out <dir>

    metadata files (paths or URLs; columns grid_cell, grid_row_u, grid_col_r, timestamp, and product_id if present)
      -> <out>/acquisitions.parquet: grid_cell, acquired, dataset, product_id   (links samples to table rows)
      -> hourly sources: stream(), <out>/hourly/ and <out>/daily/; daily sources: stream_days(), <out>/daily/
         one file per acquisition day; a rerun continues where it stopped
The dataset name is the metadata file's folder (Major-TOM/Core-S2L2A/.../metadata.parquet -> Core-S2L2A).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq

from .days import stream_days
from .sources import SOURCES
from .stream import stream

COLUMNS = ["grid_cell", "grid_row_u", "grid_col_r", "timestamp"]


def dataset_name(path: str) -> str:
    """Core-S2L2A from .../Major-TOM/Core-S2L2A/resolve/main/metadata.parquet; else the file's parent folder."""
    found = re.search(r"/datasets/[^/]+/([^/]+)/", path)
    return found.group(1) if found else Path(path).parent.name


def read_samples(path: str) -> pd.DataFrame:
    with fsspec.open(path).open() as f:
        table = pq.ParquetFile(f)
        names = COLUMNS + (["product_id"] if "product_id" in table.schema_arrow.names else [])
        samples = table.read(columns=names).to_pandas()
    return samples.assign(dataset=dataset_name(path))


def write_links(samples: pd.DataFrame, path: Path) -> None:
    acquired = pd.to_datetime(samples["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    links = samples.assign(acquired=acquired)[["grid_cell", "acquired", "dataset"] +
                                              (["product_id"] if "product_id" in samples else [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    links.sort_values(["acquired", "grid_cell", "dataset"]).to_parquet(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m majortom.climate", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("stream", help="hourly and daily tables around each acquisition")
    run.add_argument("--samples", nargs="+", required=True, help="metadata parquet files (paths or URLs)")
    run.add_argument("--source", default="era5")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--variables", nargs="*", help="short names; default: all of the source's")
    run.add_argument("--hourly", type=int, default=24, help="hours either side of each acquisition")
    run.add_argument("--daily", type=int, default=30, help="days either side of each acquisition")
    run.add_argument("--workers", type=int, default=32, help="read threads")
    run.add_argument("--progress", type=int, default=0, help="print progress every N fields")
    run.add_argument("--max-fields", type=int, help="daily sources: stop after N fields (quota budget); rerun to continue")
    run.add_argument("--cache", type=Path, help="downloads and per-day means (default ~/.cache/majortom/climate)")
    month = sub.add_parser("monthly", help="[Source]-Monthly: monthly values for every cell")
    month.add_argument("--source", required=True, help="era5-monthly, era5-land-monthly, cpc")
    month.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    month.add_argument("--out", type=Path, required=True)
    month.add_argument("--workers", type=int, default=4)
    month.add_argument("--cache", type=Path)
    clim = sub.add_parser("climatology", help="per cell and calendar month: mean, std, years, from a monthly series")
    clim.add_argument("--out", type=Path, required=True, help="the folder written by `monthly`")
    clim.add_argument("--period", type=int, nargs=2, default=(1991, 2020), metavar=("FIRST", "LAST"))
    join = sub.add_parser("merge", help="per-source Core-Weather folders -> one hourly and one daily table")
    join.add_argument("--root", type=Path, required=True, help="the folder holding one folder per source")
    join.add_argument("--sources", nargs="+", required=True, help="source folder names, e.g. era5 cams cpc s5p")
    join.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "merge":
        from .merge import merge
        print(f"{merge(args.root, args.sources, args.out)} day files written", flush=True)
        return 0
    if getattr(args, "cache", None):
        os.environ["MAJORTOM_CLIMATE_CACHE"] = str(args.cache)
    if args.command == "monthly":
        from .monthly import monthly
        print(f"{monthly(args.source, args.out, tuple(args.years), workers=args.workers)} months written", flush=True)
        return 0
    if args.command == "climatology":
        from .monthly import climatology
        print(f"{climatology(args.out, tuple(args.period)):,} rows written", flush=True)
        return 0
    samples = pd.concat([read_samples(p) for p in args.samples], ignore_index=True)
    print(f"{len(samples):,} samples from {samples['dataset'].nunique()} datasets", flush=True)
    write_links(samples, args.out / "acquisitions.parquet")
    if SOURCES[args.source].period == "day":                  # ERA5-Land, NOAA CPC, Sentinel-5P: daily table only
        reads = stream_days(samples, args.out, args.source, args.variables, args.daily, args.workers, args.progress,
                            args.max_fields)
    else:
        reads = stream(samples, args.out, args.source, args.variables, args.hourly, args.daily, args.workers,
                       args.progress)
    print(f"done: {reads:,} fields read", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
