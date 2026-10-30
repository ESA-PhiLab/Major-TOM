"""Core-Weather-style tables for any dataset's samples (docs/CLIMATE.md section 6).

    python -m majortom.climate stream --samples <metadata.parquet> [more ...] --source era5 --out <dir>
    python -m majortom.climate monthly --source era5-monthly --years 1940 2025 --out <dir>
    python -m majortom.climate climatology --out <dir> --period 1991 2020
    python -m majortom.climate merge --root <Core-Weather> --sources era5 cams cpc s5p --out <dir>
    python -m majortom.climate gauges --cpc <CPC-Monthly build> --raw <cache>/cpc/raw

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
    run.add_argument("--max-fields", type=int, help="stop after N fields (quota budget); rerun to continue")
    run.add_argument("--cache", type=Path, help="downloads and per-day means (default ~/.cache/majortom/climate)")
    month = sub.add_parser("monthly", help="[Source]-Monthly: monthly values for every cell")
    month.add_argument("--source", required=True, help="era5-monthly, era5-land-monthly, cpc")
    month.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    month.add_argument("--out", type=Path, required=True)
    month.add_argument("--workers", type=int, default=4)
    month.add_argument("--cache", type=Path)
    month.add_argument("--prefetch", type=int, default=0, help="years requested ahead (queued sources)")
    still = sub.add_parser("static", help="fixed fields (ERA5: terrain, soil, vegetation, lakes) for every cell")
    still.add_argument("--source", default="era5-all")
    still.add_argument("--out", type=Path, required=True)
    clim = sub.add_parser("climatology", help="per cell and calendar month: mean, std, years, from a monthly series")
    clim.add_argument("--out", type=Path, required=True, help="the folder written by `monthly`")
    clim.add_argument("--period", type=int, nargs=2, default=(1991, 2020), metavar=("FIRST", "LAST"))
    pack = sub.add_parser("repack", help="moved: python -m majortom.release tables repack (forwarded)")
    pack.add_argument("--src", type=Path, required=True, help="a monthly folder (cells, series, climatology) or static")
    pack.add_argument("--out", type=Path, required=True)
    pack.add_argument("--workers", type=int, default=None)
    join = sub.add_parser("merge", help="per-source Core-Weather folders -> one hourly and one daily table")
    join.add_argument("--root", type=Path, required=True, help="the folder holding one folder per source")
    join.add_argument("--sources", nargs="+", required=True, help="source folder names, e.g. era5 cams cpc s5p")
    join.add_argument("--out", type=Path, required=True)
    gauge = sub.add_parser("gauges", help="add cpc_gauges (rain gauges near each cell) to a CPC monthly build")
    gauge.add_argument("--cpc", type=Path, required=True, help="the CPC-Monthly build folder (cells.parquet, series/)")
    gauge.add_argument("--raw", type=Path, required=True, help="CPC's raw daily files, one folder per year")
    gauge.add_argument("--workers", type=int, default=None)
    get = sub.add_parser("fetch", help="download months of a queued source (CAMS) into the cache, several at a time")
    get.add_argument("--source", default="cams")
    get.add_argument("--months", nargs=2, required=True, metavar=("FIRST", "LAST"), help="YYYY-MM YYYY-MM")
    get.add_argument("--parallel", type=int, default=3, help="months in flight (the store still runs one at a time)")
    get.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        if args.cache:
            os.environ["MAJORTOM_CLIMATE_CACHE"] = str(args.cache)
        import time
        from concurrent.futures import ThreadPoolExecutor

        from .readers import queue
        src = SOURCES[args.source]
        surface, single = src.options["surface"][0], next(v for v in src.variables if v not in src.options["surface"])
        months = pd.date_range(f"{args.months[0]}-01", f"{args.months[1]}-01", freq="MS")
        start = time.monotonic()

        def one(month):
            for v in (single, surface):                       # both of the month's requests
                queue._cams(src, v, month)
            print(f"{month:%Y-%m} cached after {(time.monotonic() - start) / 60:.0f} min", flush=True)

        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(one, months))
        return 0
    if args.command == "gauges":
        from .gauges import add_gauges
        print(f"{add_gauges(args.cpc, args.raw, args.workers)} monthly files have cpc_gauges", flush=True)
        return 0
    if args.command == "merge":
        from .merge import merge
        print(f"{merge(args.root, args.sources, args.out)} day files written", flush=True)
        return 0
    if getattr(args, "cache", None):
        os.environ["MAJORTOM_CLIMATE_CACHE"] = str(args.cache)
    if args.command == "monthly":
        from .monthly import monthly
        print(f"{monthly(args.source, args.out, tuple(args.years), workers=args.workers, prefetch=args.prefetch)} "
              "months written", flush=True)
        return 0
    if args.command == "static":
        from .monthly import static_table
        print(f"{static_table(args.source, args.out):,} cells written", flush=True)
        return 0
    if args.command == "climatology":
        from .monthly import climatology
        print(f"{climatology(args.out, tuple(args.period)):,} rows written", flush=True)
        return 0
    if args.command == "repack":                              # publishing formats live in majortom.release
        from ..release.__main__ import main as release_main
        print("note: moved to `python -m majortom.release tables repack`", flush=True)
        release_main(["tables", "repack", "--src", str(args.src), "--out", str(args.out),
                      *(["--workers", str(args.workers)] if args.workers else [])])
        return 0
    samples = pd.concat([read_samples(p) for p in args.samples], ignore_index=True)
    print(f"{len(samples):,} samples from {samples['dataset'].nunique()} datasets", flush=True)
    write_links(samples, args.out / "acquisitions.parquet")
    if SOURCES[args.source].period == "day":                  # ERA5-Land, NOAA CPC, Sentinel-5P: daily table only
        reads = stream_days(samples, args.out, args.source, args.variables, args.daily, args.workers, args.progress,
                            args.max_fields)
    else:
        reads = stream(samples, args.out, args.source, args.variables, args.hourly, args.daily, args.workers,
                       args.progress, args.max_fields)
    print(f"done: {reads:,} fields read", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
