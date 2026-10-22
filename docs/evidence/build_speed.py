"""Reading speed of every product from every archive, published as a dated report in docs/reports/.

    (product, archive) pairs  ->  N cells near Snowbird each (dates from tests/test_products.py)
      -> wall time per pair (samples/hour at the archive's workers), failures with their reasons
      -> each sample's timing: search, scene choice, seconds per band (pixel size from the array), retries
      -> docs/reports/<date>-build-speed.md (one table per kind) + <date>-build-speed.csv (one row per band)

The report states where and when it ran, anonymised: country only (from Cloudflare's public trace endpoint,
or --region; --region none to skip), UTC run window, CPUs, software versions, round-trip to each archive.
No IP address, host name, user name or path is written.

Run as a Slurm job (about an hour):
    sbatch --cpus-per-task=16 --mem=32G -J build-speed ~/slurm-templates/cpu.sbatch \
        .venv/bin/python docs/evidence/build_speed.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_products import DATES, has_cdse_keys                   # noqa: E402

from majortom import build                                       # noqa: E402
from majortom.build.archives import ARCHIVES                     # noqa: E402
from majortom.build.collections import COLLECTIONS               # noqa: E402

AREA = (-111.9, 40.45, -111.45, 40.8)                            # Wasatch, around Snowbird
REPORTS = ROOT / "docs" / "reports"
STORAGE = {                                                      # one file host per archive, for round-trips
    "planetary-computer": "https://sentinel2l2a01.blob.core.windows.net",
    "earth-search": "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
    "cdse": "https://eodata.dataspace.copernicus.eu",
}


def country() -> str:
    """Country code of the network this runs from, via Cloudflare's trace; only `loc` is kept."""
    try:
        trace = requests.get("https://www.cloudflare.com/cdn-cgi/trace", timeout=10).text
        return dict(line.split("=", 1) for line in trace.splitlines() if "=" in line).get("loc", "unknown")
    except requests.RequestException:
        return "unknown"


def round_trip(url: str, n: int = 5) -> float | None:
    """Median seconds for an HTTPS HEAD request (connection and TLS included), or None if unreachable."""
    times = []
    for _ in range(n):
        try:
            times.append(requests.head(url, timeout=10).elapsed.total_seconds())
        except requests.RequestException:
            pass
    return round(statistics.median(times), 3) if times else None


def software() -> dict:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return dict(majortom=commit.stdout.strip() + (" (with local changes)" if dirty else ""),
                rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__, python=platform.python_version())


def run_pair(name: str, archive: str, cells: pd.DataFrame, outdir: Path) -> tuple[list[dict], pd.DataFrame]:
    """Build the cells for one pair: one row per band and sample, and the failed samples."""
    start = time.perf_counter()
    errors = build.download(cells, name, DATES.get(name), archive=archive, cloud=None, outdir=outdir, force=True)
    wall = time.perf_counter() - start
    rows = []
    for sidecar in sorted((outdir / name).glob("*.json")):
        meta, arrays = json.loads(sidecar.read_text()), np.load(sidecar.with_suffix(".npz"))
        timing = meta["timing"]
        for band, seconds in timing["bands"].items():
            plane = next(arrays[k] for k in arrays.files if k == band or k.startswith(f"{band}_"))
            rows.append(dict(kind=COLLECTIONS[name].kind, product=name, archive=archive, band=band,
                             res_m=round(meta["window_m"] / plane.shape[-1]), seconds=seconds,
                             search=timing["search"], scene_choice=timing["scene_choice"], total=timing["total"],
                             retries=timing.get("retries", 0), sample=meta["index"], wall=wall,
                             built=len(cells) - len(errors), failed=len(errors)))
    return rows, errors.assign(product=name, archive=archive)


def kind_tables(rows: pd.DataFrame) -> list[str]:
    """One Markdown table per kind: throughput, per-sample steps, and seconds per band by pixel size."""
    out = []
    for kind, group in rows.groupby("kind", sort=False):
        out += [f"### {kind.capitalize()}", "",
                "| Product | Archive | Samples/hour | s/sample | Search | Scene choice | Retries | "
                "s per band, by pixel size |", "|---|---|---|---|---|---|---|---|"]
        for (product, archive), pair in group.groupby(["product", "archive"], sort=False):
            per_sample = pair.drop_duplicates("sample")
            bands = " · ".join(f"{res} m: {s:.2f}" for res, s in pair.groupby("res_m")["seconds"].mean().items())
            built = int(per_sample["built"].iloc[0])
            out.append(f"| {product} | {archive} | {3600 * built / pair['wall'].iloc[0]:.0f} | "
                       f"{per_sample['total'].mean():.1f} | {per_sample['search'].mean():.1f} | "
                       f"{per_sample['scene_choice'].mean():.1f} | {int(per_sample['retries'].sum())} | {bands} |")
        out.append("")
    return out


def report(rows: pd.DataFrame, errors: pd.DataFrame, context: dict, n_cells: int, csv_name: str) -> str:
    net = ["| Archive | Workers | Catalogue round-trip (s) | Storage round-trip (s) |", "|---|---|---|---|"]
    net += [f"| {a} | {ARCHIVES[a].workers} | {v['catalogue']} | {v['storage']} |"
            for a, v in context["network"].items()]
    if errors.empty:
        failures = ["None."]
    else:
        failures = ["| Product | Archive | Failed | First reason |", "|---|---|---|---|"]
        for (product, archive), group in errors.groupby(["product", "archive"], sort=False):
            failures.append(f"| {product} | {archive} | {len(group)} | {group['error'].iloc[0][:160]} |")
    return "\n".join([
        "> Automated benchmark report, generated by `docs/evidence/build_speed.py`. Review status: unreviewed.",
        "",
        f"# {context['date']} — majortom.build reading speed",
        "",
        f"Every product from every archive, {n_cells} cells each near Snowbird (Wasatch, Utah). "
        f"Raw rows: [{csv_name}]({csv_name}).",
        "",
        "## Where and when",
        "",
        f"- Country of the network: {context['country']} (no address or host recorded)",
        f"- Run window: {context['start']} to {context['end']} UTC",
        f"- CPUs available: {context['cpus']}",
        "",
        "## Software",
        "",
        *[f"- {k}: {v}" for k, v in context["software"].items()],
        "",
        "## Network",
        "",
        "Median of 5 HTTPS HEAD requests, connection and TLS included. Distance to each archive is part of every "
        "time below.",
        "",
        *net,
        "",
        "## Results per kind",
        "",
        "Samples/hour: built samples over the pair's wall time, at the archive's workers. s/sample: mean time per "
        "sample (search, scene choice, all band reads). Search and scene choice: mean seconds per sample. "
        "Retries: waits after failed requests, summed over the pair's samples.",
        "",
        *kind_tables(rows),
        "## Failures",
        "",
        *failures,
        "",
        "## Method",
        "",
        "- One sample is one cell's window: search, scene choice (cloud mask reads), every band read, written to "
        ".npz + .json.",
        "- Dates per product: as in `tests/test_products.py`; cloud=None, so any scene is accepted.",
        "- Each pair runs alone, one after another; archives are not loaded at the same time.",
        "- Search times depend on each archive's load at run time; compare runs, not single numbers.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="majortom.build reading-speed report")
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--region", default=None, help="country to report instead of the lookup; 'none' to skip")
    args = parser.parse_args()
    start = dt.datetime.now(dt.timezone.utc)
    context = dict(date=start.date().isoformat(), start=start.strftime("%Y-%m-%d %H:%M"),
                   country=("not stated" if args.region == "none" else args.region or country()),
                   cpus=len(os.sched_getaffinity(0)), software=software(), network={})
    archives = [a for a in ARCHIVES if a != "cdse" or has_cdse_keys()]
    context["network"] = {a: dict(catalogue=round_trip(ARCHIVES[a].stac_url), storage=round_trip(STORAGE[a]))
                          for a in archives}
    cells = build.cells_in(AREA, d=10).reset_index(drop=True).head(args.cells)
    rows, errors = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name, collection in COLLECTIONS.items():
            for archive in collection.archives():
                if archive not in archives:
                    print(f"skip {name}@{archive}: no credentials", flush=True)
                    continue
                pair_rows, pair_errors = run_pair(name, archive, cells, Path(tmp) / archive)
                print(f"{name}@{archive}: {len(pair_rows)} band reads, {len(pair_errors)} failed", flush=True)
                rows += pair_rows
                errors.append(pair_errors)
    context["end"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    stem = f"{context['date']}-build-speed"
    table, failed = pd.DataFrame(rows), pd.concat(errors, ignore_index=True)
    table.to_csv(REPORTS / f"{stem}.csv", index=False)
    (REPORTS / f"{stem}.md").write_text(report(table, failed, context, args.cells, f"{stem}.csv"))
    print(f"wrote {REPORTS / stem}.md")


if __name__ == "__main__":
    main()
