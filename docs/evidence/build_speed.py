"""Reading speed of every product from every archive, for the setup page (docs/ARCHIVES.md).

    (product, archive) pairs  ->  8 cells near Snowbird each (dates from tests/test_products.py)
      -> wall time per pair (samples/hour at the archive's workers)
      -> each sample's timing: search, scene choice, seconds per band (pixel size from the array)
      -> build_speed.csv (one row per band) + build_speed.md (one table per kind)

Run as a Slurm job (about an hour):
    sbatch --cpus-per-task=16 --mem=32G -J build-speed ~/slurm-templates/cpu.sbatch \
        .venv/bin/python docs/evidence/build_speed.py
Times are measured from wherever it runs; distance to each archive is part of them.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_products import DATES, has_cdse_keys                   # noqa: E402

from majortom import build                                       # noqa: E402
from majortom.build.collections import COLLECTIONS               # noqa: E402

AREA = (-111.9, 40.45, -111.45, 40.8)                            # Wasatch, around Snowbird
N_CELLS = 8
OUT = Path(__file__).parent


def run_pair(name: str, archive: str, cells: pd.DataFrame, outdir: Path) -> list[dict]:
    """Build the cells for one pair; one row per band and sample, with the pair's wall time."""
    start = time.perf_counter()
    errors = build.download(cells, name, DATES.get(name), archive=archive, cloud=None, outdir=outdir, force=True)
    wall = time.perf_counter() - start
    rows = []
    for sidecar in sorted((outdir / name).glob("*.json")):
        meta, arrays = json.loads(sidecar.read_text()), np.load(sidecar.with_suffix(".npz"))
        for band, seconds in meta["timing"]["bands"].items():
            plane = next(arrays[k] for k in arrays.files if k == band or k.startswith(f"{band}_"))
            rows.append(dict(kind=COLLECTIONS[name].kind, product=name, archive=archive, band=band,
                             res_m=round(meta["window_m"] / plane.shape[-1]), seconds=seconds,
                             search=meta["timing"]["search"], scene_choice=meta["timing"]["scene_choice"],
                             total=meta["timing"]["total"], sample=meta["index"], wall=wall,
                             built=len(cells) - len(errors), failed=len(errors)))
    return rows


def tables(rows: pd.DataFrame) -> str:
    """One Markdown table per kind: throughput, per-sample steps, and seconds per band by pixel size."""
    out = []
    for kind, group in rows.groupby("kind", sort=False):
        out += [f"### {kind.capitalize()}", "",
                "| Product | Archive | Samples/hour | s/sample | Search | Scene choice | s per band, by pixel size |",
                "|---|---|---|---|---|---|---|"]
        for (product, archive), pair in group.groupby(["product", "archive"], sort=False):
            per_sample = pair.drop_duplicates("sample")
            per_res = pair.groupby("res_m")["seconds"].mean()
            bands = " · ".join(f"{res} m: {s:.2f}" for res, s in per_res.items())
            built = int(per_sample["built"].iloc[0])
            out.append(f"| {product} | {archive} | {3600 * built / pair['wall'].iloc[0]:.0f} | "
                       f"{per_sample['total'].mean():.1f} | {per_sample['search'].mean():.1f} | "
                       f"{per_sample['scene_choice'].mean():.1f} | {bands} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    cells = build.cells_in(AREA, d=10).reset_index(drop=True).head(N_CELLS)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, collection in COLLECTIONS.items():
            for archive in collection.archives():
                if archive == "cdse" and not has_cdse_keys():
                    print(f"skip {name}@{archive}: no CDSE keys", flush=True)
                    continue
                pair_rows = run_pair(name, archive, cells, Path(tmp) / archive)
                print(f"{name}@{archive}: {len(pair_rows)} band reads", flush=True)
                rows += pair_rows
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "build_speed.csv", index=False)
    (OUT / "build_speed.md").write_text(tables(table))
    print(tables(table))


if __name__ == "__main__":
    main()
