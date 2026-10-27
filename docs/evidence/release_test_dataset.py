"""major-tom/release-test: one small TACO dataset per majortom.build source, made as a real release would be.

    build   (.venv311): every collection x 3 cells (Loch Lomond, the Dolomites near Corvara, the Amazon near Manaus)
              scene products: 2 dates per cell (clearest of May-June and of August-September 2023)
              yearly products: 2 years per cell (2020, 2021); static products: 1 sample per cell
              a collection with no data at any of the 3 cells falls back to St Louis (OPERA: North America)
              Earth Engine-only collections (and modis-09a1) from Earth Engine: run with EE_PROJECT=wiki-eo
              -> <root>/build/<collection>/<query>/<product>/*.npz
    release (release env): python -m majortom.release build per collection, as for Core releases
              (lossless rumi, GeoEnrich on, Hilbert order), partition limit small so each has >= 2 partitions
              -> <root>/release-test/<collection>/<collection>.zip (+ partitions, .tacocat)
    check   (release env): per dataset: taco.validate, every file read back exactly against its build sample,
              partitions, GeoEnrich columns -> <root>/release-test/check.csv
    publish (release env): each source folder -> source.coop://major-tom/release-test/<source>, then README.md,
              figures/ and check.csv at the root; a second pass must upload nothing (resume)

    python docs/evidence/release_test_dataset.py build   --root /data/Major-TOM/release-test
    python docs/evidence/release_test_dataset.py release --root /data/Major-TOM/release-test
    python docs/evidence/release_test_dataset.py check   --root /data/Major-TOM/release-test
    python docs/evidence/release_test_dataset.py publish --root /data/Major-TOM/release-test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CELLS = {"Loch Lomond": (-4.62, 56.08), "Dolomites (Corvara)": (11.87, 46.55), "Amazon (Manaus)": (-60.0, -3.1)}
FALLBACK = {"St Louis": (-90.2, 38.6)}
SCENES = {"2023-early": "2023-05-01/2023-06-30", "2023-late": "2023-08-01/2023-09-30"}
YEARS = {"2020": "2020-01-01/2020-12-31", "2021": "2021-01-01/2021-12-31"}
OVERRIDE = {"aster-l1t": {"2004-early": "2004-05-01/2004-06-30", "2004-late": "2004-08-01/2004-09-30"},   # PC: 2000-2006
            "opera-dswx-s1": {"2024-early": "2024-05-01/2024-06-30", "2024-late": "2024-08-01/2024-09-30"}}  # from 2023-12
EARTH_ENGINE = {"landsat-c2-l1", "dynamic-world", "viirs-dnb-monthly", "sentinel-1-grd-tc", "modis-09a1"}
ORDER = ["planetary-computer", "earthdata", "cdse", "earth-search"]


def _archive(name, collection) -> str:
    if name in EARTH_ENGINE:
        return "earth-engine"
    if collection.sources is None:
        return "planetary-computer"
    return next(a for a in ORDER if a in collection.sources)


def _queries(name, collection) -> dict[str, str | None]:
    if collection.time == "static":
        return {"static": None}
    if collection.time == "yearly":
        return YEARS
    return OVERRIDE.get(name, SCENES)


def build(root: Path, only: list[str] | None = None) -> None:
    from majortom.build import download
    from majortom.build.collections import COLLECTIONS
    from majortom.cells import _table, cell_at

    for name, collection in COLLECTIONS.items():
        if only and name not in only:
            continue
        archive = _archive(name, collection)
        done = 0
        for places in (CELLS, FALLBACK):
            cells = _table([cell_at(*xy) for xy in places.values()])
            for tag, when in _queries(name, collection).items():
                out = root / "build" / name / tag
                if any(out.rglob("*.npz")) and places is CELLS:
                    done += len(list(out.rglob("*.npz")))
                    continue
                try:
                    failed = download(cells, name, datetime=when, cloud=None, archive=archive, outdir=out, workers=3)
                    done += len(cells) - len(failed)
                except Exception as error:
                    print(f"{name} {tag}: ERROR {type(error).__name__}: {str(error)[:160]}", flush=True)
            if done:
                break
            print(f"{name}: nothing at {list(CELLS)}; trying {list(FALLBACK)}", flush=True)
        print(f"{name} ({archive}): {done} samples", flush=True)


def release(root: Path, only: list[str] | None = None) -> None:
    from majortom.build.collections import COLLECTIONS

    for name in sorted(p.name for p in (root / "build").iterdir() if any(p.rglob("*.npz")) and (not only or p.name in only)):
        samples = list((root / "build" / name).rglob("*.npz"))
        size = sum(p.stat().st_size for p in samples)
        seen, dupes = set(), []                     # yearly queries can return the same edition: keep one
        for npz in sorted(samples):
            side = json.loads(npz.with_suffix(".json").read_text())
            key = (side["grid_cell"], side.get("datetime"))
            if key in seen:
                dupes.append(npz)
            seen.add(key)
        if dupes:
            for npz in dupes:                       # moved aside, not deleted
                target = root / "duplicates" / name / npz.parent.name
                target.mkdir(parents=True, exist_ok=True)
                for f in (npz, npz.with_suffix(".json")):
                    f.rename(target / f"{npz.parent.parent.name}_{f.name}")
            print(f"{name}: {len(dupes)} duplicate samples (same cell and time) moved to {root / 'duplicates'}", flush=True)
        out = root / "release-test" / name / f"{name}.zip"
        args = [sys.executable, "-W", "ignore", "-m", "majortom.release", "build", "--samples", str(root / "build" / name),
                "--out", str(out), "--id", f"release-test-{name}", "--overwrite",
                "--title", f"Major TOM release test: {name}",
                "--description", f"Test release of {name} samples (3 cells x 2 dates) made with majortom.release; "
                                 "not for use, for testing the release framework",
                "--license", "CC-BY-4.0", "--provider", "Major TOM", "--keyword", "test",
                "--max-bytes", str(max(1, size // 3))]                     # ~3 partitions of the raw size, at least 2
        if COLLECTIONS[name].time == "static":
            args.append("--no-time")
        r = subprocess.run(args, capture_output=True, text=True)
        print(f"{name}: {(r.stdout.strip().splitlines() or [''])[-1] if r.returncode == 0 else 'FAILED ' + r.stderr[-300:]}",
              flush=True)


def check(root: Path) -> None:
    import rumi
    import taco

    from majortom.release import groups
    from majortom.release.taco import sample_id

    rows = []
    for folder in sorted((root / "release-test").iterdir()):
        if not folder.is_dir() or not any(folder.glob("*.zip")):          # skip figures/ and the like
            continue
        name = folder.name
        path = folder / ".tacocat" if (folder / ".tacocat").exists() else folder / f"{name}.zip"
        row = dict(collection=name)
        try:
            row["valid"] = bool(taco.validate(str(path)).ok)
            table = taco.open_dataset(str(path)).sql("SELECT * FROM dataset").to_pandas()
            row["samples"], row["partitions"] = len(table), table["source_file"].nunique() if "source_file" in table else 1
            row["geoenrich"] = sum(c.startswith("geoenrich:") for c in table.columns)
            exact = files = 0
            static = all(len(i.split("_")) == 3 for i in table["id"])      # MT10km_0624U_0029L: cell code alone
            for npz in (root / "build" / name).rglob("*.npz"):
                found, side = groups(npz)
                sid = sample_id(side["grid_cell"], side["grid_km"], None if static else side.get("datetime"))
                match = table[table["id"] == sid]
                if match.empty:
                    continue
                r = match.iloc[0]
                for g in found:
                    back = np.asarray(rumi.read(r[f"{g.name}.rumi::location"], r[f"{g.name}.rumi::header"]))
                    exact += bool(np.array_equal(back.reshape(g.data.shape), g.data, equal_nan=True))
                    files += 1
            row["files_exact"] = f"{exact}/{files}"
            row["status"] = "ok" if row["valid"] and exact == files and files else "CHECK"
        except Exception as error:
            row.update(status="FAIL", why=f"{type(error).__name__}: {error}"[:200])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(root / "release-test" / "check.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


def publish_all(root: Path, destination: str = "source.coop://major-tom/release-test") -> None:
    from majortom.release.publish import publish

    out = root / "release-test"
    for rerun in (False, True):                      # the second pass checks resuming: nothing should be uploaded
        sent = 0
        for folder in sorted(f for f in out.iterdir() if f.is_dir() and any(f.glob("*.zip"))):
            r = publish(folder, f"{destination}/{folder.name}")
            sent += len(r.uploaded)
            if not rerun:
                print(r, flush=True)
            assert r.verified, f"{folder.name}: remote does not match"
        r = publish(out, destination, include=["README.md", "figures/*.png", "check.csv"])
        sent += len(r.uploaded)
        print(f"{'second' if rerun else 'first'} pass: {sent} files uploaded; root {r}", flush=True)
    assert sent == 0, "the second pass uploaded files: resuming does not work"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("phase", choices=["build", "release", "check", "publish"])
    p.add_argument("--root", required=True)
    p.add_argument("--only", nargs="*", help="build only these collections")
    a = p.parse_args()
    {"build": lambda: build(Path(a.root), a.only), "release": lambda: release(Path(a.root), a.only),
     "check": lambda: check(Path(a.root)), "publish": lambda: publish_all(Path(a.root))}[a.phase]()
