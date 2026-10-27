"""Does the release path work for every collection majortom.build supports? (docs/RELEASE.md)

    build (.venv311): one sample per collection, one cell (St Louis: land, rivers, inside OPERA's North America),
      summer 2023, the first archive that serves it without Earth Engine -> <out>/<collection>/<product>/*.npz
    check (release-bench env): per sample
      bands: pixel size and dtype of every band (from the npz, before any grouping)
      -> groups -> write_rumi (lossless) -> read back exact -> read(as_="cog") exact, same CRS and transform
      -> partition over all samples -> <out>/matrix.csv: one row per collection and group, pass / fail / why

    python docs/evidence/release_matrix.py build --out /data/Major-TOM/release-bench/matrix
    python docs/evidence/release_matrix.py check --out /data/Major-TOM/release-bench/matrix
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CELL = (-90.2, 38.6)                         # St Louis, Missouri
WHEN = "2023-06-01/2023-09-30"
WHEN_FOR = {"aster-l1t": "2005-06-01/2005-09-30",            # Planetary Computer holds 2000-2006
            "opera-dswx-s1": "2024-06-01/2024-09-30",        # from 2023-12
            "modis-09a1": "2023-01-01/2023-12-31"}           # none found in summer 2023
ORDER = ["planetary-computer", "earthdata", "cdse", "earth-search"]   # Earth Engine: quota until 1 November


def _archive(collection) -> str | None:
    if collection.sources is None:
        return "planetary-computer"
    return next((a for a in ORDER if a in collection.sources), None)


EARTH_ENGINE = ["landsat-c2-l1", "dynamic-world", "viirs-dnb-monthly", "sentinel-1-grd-tc", "modis-09a1"]


def build(out: Path, earth_engine: bool = False) -> None:
    """earth_engine: build only EARTH_ENGINE from Earth Engine (small tests; EE_PROJECT and key from the environment)."""
    from majortom.build import download
    from majortom.build.collections import COLLECTIONS
    from majortom.cells import _table, cell_at

    cells = _table([cell_at(*CELL)])
    for name, collection in COLLECTIONS.items():
        if earth_engine and name not in EARTH_ENGINE:
            continue
        archive = "earth-engine" if earth_engine else _archive(collection)
        if archive is None:
            print(f"{name}: Earth Engine only, pending", flush=True)
            continue
        if any((out / name).rglob("*.npz")):
            print(f"{name}: already built", flush=True)
            continue
        try:
            failed = download(cells, name, datetime=WHEN_FOR.get(name, WHEN), cloud=None, archive=archive, outdir=out / name, workers=1)
            print(f"{name} ({archive}): {'built' if failed.empty else failed.to_dict('records')}", flush=True)
        except Exception as error:
            print(f"{name} ({archive}): ERROR {type(error).__name__}: {str(error)[:200]}", flush=True)


def check(out: Path) -> None:
    import rasterio
    import rumi

    from majortom.release import groups, partition, read, write_rumi

    rows, parts = [], []
    for npz in sorted(out.glob("*/*/*.npz")):
        collection = npz.parts[-3]
        side = json.loads(npz.with_suffix(".json").read_text())
        arrays = np.load(npz)
        layout = {}
        for band in side["bands"]:
            key = (round(side["window_m"] / arrays[band].shape[-1], 3), str(arrays[band].dtype))
            layout.setdefault(key, []).append(band)
        by_res = {}
        for (res, dtype), bands in layout.items():
            by_res.setdefault(res, set()).add(dtype)
        mixed = {r: sorted(d) for r, d in by_res.items() if len(d) > 1}
        base = dict(collection=collection, archive=side["archive"], datetime=side.get("datetime"),
                    window_m=side["window_m"], mixed_dtypes=str(mixed) if mixed else "")
        try:
            found, _ = groups(npz)
        except Exception as error:
            rows.append(dict(base, group="-", status="FAIL groups", why=f"{type(error).__name__}: {error}"[:200]))
            continue
        total = 0
        for g in found:
            row = dict(base, group=g.name, bands=" ".join(g.bands), dtype=str(g.data.dtype), shape=str(g.data.shape))
            try:
                path = out / "_rumi" / collection / f"{g.name}.rumi"
                w = write_rumi(path, g.data, g.transform, g.crs, g.bands, side.get("datetime") or "2000-01-01",
                               categorical=g.categorical)
                if g.categorical:                               # lossy profiles must be refused
                    try:
                        write_rumi(path.with_suffix(".lossy"), g.data, g.transform, g.crs, g.bands,
                                   side.get("datetime") or "2000-01-01", "near-lossless-1", categorical=True)
                        raise AssertionError("lossy profile accepted for categorical data")
                    except ValueError:
                        pass
                back = np.asarray(rumi.read(str(path), w.header)).reshape(g.data.shape)
                assert np.array_equal(back, g.data, equal_nan=True), "rumi read-back differs"
                with read(path, w.header, as_="cog", nodata=g.nodata) as mem, mem.open() as f:
                    assert np.array_equal(f.read(), g.data, equal_nan=True), "COG pixels differ"
                    assert f.transform == g.transform and f.crs == rasterio.crs.CRS.from_user_input(g.crs), "COG georef"
                total += path.stat().st_size
                row.update(status="ok", categorical=g.categorical, graph=w.graph, ratio=round(g.data.nbytes / path.stat().st_size, 2), why="")
            except Exception as error:
                row.update(status="FAIL", why=f"{type(error).__name__}: {error}"[:200])
                traceback.print_exc()
            rows.append(row)
        parts.append(dict(grid_cell=side["grid_cell"], acquired=pd.Timestamp(side.get("datetime") or "2000-01-01"),
                          bytes=total))
    df = pd.DataFrame(rows)
    df.to_csv(out / "matrix.csv", index=False)
    ids, summary = partition(pd.DataFrame(parts), max_bytes=10**8)
    pd.set_option("display.width", 250)
    print(df[["collection", "group", "dtype", "shape", "status", "categorical", "ratio", "why"]].to_string(index=False))
    print(f"\npartition: {len(parts)} samples -> {len(summary)} partitions")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("phase", choices=["build", "check"])
    p.add_argument("--out", required=True)
    p.add_argument("--earth-engine", action="store_true", help="build the Earth Engine-only collections")
    a = p.parse_args()
    build(Path(a.out), a.earth_engine) if a.phase == "build" else check(Path(a.out))
