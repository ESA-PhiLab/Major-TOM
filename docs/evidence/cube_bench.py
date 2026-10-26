"""Time series in rumi: one file per acquisition, or one cube per cell? (docs/RELEASE.md, open question)

    build (.venv311, majortom.build): CELLS x 12 months of 2023, the clearest Sentinel-2 L2A scene per month
      (cloud <= 20%), Planetary Computer, v2 window -> <out>/build/2023-MM/sentinel-2-l2a/*.npz (+ .json)
    bench (release-bench env): per cell, the 10 m group as (T, 4, 1056, 1056), lossless
      A  one .rumi per acquisition, frames "b (row h) (col w) -> row col (b h w)"     (the current plan)
      B  one cube, "t b (row h) (col w) -> row col t b (h w)"                         (one band, one date per frame)
      C  one cube, "t b (row h) (col w) -> row col (t b h w)"                         (a tile's dates and bands)
      D  one cube, "t b (row h) (col w) -> row col (b t h w)"                         (same, band-major)
      -> bytes, write s, read one date, read the whole series at a random 256 px crop -> <out>/cube.csv

    python docs/evidence/cube_bench.py build --out /data/Major-TOM/release-bench/cube     (.venv311, Slurm)
    python docs/evidence/cube_bench.py bench --out /data/Major-TOM/release-bench/cube     (release-bench env)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CELLS = {"Paris farmland": (2.0, 48.6), "Sahara": (13.0, 24.0), "Amazon": (-60.0, -3.0), "Siberian taiga": (100.0, 60.0),
         "Ganges plain": (82.0, 26.0), "US Midwest": (-93.0, 42.0), "Alps": (8.0, 46.5), "Australian outback": (134.0, -24.0)}
MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]
BANDS = ["B02", "B03", "B04", "B08"]
LAYOUTS = {"B": "t b (row h) (col w) -> row col t b (h w)", "C": "t b (row h) (col w) -> row col (t b h w)",
           "D": "t b (row h) (col w) -> row col (b t h w)"}
CROP, CROPS = 256, 20


def build(out: Path) -> None:
    from majortom.build import download
    from majortom.cells import _table, cell_at

    cells = _table([cell_at(lon, lat) for lon, lat in CELLS.values()])
    for month in MONTHS:
        failed = download(cells, "sentinel-2-l2a", datetime=month, cloud=(0.0, 0.2), prefer="clearest",
                          outdir=out / "build" / month, workers=8)
        print(f"{month}: {len(cells) - len(failed)} of {len(cells)} built", flush=True)


def series(out: Path) -> dict[str, tuple[np.ndarray, list[str], dict]]:
    """grid_cell -> (T x 4 x 1056 x 1056 uint16, dates, sidecar of the first date) over the months built."""
    found: dict[str, list] = {}
    for npz in sorted((out / "build").glob("2023-*/sentinel-2-l2a/*.npz")):
        side = json.loads(npz.with_suffix(".json").read_text())
        arrays = np.load(npz)
        found.setdefault(side["grid_cell"], []).append((side, np.stack([arrays[b] for b in BANDS])))
    result = {}
    for cell, items in found.items():
        items.sort(key=lambda it: it[0]["datetime"])
        shapes = {a.shape for _, a in items}
        if len(items) < 6 or len(shapes) != 1:
            print(f"{cell}: skipped ({len(items)} dates, shapes {shapes})", flush=True)
            continue
        result[cell] = (np.stack([a for _, a in items]), [s["datetime"][:19] for s, _ in items], items[0][0])
    return result


def _graph(frames: list[np.ndarray]) -> str:
    """Best lossless graph over a few frames, all predictor families (as majortom.release.profiles.choose_graph)."""
    import geozl

    total: dict[str, int] = {}
    for f in frames:
        for r in geozl.profile(f, prior=None, reps=1):
            total[r["graph"]] = total.get(r["graph"], 0) + r["bytes"]
    return min(total, key=total.get)


def _write_cube(path: Path, cube: np.ndarray, pattern: str, dates: list[str], transform, crs) -> tuple[bytes, str]:
    import geozl
    import rumi

    frames = rumi.frames(cube, pattern, tile_size=176)
    listed = list(frames)
    graph = _graph([f.data for f in listed[:: max(1, len(listed) // 4)][:4]])
    built: dict = {}
    for f in frames:
        if f.data.shape not in built:
            built[f.data.shape] = geozl.graph(f.data, graph)
        f.compressed = geozl.compress(f.data, graph=built[f.data.shape])
    _, header = rumi.write(str(path), frames, bands=BANDS, time=dates, transform=tuple(transform)[:6], crs=crs)
    return header, graph


def bench(out: Path) -> None:
    import rumi
    from affine import Affine

    from majortom.release.rumi import write_rumi

    rumi.set_num_threads(1)
    rng = np.random.default_rng(0)
    rows = []
    for cell, (cube, dates, side) in series(out).items():
        t_n = cube.shape[0]
        transform = Affine(side["res"], 0, side["left"], 0, -side["res"], side["top"])   # 10 m: the finest
        crs = side["crs"]
        crops = rng.integers(0, cube.shape[-1] - CROP, (CROPS, 2))
        folder = out / "rumi" / cell
        folder.mkdir(parents=True, exist_ok=True)
        # A: one file per acquisition, one graph per cell (chosen on the first date)
        t = time.perf_counter()
        first = write_rumi(folder / "A_00.rumi", cube[0], transform, crs, BANDS, dates[0])
        files = [first] + [write_rumi(folder / f"A_{k:02d}.rumi", cube[k], transform, crs, BANDS, dates[k],
                                      graph=first.graph) for k in range(1, t_n)]
        write_s = time.perf_counter() - t
        t = time.perf_counter()
        rumi.read(str(files[t_n // 2].path), files[t_n // 2].header)
        one_s = time.perf_counter() - t
        t = time.perf_counter()
        for y, x in crops:
            np.stack([rumi.read(str(f.path), f.header, window=(int(y), int(x), CROP, CROP)) for f in files])
        rows.append(dict(cell=cell, dates=t_n, layout="A per acquisition", graph=first.graph,
                         bytes=sum(f.path.stat().st_size for f in files), write_s=write_s, one_date_s=one_s,
                         series_crop_ms=(time.perf_counter() - t) / CROPS * 1000, raw=cube.nbytes))
        # B, C, D: one cube per cell
        for name, pattern in LAYOUTS.items():
            path = folder / f"{name}.rumi"
            t = time.perf_counter()
            header, graph = _write_cube(path, cube, pattern, dates, transform, crs)
            write_s = time.perf_counter() - t
            back = np.asarray(rumi.read(str(path), header)).reshape(cube.shape)
            assert np.array_equal(back, cube), f"{cell} {name}: not lossless"
            t = time.perf_counter()
            rumi.read(str(path), header, time=[t_n // 2])
            one_s = time.perf_counter() - t
            t = time.perf_counter()
            for y, x in crops:
                rumi.read(str(path), header, window=(int(y), int(x), CROP, CROP))
            rows.append(dict(cell=cell, dates=t_n, layout=f"{name} cube {pattern.split('-> ')[1]}", graph=graph,
                             bytes=path.stat().st_size, write_s=write_s, one_date_s=one_s,
                             series_crop_ms=(time.perf_counter() - t) / CROPS * 1000, raw=cube.nbytes))
        print(f"{cell}: {t_n} dates done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cube.csv", index=False)
    s = df.groupby("layout").agg(cells=("cell", "nunique"), raw=("raw", "sum"), mb=("bytes", "sum"), write_s=("write_s", "sum"),
                                 one_date_ms=("one_date_s", "mean"), series_crop_ms=("series_crop_ms", "mean"))
    s["ratio"] = s.raw / s.mb
    s["mb"] /= 1e6
    s["one_date_ms"] *= 1000
    print(s.drop(columns="raw").round(2).to_string())
    print(df.groupby("layout").graph.agg(lambda g: g.value_counts().index[0]).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("phase", choices=["build", "bench"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    (build if a.phase == "build" else bench)(Path(a.out))
