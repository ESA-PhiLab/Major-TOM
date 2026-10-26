"""Payload benchmark for Major TOM releases: COG against rumi on real Sentinel-2 v2 windows (docs/RELEASE.md 10).

    Core-S2L2A metadata -> random samples (seeded, no missing pixels)
      -> each band cut to the v2 window with crop_core_to_v2 (samples where it does not fit are skipped)
      -> groups by pixel size: 10 m (B02 B03 B04 B08), 20 m (B05 B06 B07 B8A B11 B12), 60 m (B01 B09)
      -> each group written in every FORMAT -> <out>/<format>/<sample>_<group>.(tif|rumi)
      -> per format: bytes, write s, full read s, random 256 px crops of the 10 m group per s
      -> <out>/results.csv and a summary table printed

    .venv (Python 3.12, rasterio 1.5 with GDAL 3.12, rumi-eo, geozl) at /data/Major-TOM/release-bench:
    python docs/evidence/release_bench.py --n 5 --out /data/Major-TOM/release-bench/tiny          (Slurm CPU job)

Timings are single-threaded and warm (files just written, in the page cache), as in repeated training epochs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import fsspec
import geozl
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import rumi
from fsspec.parquet import open_parquet_file
from rasterio.io import MemoryFile
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import Cell                      # noqa: E402
from majortom.spec import crop_core_to_v2           # noqa: E402

META = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"
GROUPS = {"10m": ["B02", "B03", "B04", "B08"], "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"], "60m": ["B01", "B09"]}
FORMATS = {                                          # name -> (kind, settings)
    "cog-tile": ("cog", {"INTERLEAVE": "TILE"}),
    "cog-pixel": ("cog", {"INTERLEAVE": "PIXEL"}),
    "rumi-176": ("rumi", {"tile": 176, "error": None}),
    "rumi-264": ("rumi", {"tile": 264, "error": None}),
    "rumi-176-e1": ("rumi", {"tile": 176, "error": 1}),
}
PATTERN = "b (row h) (col w) -> row col (b h w)"     # all bands of a tile in one frame, as COG tile interleave
CROPS, CROP = 50, 256
rumi.set_num_threads(1)


def samples(n: int, seed: int):
    """Yield (name, cell, acquired, {group: (array, transform, crs)}) for n samples whose v2 window fits."""
    with fsspec.open(META).open() as f:
        meta = pq.read_table(f, columns=["grid_cell", "grid_row_u", "grid_col_r", "timestamp", "parquet_url",
                                         "parquet_row", "nodata"]).to_pandas()
    meta = meta[meta.nodata == 0].sample(frac=1, random_state=seed)
    found = skipped = 0
    for r in meta.itertuples():
        if found == n:
            break
        cell = Cell(int(r.grid_row_u), int(r.grid_col_r), 10.0)
        bands = [b for g in GROUPS.values() for b in g]
        with open_parquet_file(r.parquet_url, columns=bands, row_groups=[int(r.parquet_row)]) as f:
            table = pq.ParquetFile(f).read_row_group(int(r.parquet_row), columns=bands)
        groups = {}
        for g, names in GROUPS.items():
            arrays = []
            for b in names:
                with MemoryFile(table.column(b)[0].as_py()) as m, m.open() as src:
                    out = crop_core_to_v2(src.read(1), src.transform, src.crs.to_string(), cell, "Core-S2L2A")
                    crs = src.crs.to_string()
                if out is None:
                    break
                arrays.append(out[0])
                transform = out[1]
            if out is None:
                break
            groups[g] = (np.stack(arrays), transform, crs)
        if len(groups) < len(GROUPS):
            skipped += 1
            continue
        found += 1
        print(f"sample {found}/{n}: {r.grid_cell} lat {cell.centroid[1]:.1f} ({skipped} skipped so far)", flush=True)
        yield r.grid_cell, cell, pd.Timestamp(r.timestamp), groups


def write_cog(path: Path, data: np.ndarray, transform, crs, settings: dict) -> None:
    with rasterio.open(path, "w", driver="COG", width=data.shape[2], height=data.shape[1], count=data.shape[0],
                       dtype=data.dtype, crs=crs, transform=transform, BLOCKSIZE=176, COMPRESS="ZSTD", PREDICTOR=2,
                       OVERVIEWS="NONE", **settings) as f:
        f.write(data)


_graphs: dict = {}


def write_rumi(path: Path, data: np.ndarray, transform, crs, settings: dict, group: str, when) -> bytes:
    """Frames of `tile` px; the GeoZL method is profiled once per (group, tile, error) on the first frame."""
    frames = rumi.frames(data, PATTERN, tile_size=settings["tile"])
    key = (group, settings["tile"], settings["error"])
    if key not in _graphs:
        first = list(frames)[0].data
        _graphs[key] = geozl.profile(first, error=settings["error"], reps=1)[0]["graph"]
        print(f"   geozl method for {key}: {_graphs[key]}", flush=True)
    built: dict = {}
    for frame in frames:
        shape = frame.data.shape
        if shape not in built:
            built[shape] = geozl.graph(frame.data, _graphs[key], error=settings["error"])
        frame.compressed = geozl.compress(frame.data, graph=built[shape])
    _, header = rumi.write(path, frames, bands=GROUPS[group], time=[when.strftime("%Y-%m-%d")],
                           transform=tuple(transform)[:6], crs=crs)
    return header


def main(n: int, out: str, seed: int) -> None:
    out = Path(out)
    rows = []
    rng = np.random.default_rng(seed)
    for name, cell, when, groups in samples(n, seed):
        for fmt, (kind, settings) in FORMATS.items():
            folder = out / fmt
            folder.mkdir(parents=True, exist_ok=True)
            for group, (data, transform, crs) in groups.items():
                path = folder / f"{name}_{group}.{'tif' if kind == 'cog' else 'rumi'}"
                t = time.perf_counter()
                if kind == "cog":
                    write_cog(path, data, transform, crs, settings)
                    header = None
                else:
                    header = write_rumi(path, data, transform, crs, settings, group, when)
                write_s = time.perf_counter() - t
                t = time.perf_counter()
                if kind == "cog":
                    with rasterio.open(path) as f:
                        back = f.read()
                        layout = f.tags(ns="IMAGE_STRUCTURE").get("INTERLEAVE")
                else:
                    back = np.asarray(rumi.read(path, header)).reshape(data.shape)
                    layout = "frames"
                read_s = time.perf_counter() - t
                exact = bool(np.array_equal(back, data))
                maxerr = int(np.abs(back.astype(np.int32) - data.astype(np.int32)).max())
                crop_s = np.nan
                if group == "10m":
                    size = data.shape[1]
                    yx = rng.integers(0, size - CROP, (CROPS, 2))
                    t = time.perf_counter()
                    if kind == "cog":
                        with rasterio.open(path) as f:
                            for y, x in yx:
                                f.read(window=Window(int(x), int(y), CROP, CROP))
                    else:
                        for y, x in yx:
                            rumi.read(path, header, window=(int(y), int(x), CROP, CROP))
                    crop_s = (time.perf_counter() - t) / CROPS
                rows.append(dict(sample=name, lat=cell.centroid[1], format=fmt, group=group, layout=layout,
                                 raw=data.nbytes, bytes=path.stat().st_size, write_s=write_s, read_s=read_s,
                                 crop_s=crop_s, exact=exact, maxerr=maxerr))
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summary = df.groupby("format").agg(layout=("layout", "first"), ratio=("raw", "sum"), mb=("bytes", "sum"),
                                       write_s=("write_s", "sum"), read_s=("read_s", "sum"), crop_ms=("crop_s", "mean"),
                                       exact=("exact", "all"), maxerr=("maxerr", "max"))
    summary["ratio"] = summary["ratio"] / summary["mb"]
    summary["mb"] = summary["mb"] / 1e6
    summary["crop_ms"] *= 1000
    print(f"\n{df['sample'].nunique()} samples, latitudes {sorted(df.groupby('sample').lat.first().round(1).tolist())}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.n, a.out, a.seed)
