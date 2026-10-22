"""Which source pixels does each Core sample hold? A set of samples per Core dataset, matched to its exact source.

    Core metadata (Hugging Face)  ->  ~10 samples per dataset, spread over processing versions, hemispheres, zone edges
      -> the Core band and its stored geotransform (one row-group read from the Core parquet)
      -> the source product:  S2 L1C/L2A  CDSE by product id (Core was built from the same EODATA store), else
                                          Planetary Computer for L2A originals (same name without the version)
                              S1 RTC      Planetary Computer, same product id
                              DEM         Planetary Computer Copernicus DEM, tiles over the window
      -> S2, S1: the whole-pixel offset where the Core band equals the source (searched +-3 px around the stored
         origin); the true origin; the stored origin's error
      -> DEM: the shift of the stored grid (in metres) at which the resampled source best matches Core
      -> docs/evidence/pixel_match.csv

Run as a Slurm job (needs CDSE keys):
    sbatch --cpus-per-task=8 --mem=16G -J pixel-match ~/slurm-templates/cpu.sbatch \
        .venv/bin/python docs/evidence/pixel_match_set.py [samples per dataset] [datasets, comma-separated]
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from fsspec.parquet import open_parquet_file
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from majortom.build.archives import ARCHIVES                        # noqa: E402
from majortom.build.backends import stac                             # noqa: E402

OUT = Path(__file__).parent / "pixel_match.csv"
PER_DATASET = int(sys.argv[1]) if len(sys.argv) > 1 else 10    # samples per dataset
PAD = 3                                                              # pixels searched around the stored origin
BANDS = {"Core-S2L2A": ["B04", "B11", "B01"], "Core-S2L1C": ["B04", "B11", "B01"], "Core-S1RTC": ["vv"],
         "Core-DEM": ["DEM"]}


def metadata(dataset: str) -> pd.DataFrame:
    url = f"https://huggingface.co/datasets/Major-TOM/{dataset}/resolve/main/metadata.parquet"
    columns = ["grid_cell", "centre_lat", "centre_lon", "crs", "parquet_url", "parquet_row"]
    names = pq.ParquetFile(fsspec.open(url).open()).schema_arrow.names
    return pq.read_table(fsspec.open(url).open(), columns=columns + (["product_id"] if "product_id" in names else [])
                         ).to_pandas()


def stratum(row) -> tuple:
    """Spread the picks: processing version (Sentinel-2), hemisphere, and closeness to a UTM zone edge."""
    version = re.search(r"_N(\d{2})(\d{2})_", str(getattr(row, "product_id", "")))
    group = None if not version else ("02-03" if version.group(1) in ("02", "03") else
                                      "04.00" if version.group(0) == "_N0400_" else
                                      "05.00" if version.group(0) == "_N0500_" else "05.xx")
    edge = min((row.centre_lon + 180) % 6, 6 - (row.centre_lon + 180) % 6) < 0.3
    return group, row.centre_lat >= 0, edge


def pick(table: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """n rows, one per stratum in turn, so each kind of sample is represented."""
    rng = random.Random(seed)
    groups = {key: list(g.index) for key, g in table.groupby(table.apply(stratum, axis=1))}
    for rows in groups.values():
        rng.shuffle(rows)
    picked = []
    while len(picked) < n and any(groups.values()):
        for key in sorted(groups, key=str):
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return table.loc[picked]


def core_band(row, band: str) -> tuple[np.ndarray, Affine, str]:
    with open_parquet_file(row.parquet_url, columns=[band], row_groups=[int(row.parquet_row)]) as f:
        table = pq.ParquetFile(f).read_row_group(int(row.parquet_row), columns=[band])
    with MemoryFile(table[band][0].as_py()) as m, m.open() as src:
        return src.read(1), src.transform, str(src.crs)


def source(dataset: str, row):
    """(archive, item, asset key per band) of the product a Core sample was cut from, or None."""
    pid = row.product_id if hasattr(row, "product_id") else None
    if dataset.startswith("Core-S2"):
        level = "l2a" if "L2A" in dataset else "l1c"
        keys = {"B04": "B04_10m", "B11": "B11_20m", "B01": "B01_60m"} if level == "l2a" else {}
        items = stac.search(ARCHIVES["cdse"], f"sentinel-2-{level}", ids=[pid], max_items=1)
        if items:
            return "cdse", items[0], keys
        # the version Core used is gone; the same acquisition reprocessed keeps the tile grid (geometry check only)
        when, tile = re.search(r"_(\d{8}T\d{6})_", pid).group(1), re.search(r"_T(\w{5})_", pid).group(1)
        day = f"{when[:4]}-{when[4:6]}-{when[6:8]}"
        box = (row.centre_lon - .05, row.centre_lat - .05, row.centre_lon + .05, row.centre_lat + .05)
        found = stac.search(ARCHIVES["cdse"], f"sentinel-2-{level}", datetime=f"{day}T00:00:00Z/{day}T23:59:59Z",
                            bbox=box, max_items=10)
        items = [it for it in found if when in it.id and f"_T{tile}_" in it.id]
        if items:
            return "cdse", items[0], keys
        if level == "l2a":                                            # originals: Planetary Computer keeps them
            pc_id = re.sub(r"_N\d{4}_", "_", pid)
            items = stac.search(ARCHIVES["planetary-computer"], "sentinel-2-l2a", ids=[pc_id], max_items=1)
            if items:
                return "planetary-computer", items[0], {}
        return None
    if dataset == "Core-S1RTC":
        items = stac.search(ARCHIVES["planetary-computer"], "sentinel-1-rtc", ids=[pid], max_items=1)
        return ("planetary-computer", items[0], {}) if items else None
    return "planetary-computer", None, {}


def native_match(core: np.ndarray, core_tf: Affine, href: str) -> dict:
    """Whole-pixel offset with the most equal pixels, around the stored origin; the stored origin's error. If no
    offset gives a copy (a reprocessed source), the offset with the best correlation, as -correlation."""
    with rasterio.open(href) as src:
        tf = src.transform
        res = tf.a
        col_f = (core_tf.c - tf.c) / res
        row_f = (tf.f - core_tf.f) / res
        c0, r0 = int(np.floor(col_f)) - PAD, int(np.floor(row_f)) - PAD
        n_r, n_c = core.shape
        block = src.read(1, window=Window(c0, r0, n_c + 2 * PAD, n_r + 2 * PAD), boundless=True, fill_value=0)
    best = (-1.0, 0, 0, 0)
    for dr in range(2 * PAD + 1):
        for dc in range(2 * PAD + 1):
            sub = block[dr:dr + n_r, dc:dc + n_c].astype(np.int64)
            valid = (sub != 0) & (core != 0)
            if not valid.any():
                continue
            for shift in (0, 1000, -1000):                           # an offset added or removed in between
                share = float((core.astype(np.int64)[valid] == sub[valid] + shift).mean())
                if share > best[0]:
                    best = (share, r0 + dr, c0 + dc, shift)
    share, row, col, shift = best
    if share < 0.99:                                         # not a copy (reprocessed): best correlation instead
        scores = []
        for dr in range(2 * PAD + 1):
            for dc in range(2 * PAD + 1):
                sub = block[dr:dr + n_r, dc:dc + n_c].astype(float)
                valid = (sub != 0) & (core != 0)
                if valid.sum() > 100:
                    scores.append((np.corrcoef(core[valid], sub[valid])[0, 1], r0 + dr, c0 + dc))
        if scores:
            corr, row, col = max(scores)
            share, shift = -float(corr), 0                   # negative: a correlation, not a share of equal pixels
    true_x, true_y = tf.c + col * res, tf.f - row * res
    return dict(equal_share=share, value_shift=shift, offset_col=col, offset_row=row, col_frac=col_f % 1,
                row_frac=row_f % 1, error_x=core_tf.c - true_x, error_y=core_tf.f - true_y,
                rounded_col=round(col_f) == col, rounded_row=round(row_f) == row)


def read_around(href: str, core_tf: Affine, n: tuple[int, int], crs: str, res: float) -> tuple[np.ndarray, Affine, str]:
    """The part of a source file around a Core window (with a margin), read once, with its transform and CRS."""
    with rasterio.open(href) as src:
        left, top = core_tf.c, core_tf.f
        bounds = transform_bounds(crs, src.crs, left - 200, top - n[0] * res - 200, left + n[1] * res + 200, top + 200)
        window = src.window(*bounds).round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
        return src.read(1, window=window).astype("float32"), src.window_transform(window), str(src.crs)


def shift_match(core: np.ndarray, core_tf: Affine, crs: str, sources: list[tuple[np.ndarray, Affine, str]],
                res: float, nodata: float | None = 0) -> dict:
    """Shift of the stored grid (metres) at which the source, resampled onto it, best matches Core: a 5 m search
    over +-30 m, then 1 m around the best."""
    valid_core = core != nodata if nodata is not None else np.isfinite(core)

    def resampled(dx: float, dy: float) -> np.ndarray:
        out = np.full(core.shape, np.nan, "float32")
        for array, transform, src_crs in sources:
            part = np.full(core.shape, np.nan, "float32")
            reproject(array, part, src_transform=transform, src_crs=src_crs, src_nodata=nodata,
                      dst_transform=core_tf * Affine.translation(dx / res, -dy / res), dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
            out = np.where(np.isnan(out), part, out)
        return out

    def rmse(dx: float, dy: float) -> float:
        diff = (resampled(dx, dy) - core)[valid_core]
        return float(np.sqrt(np.nanmean(diff ** 2)))

    best = min(((rmse(dx, dy), dx, dy) for dx in range(-30, 31, 5) for dy in range(-30, 31, 5)))
    best = min(((rmse(dx, dy), dx, dy) for dx in range(best[1] - 4, best[1] + 5)
                for dy in range(best[2] - 4, best[2] + 5)))
    return dict(rmse_stored=rmse(0, 0), rmse_best=best[0], shift_x=best[1], shift_y=best[2])


def dem_match(core: np.ndarray, core_tf: Affine, crs: str) -> dict:
    """The Copernicus DEM tiles over the window, resampled onto Core's grid: the best shift."""
    n = core.shape[0]
    bounds = transform_bounds(crs, "EPSG:4326", core_tf.c, core_tf.f - n * 30, core_tf.c + n * 30, core_tf.f)
    items = stac.search(ARCHIVES["planetary-computer"], "cop-dem-glo-30", bbox=bounds, max_items=4)
    sources = [read_around(item.assets["data"].href, core_tf, core.shape, crs, 30) for item in items]
    return shift_match(core, core_tf, crs, sources, 30, nodata=None)


def main() -> None:
    rows = []
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(BANDS)   # e.g. "Core-S1RTC"
    for dataset in only:
        table = metadata(dataset)
        for row in pick(table, PER_DATASET).itertuples():
            base = dict(dataset=dataset, grid_cell=row.grid_cell, product_id=getattr(row, "product_id", None),
                        crs=row.crs, lat=round(row.centre_lat, 3), lon=round(row.centre_lon, 3))
            try:
                found = source(dataset, row)
            except Exception as error:
                rows.append(dict(base, status=f"search failed: {error}"[:160]))
                continue
            if found is None:
                rows.append(dict(base, status="source not found"))
                continue
            name, item, keys = found
            archive = ARCHIVES[name]
            for band in BANDS[dataset]:
                try:
                    core, core_tf, crs = core_band(row, band)
                    with stac.session(archive):
                        if dataset == "Core-DEM":
                            result = dict(method="resampled", **dem_match(core, core_tf, crs))
                        else:
                            href = item.assets[stac.find_asset(item, keys.get(band, band))].href
                            with rasterio.open(href) as src:
                                same_crs = src.crs == rasterio.crs.CRS.from_user_input(crs)
                            native = native_match(core, core_tf, href) if same_crs else {"equal_share": -1.0}
                            if native["equal_share"] >= 0.99:        # native: an exact copy at a whole-pixel offset
                                result = dict(method="native", **native)
                            elif same_crs and dataset.startswith("Core-S2") and native["equal_share"] < -0.9:
                                result = dict(method="native, reprocessed source", **native)   # geometry only
                            else:                                    # resampled: other zone, or not on the grid
                                res = abs(core_tf.a)
                                around = read_around(href, core_tf, core.shape, crs, res)
                                result = dict(method="resampled",
                                              **shift_match(core.astype("float32"), core_tf, crs, [around], res))
                    rows.append(dict(base, band=band, archive=name, source=item.id if item else "cop-dem-glo-30",
                                     status="ok", core_shape=f"{core.shape[0]}x{core.shape[1]}", **result))
                except Exception as error:
                    rows.append(dict(base, band=band, archive=name, status=f"failed: {error}"[:160]))
            last = rows[-1]
            print(dataset, row.grid_cell, last.get("status"), last.get("method"),
                  last.get("equal_share", last.get("rmse_best")), flush=True)
    table = pd.DataFrame(rows)
    if len(only) < len(BANDS) and OUT.exists():              # replace only the datasets run this time
        old = pd.read_csv(OUT)
        table = pd.concat([old[~old.dataset.isin(only)], table], ignore_index=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
