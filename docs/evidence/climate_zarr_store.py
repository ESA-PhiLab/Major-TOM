"""Zarr test store for the climate globe: CPC-Monthly precipitation on the sinusoidal (row, column) array.

    CPC-Monthly release: cells.parquet (positions) + series/year=YYYY/MM.parquet (cpc_precip, mm per month)
      -> per month, the (2004, 4008) array (row = grid_row_u + 1002, column = grid_col_r + 2004), NaN where no
         cell or no value: the same values as the Parquet files, float32
      -> <out>/cpc-precip.zarr (Zarr v3, zstd; chunks with no value are not written, readers fill them with NaN)
           map/0   (24, 2004, 4008)  chunks (1, 512, 512)    2024-01..2025-12, one month per chunk
           map/1   (24, 1002, 2004)  chunks (1, 512, 512)    2 x 2 block means (missing values ignored)
           map/2   (24, 501, 1002)   chunks (1, 512, 512)    4 x 4 block means
           series  (564, 2004, 4008) shards (564, 512, 512) of chunks (564, 32, 32): 1979-01..2025-12, so one
                                     cell's 564 months sit in one chunk
    For comparing reads with the Parquet layout in apps/climate-globe (map months and one cell's series);
    uploaded to major-tom/climate/_experiments and removed after the tests. Needs ~40 GB of memory (Slurm).

    python docs/evidence/climate_zarr_store.py /data/Major-TOM/climate-release/CPC-Monthly /data/Major-TOM/experiments
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from zarr.codecs import ZstdCodec

ROWS, COLS = 2004, 4008
YEARS, MAP_YEARS = (1979, 2025), (2024, 2025)


def positions(src: Path) -> np.ndarray:
    """Flat index into the (row, column) array of each row of the dataset's files (they share cells.parquet's order)."""
    t = pq.ParquetFile(src / "cells.parquet").read(columns=["grid_row_u", "grid_col_r"])
    r, c = (t.column(n).to_numpy().astype(np.int64) for n in ("grid_row_u", "grid_col_r"))
    return (r + ROWS // 2) * COLS + c + COLS // 2


def month_array(src: Path, index: np.ndarray, year: int, month: int) -> np.ndarray:
    v = pq.ParquetFile(src / "series" / f"year={year}" / f"{month:02d}.parquet").read(columns=["cpc_precip"])
    image = np.full(ROWS * COLS, np.nan, np.float32)
    image[index] = v.column(0).to_numpy(zero_copy_only=False).astype(np.float32)          # null -> NaN
    return image.reshape(ROWS, COLS)


def coarsen(a: np.ndarray, f: int) -> np.ndarray:
    """f x f block means over the last two axes, missing values ignored (NaN where a block has none)."""
    h, w = -(-a.shape[-2] // f) * f, -(-a.shape[-1] // f) * f
    padded = np.full(a.shape[:-2] + (h, w), np.nan, np.float32)
    padded[..., : a.shape[-2], : a.shape[-1]] = a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                  # all-NaN blocks
        return np.nanmean(padded.reshape(a.shape[:-2] + (h // f, f, w // f, f)), axis=(-3, -1)).astype(np.float32)


def array(group, name: str, shape: tuple, chunks: tuple, shards: tuple | None = None):
    return group.create_array(name, shape=shape, chunks=chunks, shards=shards, dtype="float32", fill_value=np.nan,
                              compressors=ZstdCodec(level=3), config={"write_empty_chunks": False})


def build(src: Path, out: Path) -> None:
    index = positions(src)
    months = [(y, m) for y in range(YEARS[0], YEARS[1] + 1) for m in range(1, 13)]
    root = zarr.open_group(out / "cpc-precip.zarr", mode="w", zarr_format=3)
    root.attrs.update({
        "variable": "cpc_precip", "unit": "mm per month", "source": "CPC-Monthly (major-tom/climate), unchanged values",
        "grid": "Major TOM 10 km cells: row = grid_row_u + 1002, column = grid_col_r + 2004",
        "series_months": [f"{y}-{m:02d}" for y, m in months],
        "map_months": [f"{y}-{m:02d}" for y, m in months if MAP_YEARS[0] <= y <= MAP_YEARS[1]],
    })

    series = np.empty((len(months), ROWS, COLS), np.float32)
    for k, (y, m) in enumerate(months):
        series[k] = month_array(src, index, y, m)
    print(f"read {len(months)} months", flush=True)

    s = array(root, "series", series.shape, (len(months), 32, 32), (len(months), 512, 512))
    for i in range(0, ROWS, 512):                                                        # one shard row at a time
        s[:, i:i + 512, :] = series[:, i:i + 512, :]
    print("series written", flush=True)

    first = months.index((MAP_YEARS[0], 1))
    level = series[first:]
    maps = root.create_group("map")
    for lv in range(3):
        if lv:
            level = coarsen(level, 2)
        array(maps, str(lv), level.shape, (1, 512, 512))[:] = level
        print(f"map level {lv}: {level.shape}", flush=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
