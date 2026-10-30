"""CPC gauge counts per cell and month: how many rain gauges stand behind CPC-Monthly's precipitation.

    CPC's own daily files (readers/files.py, cpc_raw_files; NOAA PSL's netCDF copy leaves the count out):
    little-endian float32, 2 x 360 x 720 (rain in 0.1 mm, then gnum, the gauges in each 0.5° pixel), row 0 south;
    gzip-compressed up to some year (names ...lnx.19790101.gz, ...lnx.20060101RT.gz, then ...lnx.20230101.RT)
      -> per day: gnum, flipped north-up (the grid of SOURCES["cpc"])
      -> per month: each pixel's mean daily count, summed over its 3 x 3 neighbours (1.5° x 1.5°): CPC's analysis
         draws on nearby gauges, so the neighbourhood describes a pixel's support better than the pixel alone
      -> per cell: the area-weighted mean over the pixels it overlaps (weights.py, as cpc_precip)
      -> column cpc_gauges in <cpc>/series/year=YYYY/MM.parquet (the build folder), before cpc_valid; each file is
         rewritten whole and swapped in, and files that have the column already are skipped (a rerun continues)

    python -m majortom.climate gauges --cpc /data/Major-TOM/CPC-Monthly --raw <cache>/cpc/raw
"""
from __future__ import annotations

import gzip
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import SOURCES
from .weights import cell_means, weights

HEIGHT, WIDTH = 360, 720
COLUMN = "cpc_gauges"


def day_gauges(path: Path) -> np.ndarray:
    """Gauges per 0.5° pixel on one day, north-up (height x width)."""
    raw = gzip.open(path).read() if path.suffix == ".gz" else path.read_bytes()
    data = np.frombuffer(raw, "<f4")
    if data.size != 2 * HEIGHT * WIDTH:
        raise ValueError(f"{path}: {data.size} values, expected {2 * HEIGHT * WIDTH}")
    return np.flipud(np.clip(data.reshape(2, HEIGHT, WIDTH)[1], 0, None))


def neighbourhood(field: np.ndarray) -> np.ndarray:
    """Sum over each pixel's 3 x 3 neighbours: wrapping in longitude, nothing beyond the poles."""
    padded = np.pad(field, ((1, 1), (0, 0)))
    rows = padded[:-2] + padded[1:-1] + padded[2:]
    return np.roll(rows, 1, axis=1) + rows + np.roll(rows, -1, axis=1)


def month_gauges(raw: Path, year: int, month: int) -> tuple[np.ndarray, int]:
    """(3 x 3 sum of the mean daily gauges per pixel, days found) for one month."""
    days = sorted((raw / str(year)).glob(f"*.{year}{month:02d}[0-3][0-9]*"))
    if not days:
        raise FileNotFoundError(f"no CPC raw files for {year}-{month:02d} in {raw / str(year)}")
    total = np.zeros((HEIGHT, WIDTH), np.float64)
    for day in days:
        total += day_gauges(day)
    return neighbourhood(total / len(days)), len(days)


_w = None                                                         # weights, set once per worker process


def _start(cells: pd.DataFrame) -> None:
    global _w
    _w = weights(cells, SOURCES["cpc"].grid)


def _one(path: Path, raw: Path) -> str:
    table = pd.read_parquet(path)
    if COLUMN in table.columns:
        return f"{path.parent.name}/{path.name}: has {COLUMN}"
    year, month = int(table["year"].iloc[0]), int(table["month"].iloc[0])
    field, days = month_gauges(raw, year, month)
    values, _ = cell_means(_w, field)
    if len(values) != len(table):
        raise ValueError(f"{path}: {len(table)} rows, cells.parquet gives {len(values)}")
    table.insert(table.columns.get_loc("cpc_valid"), COLUMN, values.astype(np.float32))
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return f"{year}-{month:02d}: {days} days, mean {np.nanmean(values):.1f} gauges per cell"


def add_gauges(cpc: Path | str, raw: Path | str, workers: int | None = None) -> int:
    """Add cpc_gauges to every monthly file of the CPC build folder `cpc`; returns the files written or checked."""
    cpc, raw = Path(cpc), Path(raw)
    cells = pd.read_parquet(cpc / "cells.parquet")                # the build's row order
    files = sorted(cpc.glob("series/year=*/*.parquet"))
    workers = workers or min(8, len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(workers, initializer=_start, initargs=(cells,)) as pool:
        for i, line in enumerate(pool.map(_one, files, [raw] * len(files)), 1):
            if i % 12 == 0 or i == len(files):
                print(f"{i}/{len(files)} {line}", flush=True)
    return len(files)
