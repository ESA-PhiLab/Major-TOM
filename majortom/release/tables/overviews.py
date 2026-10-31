"""Overview tables: block means of a dataset's monthly files, so a map of the whole globe reads one small file.

    <src>/series/year=YYYY/MM.parquet (grid_cell, grid_row_u, grid_col_r, year, month, <variables>)
      -> per file: blocks of 2^level x 2^level cells of the (row, column) array,
         block_row = (grid_row_u + 1002) >> level, block_col = (grid_col_r + 2004) >> level
      -> per block: the mean of each float column over its cells with a value (missing values ignored; cells are
         equal-area, so the plain mean), and n_cells, its cells with a value in any column
      -> <dst>/overviews/year=YYYY/MM.parquet: blocks with a value only, rows by block_row then block_col, zstd
    Level 2 (4 x 4 cells, ~40 km) suits a globe-wide map: CPC ~1.6 MB a month for three variables.

    python -m majortom.release tables overviews --src <dataset folder> --out <dataset folder> [--level 2]
    --columns <names> limits them to some columns (ERA5-Monthly: the common surface variables, not all 202)

    <src>/climatology/band=NN.parquet (one row per cell and calendar month)
      -> per calendar month: every band read in batches, that month's rows, the <variable>_mean columns
      -> the same block means -> <dst>/climatology_overviews/MM.parquet: each variable's normal per block
    python -m majortom.release tables overviews --climatology --src <dataset folder> --out <dataset folder>
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROWS, COLS = 2004, 4008                                    # the (row, column) array of 10 km cells
POSITION = {"grid_cell", "grid_row_u", "grid_col_r", "year", "month"}


def block_means(table: pa.Table, level: int, columns: list[str] | None = None) -> pa.Table:
    """One file's rows -> one row per block with a value: block_row, block_col, n_cells, each float column's mean."""
    w = -(-COLS // 2 ** level)
    r = table.column("grid_row_u").to_numpy().astype(np.int64) + ROWS // 2
    c = table.column("grid_col_r").to_numpy().astype(np.int64) + COLS // 2
    block = (r >> level) * w + (c >> level)
    size = (-(-ROWS // 2 ** level)) * w
    names = [f.name for f in table.schema if f.name not in POSITION and pa.types.is_floating(f.type)
             and (columns is None or f.name in columns)]
    any_value = np.zeros(len(block), bool)
    means = {}
    for name in names:
        x = table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)          # null -> NaN
        ok = np.isfinite(x)
        any_value |= ok
        total = np.bincount(block[ok], weights=x[ok], minlength=size)
        count = np.bincount(block[ok], minlength=size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                               # blocks without a value
            means[name] = (total / count).astype(np.float32)
    n_cells = np.bincount(block[any_value], minlength=size)
    keep = np.flatnonzero(n_cells)                                                        # ascending: row, then col
    return pa.table({"block_row": (keep // w).astype(np.int16), "block_col": (keep % w).astype(np.int16),
                     "n_cells": n_cells[keep].astype(np.int32),
                     **{n: pa.array(means[n][keep], from_pandas=True) for n in names}})  # NaN -> null


def _one(src: Path, dst: Path, level: int, columns: list[str] | None = None) -> int:
    read = None if columns is None else ["grid_row_u", "grid_col_r", *columns]
    out = block_means(pq.ParquetFile(src).read(columns=read), level, columns)        # no columns from folder names
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst, compression="zstd", write_statistics=True)
    return out.num_rows


def overviews(src: Path | str, dst: Path | str, level: int = 2, workers: int | None = None,
              columns: list[str] | None = None) -> int:
    """Write the overview of every monthly file of `src` (all float columns, or `columns`) into `dst`/overviews;
    returns the number of files."""
    src, dst = Path(src), Path(dst)
    files = sorted(src.glob("series/year=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly files (series/year=*/*.parquet) in {src}")
    workers = workers or min(len(files), len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(_one, f, dst / "overviews" / f.relative_to(src / "series"), level, columns) for f in files]
        for i, job in enumerate(jobs, 1):
            job.result()
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} files", flush=True)
    return len(files)


def _climatology_month(src: Path, dst: Path, month: int, level: int) -> int:
    parts = []
    for band in sorted(src.glob("climatology/band=*.parquet")):
        f = pq.ParquetFile(band)
        means = [n for n in f.schema_arrow.names if n.endswith("_mean")]
        for batch in f.iter_batches(batch_size=262_144, columns=["grid_row_u", "grid_col_r", "month", *means]):
            t = pa.Table.from_batches([batch])
            parts.append(t.filter(pc.equal(t.column("month"), month)))
    out = block_means(pa.concat_tables(parts), level)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst, compression="zstd", write_statistics=True)
    return out.num_rows


def climatology_overviews(src: Path | str, dst: Path | str, level: int = 2, workers: int | None = None) -> int:
    """Write `dst`/climatology_overviews/MM.parquet: block means of every _mean column of `src`'s climatology, one
    file per calendar month; returns the number of files."""
    src, dst = Path(src), Path(dst)
    if not any(src.glob("climatology/band=*.parquet")):
        raise FileNotFoundError(f"no climatology files (climatology/band=NN.parquet) in {src}")
    workers = workers or min(12, len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(workers) as pool:
        jobs = {m: pool.submit(_climatology_month, src, dst / "climatology_overviews" / f"{m:02d}.parquet", m, level)
                for m in range(1, 13)}
        for m, job in jobs.items():
            print(f"month {m:02d}: {job.result():,} blocks", flush=True)
    return len(jobs)
