"""Training-style reads of CPC precipitation series from source.coop: by-cell test layout vs the monthly files.

    cells.parquet (local release copy) -> 1,000 random cells, and 1,000 neighbouring cells (consecutive on the curve)
      by cell:  each cell's band file (_experiments/cpc-bycell) -> the row groups holding the cells (116 cells each)
      monthly:  every series/year=YYYY/MM.parquet (the release) -> cpc_precip of every row group holding a cell
    pyarrow over HTTP (fsspec), 16 reads at a time, as a data loader would. Prints time, MB (compressed column
    chunks read, from the files' metadata) and cells per second, and checks both layouts give the same values.

    python docs/evidence/climate_bycell_bench.py /data/Major-TOM/climate-release/CPC-Monthly
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq

BASE = "https://data.source.coop/major-tom/climate"
BYCELL = f"{BASE}/_experiments/cpc-bycell/by_cell"
MONTHLY = f"{BASE}/CPC-Monthly/series"
ROWS, BANDS, CELLS_PER_GROUP, PARALLEL = 2004, 20, 116, 16
fs = fsspec.filesystem("https")


def chunk_bytes(md, group: int, columns: list[str]) -> int:
    rg = md.row_group(group)
    return sum(rg.column(i).total_compressed_size for i in range(rg.num_columns) if rg.column(i).path_in_schema in columns)


def by_cell(cells, picked: np.ndarray) -> tuple[dict, int, int]:
    """Series of the picked cells (positions in cells.parquet) from the by-cell files: {cell: 564 values}."""
    band = (cells.grid_row_u.to_numpy().astype(np.int64) + ROWS // 2) * BANDS // ROWS
    order = {}                                                    # position within its band (Hilbert order kept)
    for b in np.unique(band):
        at = np.flatnonzero(band == b)
        order.update(zip(at.tolist(), range(len(at))))
    jobs = sorted({(int(band[i]), order[i] // CELLS_PER_GROUP) for i in picked})
    names = set(cells.grid_cell.to_numpy()[picked])

    def read(job):
        b, g = job
        with fs.open(f"{BYCELL}/band={b:02d}.parquet", block_size=2**20, cache_type="none") as f:
            pf = pq.ParquetFile(f)
            t = pf.read_row_group(g, columns=["grid_cell", "cpc_precip"]).to_pandas()
            return t[t.grid_cell.isin(names)], chunk_bytes(pf.metadata, g, ["grid_cell", "cpc_precip"])

    with ThreadPoolExecutor(PARALLEL) as pool:
        parts = list(pool.map(read, jobs))
    series = {}
    for t, _ in parts:
        for name, g in t.groupby("grid_cell", observed=True):
            series[str(name)] = g.cpc_precip.to_numpy(np.float32)
    return series, sum(n for _, n in parts), len(jobs)


def monthly(cells, picked: np.ndarray, months: list[tuple[int, int]]) -> tuple[dict, int, int]:
    """Series of the picked cells from the monthly files: every file, the row groups holding them."""
    groups = sorted({int(i) // 65_536 for i in picked})                   # release row groups of 65,536 rows

    def read(ym):
        y, m = ym
        with fs.open(f"{MONTHLY}/year={y}/{m:02d}.parquet", block_size=2**20, cache_type="none") as f:
            pf = pq.ParquetFile(f)
            v = np.full(pf.metadata.num_rows, np.nan, np.float32)
            for g in groups:
                start = g * 65_536
                col = pf.read_row_group(g, columns=["cpc_precip"]).column(0).to_numpy(zero_copy_only=False)
                v[start:start + len(col)] = col
            return v[picked], sum(chunk_bytes(pf.metadata, g, ["cpc_precip"]) for g in groups)

    with ThreadPoolExecutor(PARALLEL) as pool:
        parts = list(pool.map(read, months))
    values = np.stack([v for v, _ in parts])                               # (months, picked)
    names = cells.grid_cell.to_numpy()[picked]
    return {str(n): values[:, k] for k, n in enumerate(names)}, sum(n for _, n in parts), len(months) * len(groups)


def run(name, fn, *args):
    t0 = time.time()
    series, nbytes, reads = fn(*args)
    s = time.time() - t0
    print(f"{name:34} {len(series):5} cells  {s:6.1f} s  {nbytes / 1e6:8.1f} MB  {reads:6} row-group reads  "
          f"{len(series) / s:7.1f} cells/s", flush=True)
    return series


if __name__ == "__main__":
    src = Path(sys.argv[1])
    cells = pq.ParquetFile(src / "cells.parquet").read(columns=["grid_cell", "grid_row_u", "grid_col_r"]).to_pandas()
    months = [(y, m) for y in range(1979, 2026) for m in range(1, 13)]
    rng = np.random.default_rng(0)
    random = np.sort(rng.choice(len(cells), 1000, replace=False))
    start = int(rng.integers(0, len(cells) - 1000))
    near = np.arange(start, start + 1000)
    a = run("by cell, 1,000 random cells", by_cell, cells, random)
    run("by cell, 1,000 neighbouring cells", by_cell, cells, near)
    b = run("monthly files, 1,000 random cells", monthly, cells, random, months)
    same = all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
    print("same values in both layouts:", same, f"({len(a)} cells)")
