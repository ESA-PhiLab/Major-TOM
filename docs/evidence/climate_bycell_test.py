"""Parquet test layouts for the climate globe and time-series training: CPC-Monthly by cell, and overview tables.

    CPC-Monthly release: cells.parquet (Hilbert order) + series/year=YYYY/MM.parquet (cpc_precip, cpc_tmax,
    cpc_tmin, cpc_valid per cell)
      -> every month read once: values[variable][month, cell]
      -> by_cell/band=NN.parquet (NN as the climatology: band = (grid_row_u + 1002) * 20 // 2004)
           rows: the band's cells in Hilbert order, each with its 564 months (year, month) in turn
           columns: grid_cell (dictionary), grid_row_u, grid_col_r, year, month, the four variables
           row groups of 116 cells x 564 months (65,424 rows): one cell's series never spans two row groups
      -> overviews/level=L/year=YYYY/MM.parquet, L = 1, 2 (2024-2025)
           2^L x 2^L block means of the (row, column) array (missing values ignored), blocks with a value only
           columns: block_row = (grid_row_u + 1002) >> L, block_col = (grid_col_r + 2004) >> L, the three variables
    Values are unchanged (by_cell) or means of unchanged values (overviews). Uploaded to
    major-tom/climate/_experiments/cpc-bycell and removed after the tests. Needs ~40 GB of memory (Slurm).

    python docs/evidence/climate_bycell_test.py /data/Major-TOM/climate-release/CPC-Monthly /data/Major-TOM/experiments
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS, COLS, BANDS = 2004, 4008, 20
YEARS, OVERVIEW_YEARS = (1979, 2025), (2024, 2025)
VARIABLES = ["cpc_precip", "cpc_tmax", "cpc_tmin", "cpc_valid"]
CELLS_PER_GROUP = 116                                      # x 564 months = 65,424 rows, close to the release's 65,536


def read_months(src: Path, months: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """values[variable][month, cell], cells in cells.parquet order (every file of the release shares it)."""
    n = pq.ParquetFile(src / "cells.parquet").metadata.num_rows
    values = {v: np.empty((len(months), n), np.float32) for v in VARIABLES}
    for k, (y, m) in enumerate(months):
        t = pq.ParquetFile(src / "series" / f"year={y}" / f"{m:02d}.parquet").read(columns=VARIABLES)
        for v in VARIABLES:
            values[v][k] = t.column(v).to_numpy(zero_copy_only=False)                     # null -> NaN
    return values


def by_cell(src: Path, out: Path, months: list[tuple[int, int]], values: dict[str, np.ndarray]) -> None:
    cells = pq.ParquetFile(src / "cells.parquet").read().to_pandas()
    band = (cells.grid_row_u.to_numpy().astype(np.int64) + ROWS // 2) * BANDS // ROWS
    year = np.array([y for y, _ in months], np.int16)
    month = np.array([m for _, m in months], np.int8)
    (out / "by_cell").mkdir(parents=True, exist_ok=True)
    for b in range(BANDS):
        at = np.flatnonzero(band == b)                                                    # Hilbert order kept
        if not len(at):
            continue
        n, t = len(at), len(months)
        names = pa.array(cells.grid_cell.to_numpy()[at])
        table = pa.table({
            "grid_cell": pa.DictionaryArray.from_arrays(pa.array(np.repeat(np.arange(n, dtype=np.int32), t)), names),
            "grid_row_u": np.repeat(cells.grid_row_u.to_numpy()[at].astype(np.int16), t),
            "grid_col_r": np.repeat(cells.grid_col_r.to_numpy()[at].astype(np.int16), t),
            "year": np.tile(year, n), "month": np.tile(month, n),
            **{v: pa.array(values[v][:, at].T.ravel(), from_pandas=True) for v in VARIABLES},   # NaN -> null
        })
        pq.write_table(table, out / "by_cell" / f"band={b:02d}.parquet", row_group_size=CELLS_PER_GROUP * t,
                       compression="zstd", write_statistics=True)
        print(f"band {b:02d}: {n:,} cells, {table.num_rows:,} rows", flush=True)


def overviews(src: Path, out: Path, months: list[tuple[int, int]], values: dict[str, np.ndarray]) -> None:
    cells = pq.ParquetFile(src / "cells.parquet").read(columns=["grid_row_u", "grid_col_r"])
    r = cells.column("grid_row_u").to_numpy().astype(np.int64) + ROWS // 2
    c = cells.column("grid_col_r").to_numpy().astype(np.int64) + COLS // 2
    for level in (1, 2):
        h, w = -(-ROWS // 2 ** level), -(-COLS // 2 ** level)
        block = (r >> level) * w + (c >> level)
        for k, (y, m) in enumerate(months):
            if not OVERVIEW_YEARS[0] <= y <= OVERVIEW_YEARS[1]:
                continue
            cols = {}
            for v in VARIABLES[:3]:
                x = values[v][k]
                ok = np.isfinite(x)
                total = np.bincount(block[ok], weights=x[ok], minlength=h * w)
                count = np.bincount(block[ok], minlength=h * w)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cols[v] = (total / count).astype(np.float32)                          # NaN where no value
            keep = np.flatnonzero(np.isfinite(np.vstack(list(cols.values()))).any(axis=0))
            table = pa.table({"block_row": (keep // w).astype(np.int16), "block_col": (keep % w).astype(np.int16),
                              **{v: pa.array(cols[v][keep], from_pandas=True) for v in cols}})
            path = out / "overviews" / f"level={level}" / f"year={y}" / f"{m:02d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path, compression="zstd", write_statistics=True)
        print(f"overviews level {level}: {h} x {w} blocks", flush=True)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2]) / "cpc-bycell"
    months = [(y, m) for y in range(YEARS[0], YEARS[1] + 1) for m in range(1, 13)]
    values = read_months(src, months)
    print(f"read {len(months)} months", flush=True)
    by_cell(src, out, months, values)
    overviews(src, out, months, values)
