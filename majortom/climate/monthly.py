"""[Source]-Monthly: monthly values for every cell over the whole record, and a climatology (docs/CLIMATE.md 4).

    every cell of the grid -> weights on the source grid (once)
      -> the first month fixes the cells kept: those with data (land-only sources: cells with land)
         -> <out>/cells.parquet          one row per cell, in the order every month file uses
      -> per month: reader.month_field per variable -> cell means
         -> <out>/series/year=YYYY/MM.parquet         a rerun skips months already written
    series + reference years -> climatology(): per cell and calendar month, mean, std, n_years per variable
         -> <out>/climatology/band=NN.parquet         grid row bands, so one cell's 12 months share a file

Every month file holds the same cells in the same order, so the climatology adds months up by position,
one variable at a time, without joining on names.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..cells import Cell, n_rows
from .sources import SOURCES, reader
from .weights import cell_means, weights


def all_cells(d: float = 10.0) -> pd.DataFrame:
    """Every cell of the Major TOM grid at spacing `d` km: grid_cell, grid_row_u, grid_col_r."""
    n = n_rows(d)
    rows, cols = [], []
    for row in range(-(n // 2), n - n // 2):
        k = Cell(row, 0, d).n_cols
        cols.append(np.arange(-(k // 2), k - k // 2))
        rows.append(np.full(k, row))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    name = (pd.Series(np.abs(rows)).astype(str) + np.where(rows >= 0, "U", "D") + "_" +
            pd.Series(np.abs(cols)).astype(str) + np.where(cols >= 0, "R", "L"))
    return pd.DataFrame({"grid_cell": name, "grid_row_u": rows, "grid_col_r": cols})


def _month(rdr, src, variables, month: pd.Timestamp, w, land, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """cells x variables values, and the first variable's valid share, for one month."""
    with ThreadPoolExecutor(workers) as pool:
        fields = list(pool.map(lambda v: rdr.month_field(src, v, month), variables))
    results = [cell_means(w, f, land if v in src.land_only else None) for v, f in zip(variables, fields)]
    return np.stack([r[0] for r in results], axis=1).astype(np.float32), results[0][1].astype(np.float32)


def monthly(source: str, out: Path | str, years: tuple[int, int], variables: list[str] | None = None,
            cells: pd.DataFrame | None = None, workers: int = 4, progress: bool = True) -> int:
    """Write the monthly series of `source` for `years` (inclusive) to `out`; returns months written."""
    out, src = Path(out), SOURCES[source]
    rdr, variables = reader(src), list(variables or src.variables)
    land = rdr.land_pixels(src) if src.land_only and hasattr(rdr, "land_pixels") else None
    months = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-01", freq="MS")
    if hasattr(rdr, "valid_months"):
        first, last = rdr.valid_months(src)
        months = months[(months >= first) & (months <= last)]
    kept = out / "cells.parquet"
    if kept.exists():
        cells = pd.read_parquet(kept)
    else:
        cells = all_cells() if cells is None else cells.reset_index(drop=True)
    w = weights(cells, src.grid)
    written, start = 0, time.monotonic()
    for month in months:
        path = out / "series" / f"year={month.year}" / f"{month.month:02d}.parquet"
        if path.exists():
            continue
        values, valid = _month(rdr, src, variables, month, w, land, workers)
        if not kept.exists():                                    # the first month fixes the cells kept
            keep = np.isfinite(values).any(axis=1)
            cells, values, valid = cells[keep].reset_index(drop=True), values[keep], valid[keep]
            kept.parent.mkdir(parents=True, exist_ok=True)
            cells.to_parquet(kept, index=False)
            w = weights(cells, src.grid)
        table = pd.DataFrame({f"{src.name}_{v}": values[:, j] for j, v in enumerate(variables)})
        table.insert(0, "month", np.int8(month.month))
        table.insert(0, "year", np.int16(month.year))
        table.insert(0, "grid_cell", cells["grid_cell"].to_numpy())
        table[f"{src.name}_valid"] = valid
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        written += 1
        if progress:
            print(f"{month:%Y-%m}: {len(cells):,} cells, {written} months in {(time.monotonic() - start) / 60:.1f} min",
                  flush=True)
    return written


def climatology(out: Path | str, period: tuple[int, int] = (1991, 2020), bands: int = 20) -> int:
    """Per cell and calendar month over `period`: mean, standard deviation across years, and years with data.

    Written band by band from per-variable arrays; returns the number of rows written.
    """
    import pyarrow.parquet as pq

    out = Path(out)
    cells = pd.read_parquet(out / "cells.parquet")
    files = {(y, m): out / "series" / f"year={y}" / f"{m:02d}.parquet"
             for y in range(period[0], period[1] + 1) for m in range(1, 13)}
    missing = [k for k, f in files.items() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} months of {period} not built, e.g. {missing[0]}")
    columns = [c for c in pq.read_schema(files[(period[0], 1)]).names
               if c not in ("grid_cell", "year", "month") and not c.endswith("_valid")]
    stats: dict[str, np.ndarray] = {}
    for column in columns:                                        # one variable at a time: bounded memory
        total = np.zeros((len(cells), 12))
        squares = np.zeros((len(cells), 12))
        count = np.zeros((len(cells), 12), np.int16)
        for (y, m), f in files.items():
            x = pq.read_table(f, columns=[column]).column(0).to_numpy(zero_copy_only=False).astype(np.float64)
            ok = np.isfinite(x)
            total[ok, m - 1] += x[ok]
            squares[ok, m - 1] += x[ok] ** 2
            count[ok, m - 1] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
            std = np.sqrt(np.maximum(squares / count - mean ** 2, 0) * count / np.maximum(count - 1, 1))
        stats[f"{column}_mean"], stats[f"{column}_std"] = mean.astype(np.float32), std.astype(np.float32)
        stats.setdefault("n_years", count)
    band = (cells["grid_row_u"].to_numpy() + n_rows(10.0) // 2) * bands // n_rows(10.0)
    target = out / "climatology"
    target.mkdir(parents=True, exist_ok=True)
    rows = 0
    for b in np.unique(band):
        at = np.flatnonzero(band == b)
        part = pd.DataFrame({"grid_cell": np.repeat(cells["grid_cell"].to_numpy()[at], 12),
                             "month": np.tile(np.arange(1, 13, dtype=np.int8), len(at)),
                             **{k: v[at].ravel() for k, v in stats.items()}})
        part.to_parquet(target / f"band={b:02d}.parquet", index=False)
        rows += len(part)
    return rows
