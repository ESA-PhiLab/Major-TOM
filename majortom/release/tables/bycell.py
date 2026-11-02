"""By-cell layout: every cell's whole series together, so one place over time reads only its own pages.

    <src>/cells.parquet + series/year=YYYY/MM.parquet (as layout.repack writes them: every monthly file lists the
    cells in cells.parquet's order, with the same row groups)
      -> one band at a time (band = (grid_row_u + 1002) * 20 // 2004, as the climatology files):
           the band's cells, in slices of ~4,096 cells (whole output row groups); for each slice, from every monthly
           file only the row groups holding it (known from the positions); memory: one slice x months x columns
      -> rows: the band's cells along the curve, each with all its months (year, month) in turn
      -> <dst>/by_cell/band=NN.parquet: row groups of 1,024 cells, zstd, statistics on grid_row_u and grid_col_r;
         a page index and pages of about one cell's series, values without dictionaries (byte-stream split): a
         reader that knows a cell's rows reads only its pages (ERA5: 0.8 MB for one variable, footer included)
      -> checked: 50 random cells in 3 months equal the source rows; only then renamed into place
    Values are unchanged, missing values stay null. The monthly files suit maps of one month; this layout, time
    series of places (an app's chart, training on series).

    python -m majortom.release tables by-cell --src <dataset folder> --out <dataset folder> [--workers 4]
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROWS, BANDS = 2004, 20
CELLS = 1024                                        # cells per row group: small footers, with the page index
SLICE = 4 * CELLS                                   # cells held in memory at once (ERA5: ~28 GB per worker)
POSITION = ["grid_cell", "grid_row_u", "grid_col_r", "year", "month"]


def _months(src: Path) -> list[tuple[int, int, Path]]:
    files = sorted(src.glob("series/year=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly files (series/year=*/*.parquet) in {src}")
    return [(int(f.parent.name.split("=")[1]), int(f.stem), f) for f in files]


def _groups(path: Path) -> np.ndarray:
    """Start row of each row group of a file, and the end: [0, n0, n0 + n1, ..., num_rows]."""
    md = pq.ParquetFile(path).metadata
    return np.concatenate([[0], np.cumsum([md.row_group(g).num_rows for g in range(md.num_row_groups)])])


def _band(src: Path, dst: Path, band: int, seed: int = 0) -> int:
    cells = pq.read_table(src / "cells.parquet")
    rows = cells.column("grid_row_u").to_numpy().astype(np.int64)
    at = np.flatnonzero((rows + ROWS // 2) * BANDS // ROWS == band)                        # positions, curve order
    if not len(at):
        return 0
    months = _months(src)
    schema = pq.read_schema(months[0][2])
    names = [f.name for f in schema if f.name not in POSITION]
    bounds = _groups(months[0][2])
    for _, _, path in months:
        if not np.array_equal(_groups(path), bounds):
            raise ValueError(f"{path}: row groups differ from {months[0][2]}; repack the dataset first")
    path = dst / "by_cell" / f"band={band:02d}.parquet"
    if path.exists():                                       # written and checked by an earlier run (renamed last)
        return len(at)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer = None
    floats = [n for n in names if pa.types.is_floating(schema.field(n).type)]
    for s in range(0, len(at), SLICE):
        table = _slice(at[s:s + SLICE], months, cells, rows, schema, names, bounds)
        if writer is None:                                  # dictionaries would be read whole before any page
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd", write_statistics=True,
                                      write_page_index=True, use_dictionary=POSITION, use_byte_stream_split=floats,
                                      data_page_size=4096, write_batch_size=len(months))  # ~one cell per page
        writer.write_table(table, row_group_size=CELLS * len(months))
    writer.close()
    _check(tmp, months, at, cells, names, bounds, np.random.default_rng(seed + band))
    os.replace(tmp, path)                                                                # into place only once checked
    return len(at)


def _slice(at: np.ndarray, months, cells: pa.Table, rows: np.ndarray, schema: pa.Schema, names: list[str],
           bounds: np.ndarray) -> pa.Table:
    """Rows of the cells at positions `at`, each with all its months: from every monthly file, only the row groups
    holding them."""
    values = {n: np.empty((len(months), len(at)), schema.field(n).type.to_pandas_dtype()) for n in names}
    valid = {n: np.empty((len(months), len(at)), bool) for n in names}
    need = np.unique(np.searchsorted(bounds, at, side="right") - 1)                      # row groups holding them
    read_at = np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in need])         # positions of the rows read
    take = pa.array(np.searchsorted(read_at, at))
    for k, (_, _, path) in enumerate(months):
        t = pq.ParquetFile(path).read_row_groups(need.tolist(), columns=names)
        for n in names:
            col = t.column(n).take(take)
            valid[n][k] = col.is_valid().to_numpy(zero_copy_only=False)
            values[n][k] = col.fill_null(0).to_numpy(zero_copy_only=False)
    n, m = len(at), len(months)
    return pa.table({
        "grid_cell": pa.DictionaryArray.from_arrays(pa.array(np.repeat(np.arange(n, dtype=np.int32), m)),
                                                    cells.column("grid_cell").take(pa.array(at)).combine_chunks()),
        "grid_row_u": np.repeat(rows[at].astype(np.int16), m),
        "grid_col_r": np.repeat(cells.column("grid_col_r").to_numpy()[at].astype(np.int16), m),
        "year": np.tile(np.array([y for y, _, _ in months], np.int16), n),
        "month": np.tile(np.array([mo for _, mo, _ in months], np.int8), n),
        **{c: pa.array(values[c].T.ravel(), mask=~valid[c].T.ravel(), type=schema.field(c).type) for c in names},
    })


def _rows_of(path: Path, groups, columns: list[str], want: list[str]) -> pd.DataFrame:
    """The rows of cells `want` from row groups `groups` of a file, read one row group at a time."""
    f, parts = pq.ParquetFile(path), []
    for g in sorted(set(groups)):
        part = f.read_row_group(int(g), columns=columns).to_pandas()
        part["grid_cell"] = part.grid_cell.astype(str)
        parts.append(part[part.grid_cell.isin(want)])
    return pd.concat(parts, ignore_index=True)


def _check(path: Path, months, at: np.ndarray, cells: pa.Table, names: list[str], bounds: np.ndarray, rng) -> None:
    """50 random cells of the band in 3 months: the written rows equal the source rows.

    Only the row groups holding the sampled cells are read (a grid_cell filter cannot skip row groups, so it would
    read whole files: ~60 GB for a large ERA5 band). In the band file a cell's row group is rank // CELLS; in a
    monthly file, the one whose rows hold its position.
    """
    pick = np.sort(rng.choice(len(at), min(50, len(at)), replace=False))
    want_cells = sorted(set(cells.column("grid_cell").take(pa.array(at[pick])).to_pylist()))
    written = _rows_of(path, pick // CELLS, ["grid_cell", "year", "month", *names], want_cells)
    for y, mo, f in [months[0], months[len(months) // 2], months[-1]]:
        groups = np.searchsorted(bounds, at[pick], side="right") - 1
        src = _rows_of(f, groups, ["grid_cell", *names], want_cells).set_index("grid_cell").loc[want_cells]
        got = written[(written.year == y) & (written.month == mo)]
        got = got.set_index("grid_cell").loc[src.index, names]
        if not got.equals(src[names]):
            raise ValueError(f"{path}: {y}-{mo:02d} differs from {f}")


def by_cell(src: Path | str, dst: Path | str, workers: int = 4) -> int:
    """Write `dst`/by_cell/band=NN.parquet for every band of `src`; returns the number of cells written."""
    src, dst = Path(src), Path(dst)
    with ProcessPoolExecutor(workers) as pool:
        jobs = {b: pool.submit(_band, src, dst, b) for b in range(BANDS)}
        total = 0
        for b, job in jobs.items():
            n = job.result()
            total += n
            print(f"band {b:02d}: {n:,} cells", flush=True)
    return total
