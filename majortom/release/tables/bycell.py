"""By-cell layout: every cell's whole series together, so one place over time reads one row group.

    <src>/cells.parquet + series/year=YYYY/MM.parquet (as layout.repack writes them: every monthly file lists the
    cells in cells.parquet's order, with the same row groups)
      -> one band at a time (band = (grid_row_u + 1002) * 20 // 2004, as the climatology files):
           the band's cells; from every monthly file only the row groups holding them (known from their positions);
           their values (memory: the band's cells x months x columns, ~1/20 of the dataset)
      -> rows: the band's cells along the curve, each with all its months (year, month) in turn
      -> <dst>/by_cell/band=NN.parquet: row groups of whole cells (~65,536 rows), grid_cell dictionary-encoded, zstd,
         statistics on grid_row_u and grid_col_r (so a reader finds a cell's row group without reading others)
      -> checked: 50 random cells in 3 months equal the source rows
    Values are unchanged, missing values stay null. The monthly files suit maps of one month; this layout, time
    series of places (an app's chart, training on series).

    python -m majortom.release tables by-cell --src <dataset folder> --out <dataset folder> [--workers 4]
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS, BANDS, ROW_GROUP = 2004, 20, 65_536
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
    values = {n: np.empty((len(months), len(at)), schema.field(n).type.to_pandas_dtype()) for n in names}
    valid = {n: np.empty((len(months), len(at)), bool) for n in names}
    bounds = _groups(months[0][2])
    for k, (_, _, path) in enumerate(months):
        if not np.array_equal(_groups(path), bounds):
            raise ValueError(f"{path}: row groups differ from {months[0][2]}; repack the dataset first")
        need = np.unique(np.searchsorted(bounds, at, side="right") - 1)                  # row groups holding the band
        t = pq.ParquetFile(path).read_row_groups(need.tolist(), columns=names)
        read_at = np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in need])     # positions of the rows read
        take = np.searchsorted(read_at, at)
        for n in names:
            col = t.column(n).take(pa.array(take))
            valid[n][k] = col.is_valid().to_numpy(zero_copy_only=False)
            values[n][k] = col.fill_null(0).to_numpy(zero_copy_only=False)
    n, m = len(at), len(months)
    table = pa.table({
        "grid_cell": pa.DictionaryArray.from_arrays(pa.array(np.repeat(np.arange(n, dtype=np.int32), m)),
                                                    cells.column("grid_cell").take(pa.array(at)).combine_chunks()),
        "grid_row_u": np.repeat(rows[at].astype(np.int16), m),
        "grid_col_r": np.repeat(cells.column("grid_col_r").to_numpy()[at].astype(np.int16), m),
        "year": np.tile(np.array([y for y, _, _ in months], np.int16), n),
        "month": np.tile(np.array([mo for _, mo, _ in months], np.int8), n),
        **{c: pa.array(values[c].T.ravel(), mask=~valid[c].T.ravel(), type=schema.field(c).type) for c in names},
    })
    path = dst / "by_cell" / f"band={band:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=max(1, ROW_GROUP // m) * m, compression="zstd", write_statistics=True)
    _check(path, months, at, cells, names, np.random.default_rng(seed + band))
    return n


def _check(path: Path, months, at: np.ndarray, cells: pa.Table, names: list[str], rng) -> None:
    """50 random cells of the band in 3 months: the written rows equal the source rows."""
    pick = rng.choice(len(at), min(50, len(at)), replace=False)
    want_cells = sorted(set(cells.column("grid_cell").take(pa.array(at[pick])).to_pylist()))
    only = [("grid_cell", "in", want_cells)]                                  # read the sampled cells only
    written = pq.read_table(path, columns=["grid_cell", "year", "month", *names], filters=only).to_pandas()
    written["grid_cell"] = written.grid_cell.astype(str)
    for y, mo, f in [months[0], months[len(months) // 2], months[-1]]:
        src = pq.read_table(f, columns=["grid_cell", *names], filters=only).to_pandas().set_index("grid_cell").loc[want_cells]
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
