"""Publishing layout of a [Source]-Monthly folder or a static table: rows along a Hilbert curve, small row groups.

    <src>/cells.parquet + series/year=YYYY/MM.parquet + climatology/band=NN.parquet    (rows in build order)
      or <src>/static.parquet                                                            (one row per cell)
      -> each cell's centroid -> its position on the Hilbert curve (majortom.release.order.hilbert, as releases use)
      -> every file's rows reordered along the curve (climatology: by curve, then month)
         and grid_row_u, grid_col_r added after grid_cell (int16), so a region is a filter on two integers
      -> <dst>/same paths, row groups of ROW_GROUP rows with statistics, zstd
      -> each written file is read back and must equal its source row for row (same cells, same values;
         float NaN is stored as null, so missing values are null in every file)
    Nearby cells sit in the same row groups, so a point or region query reads one small row group per file
    instead of a whole column (~16x less for a point time series of CPC-Monthly). Values are not changed.

    python -m majortom.release tables repack --src /data/Major-TOM/CPC-Monthly --out /data/Major-TOM/climate-release/CPC-Monthly
    --drop <column> leaves a column out (and its climatology <column>_mean, <column>_std), e.g. era5_ptype
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...cells import R_KM, n_rows
from ..order import hilbert

ROW_GROUP = 65_536


def centroids(rows: np.ndarray, cols: np.ndarray, d: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of cell midpoints, vectorised from majortom.cells.Cell.centroid."""
    dlat = 180 / n_rows(d)
    lat = rows * dlat
    n_cols = np.maximum(np.ceil(2 * np.pi * R_KM * np.cos(np.radians(lat)) / d), 1)
    dlon = 360 / n_cols
    return cols * dlon + dlon / 2, lat + dlat / 2


def curve_rank(cells: pd.DataFrame) -> pd.Series:
    """grid_cell -> rank along the Hilbert curve (ties, which do not occur at 10 km, by name)."""
    lon, lat = centroids(cells["grid_row_u"].to_numpy(), cells["grid_col_r"].to_numpy())
    order = np.lexsort((cells["grid_cell"].to_numpy(), hilbert(lon, lat)))
    return pd.Series(np.argsort(order), index=cells["grid_cell"].to_numpy())


def _with_position(table: pa.Table, rows: np.ndarray, cols: np.ndarray) -> pa.Table:
    at = table.schema.get_field_index("grid_cell") + 1
    table = table.add_column(at, "grid_row_u", pa.array(rows.astype(np.int16)))
    return table.add_column(at + 1, "grid_col_r", pa.array(cols.astype(np.int16)))


def _nan_as_null(table: pa.Table) -> pa.Table:
    """Float NaN -> null, so missing values are missing everywhere (SQL tools treat NaN as a value)."""
    for i, field in enumerate(table.schema):
        if pa.types.is_floating(field.type):
            col = table.column(i)
            table = table.set_column(i, field, pc.if_else(pc.is_nan(col), pa.scalar(None, field.type), col))
    return table


def _dropped(names: list[str], drop: tuple[str, ...]) -> list[str]:
    """Columns of `names` left out: each of `drop`, and its climatology statistics."""
    return [n for n in names if any(n in (d, f"{d}_mean", f"{d}_std") for d in drop)]


def _one(src: Path, dst: Path, rank: pd.Series, where: pd.DataFrame, row_group: int, drop: tuple[str, ...] = ()) -> int:
    """Reorder one file along the curve, write it, read it back and check it against the source (less `drop`)."""
    table = pq.read_table(src)
    table = _nan_as_null(table.drop_columns(_dropped(table.column_names, drop)))
    names = table.column("grid_cell").to_numpy(zero_copy_only=False)
    key = rank.reindex(names).to_numpy()
    if np.isnan(key).any():
        raise ValueError(f"{src}: cells missing from cells.parquet")
    sort = np.lexsort((table.column("month").to_numpy(), key)) if "month" in table.column_names and \
        "year" not in table.column_names else np.argsort(key, kind="stable")
    out = table.take(pa.array(sort))
    pos = where.reindex(out.column("grid_cell").to_numpy(zero_copy_only=False))
    out = _with_position(out, pos["grid_row_u"].to_numpy(), pos["grid_col_r"].to_numpy())
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")                         # renamed only once checked: a killed run
    pq.write_table(out, tmp, row_group_size=row_group, compression="zstd", write_statistics=True)   # leaves no
    back = pq.ParquetFile(tmp).read().drop_columns(["grid_row_u", "grid_col_r"])                  # half file
    if not back.take(pa.array(np.argsort(sort))).equals(table):
        raise ValueError(f"{dst}: does not match {src} after reordering")
    os.replace(tmp, dst)
    return out.num_rows


def repack(src: Path | str, dst: Path | str, workers: int | None = None, row_group: int = ROW_GROUP,
           drop: tuple[str, ...] = (), only: tuple[str, ...] = ("series", "climatology", "static")) -> int:
    """Write the publishing layout of `src` into `dst`, without the columns in `drop`; returns the files written."""
    src, dst = Path(src), Path(dst)
    if (src / "cells.parquet").exists():
        cells = pd.read_parquet(src / "cells.parquet")
    else:                                                          # a static table: its own rows are the cells
        cells = pd.read_parquet(src / "static.parquet", columns=["grid_cell"])
        from ...cells import cell_from_name
        rc = np.array([(c.row, c.col) for c in map(cell_from_name, cells["grid_cell"])])
        cells = cells.assign(grid_row_u=rc[:, 0], grid_col_r=rc[:, 1])
    rank = curve_rank(cells)
    where = cells.set_index("grid_cell")[["grid_row_u", "grid_col_r"]]
    patterns = {"series": "series/year=*/*.parquet", "climatology": "climatology/band=*.parquet", "static": "static.parquet"}
    files = [f for part in only for f in sorted(src.glob(patterns[part]))]
    if (src / "cells.parquet").exists():
        ordered = cells.iloc[np.argsort(rank.reindex(cells["grid_cell"]).to_numpy())]
        dst.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(ordered, preserve_index=False), dst / "cells.parquet",
                       row_group_size=row_group, compression="zstd")
    workers = workers or min(len(files), len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(_one, f, dst / f.relative_to(src), rank, where, row_group, tuple(drop)) for f in files]
        for i, job in enumerate(jobs, 1):
            job.result()
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} files", flush=True)
    return len(files)
