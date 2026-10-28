"""Check the [Source]-Monthly datasets and ERA5-Static before publishing, from Parquet metadata only (no data read).

    <root>/<dataset>/cells.parquet                 -> number of cells, no duplicates
    <root>/<dataset>/series/year=YYYY/MM.parquet  -> every month from the first to the last present (no gaps);
                                                    rows == cells in every file; per variable: min, max,
                                                    months with no values at all, largest null share; valid in [0, 1]
    <root>/<dataset>/climatology/band=NN.parquet  -> rows == cells x 12; n_years range; per statistic: min, max, nulls
    <root>/ERA5-Static/static.parquet             -> rows, per field min, max, nulls
      -> <out>/<dataset>.csv (one row per column) and a printed summary; problems marked CHECK
    Min and max come from each file's column statistics, written by pyarrow for every row group.

    python docs/evidence/monthly_validate.py /data/Major-TOM <out folder>      (Slurm CPU job)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATASETS = ["CPC-Monthly", "IMERG-Monthly", "CAMS-Monthly", "ERA5Land-Monthly", "ERA5-Monthly-v2"]
KEYS = {"grid_cell", "year", "month", "n_years"}


def stats(path: Path) -> tuple[int, dict[str, tuple[float, float, int]]]:
    """Rows and (min, max, nulls) per numeric column, over the file's row groups."""
    meta = pq.ParquetFile(path).metadata
    out = {}
    for j in range(meta.num_columns):
        name = meta.schema.column(j).name
        lo, hi, nulls = np.inf, -np.inf, 0
        for g in range(meta.num_row_groups):
            s = meta.row_group(g).column(j).statistics
            if s is None:
                continue
            nulls += s.null_count or 0
            if s.has_min_max and isinstance(s.min, (int, float)):
                lo, hi = min(lo, float(s.min)), max(hi, float(s.max))
        out[name] = (lo, hi, nulls)
    return meta.num_rows, out


def table(results: list[tuple[int, dict]], cells: int) -> pd.DataFrame:
    """One row per column: overall min and max, files where it is entirely null, largest null share."""
    rows = {}
    for n, cols in results:
        for name, (lo, hi, nulls) in cols.items():
            r = rows.setdefault(name, dict(column=name, min=np.inf, max=-np.inf, empty_files=0, max_null_share=0.0))
            r["min"], r["max"] = min(r["min"], lo), max(r["max"], hi)
            r["empty_files"] += int(nulls == n)
            r["max_null_share"] = max(r["max_null_share"], nulls / n if n else 1.0)
    return pd.DataFrame(rows.values())


def series(folder: Path, cells: int, pool) -> tuple[list[str], pd.DataFrame]:
    files = sorted(folder.glob("series/year=*/*.parquet"))
    months = pd.PeriodIndex([f"{f.parent.name[5:]}-{f.stem}" for f in files], freq="M")
    problems = []
    gaps = pd.period_range(months.min(), months.max(), freq="M").difference(months)
    if len(gaps):
        problems.append(f"{len(gaps)} missing months, first {gaps[0]}")
    results = list(pool.map(stats, files, chunksize=16))
    wrong = [f"{m}" for m, (n, _) in zip(months, results) if n != cells]
    if wrong:
        problems.append(f"{len(wrong)} months with rows != cells, first {wrong[0]}")
    t = table(results, cells)
    valid = t[t.column.str.endswith("_valid")]
    if ((valid["min"] < 0) | (valid["max"] > 1)).any():
        problems.append("valid share outside [0, 1]")
    empty = t[(t.empty_files > 0) & ~t.column.isin(KEYS)]
    if not empty.empty:
        problems.append("variables with months of no values: " + ", ".join(f"{c} ({n})" for c, n in
                                                                       zip(empty.column, empty.empty_files)))
    return [f"series {months.min()} to {months.max()}, {len(files)} months"] + problems, t


def climatology(folder: Path, cells: int, pool) -> tuple[list[str], pd.DataFrame]:
    files = sorted(folder.glob("climatology/band=*.parquet"))
    if not files:
        return ["no climatology"], pd.DataFrame()
    results = list(pool.map(stats, files))
    rows = sum(n for n, _ in results)
    t = table(results, cells)
    years = t[t.column == "n_years"].iloc[0]
    notes = [f"climatology {len(files)} bands, {rows:,} rows, n_years {years['min']:.0f}-{years['max']:.0f}"]
    if rows != cells * 12:
        notes.append(f"climatology rows {rows:,} != cells x 12 ({cells * 12:,})")
    return notes, t


def main(root: str, out: str) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(16) as pool:
        for name in DATASETS:
            folder = root / name
            ids = pq.read_table(folder / "cells.parquet", columns=["grid_cell"]).column(0)
            cells = len(ids)
            notes = [f"{cells:,} cells" + ("" if len(set(ids.to_pylist())) == cells else " (DUPLICATES: CHECK)")]
            s_notes, s = series(folder, cells, pool)
            c_notes, c = climatology(folder, cells, pool)
            notes += s_notes + c_notes
            pd.concat([s.assign(table="series"), c.assign(table="climatology")]).to_csv(out / f"{name}.csv", index=False)
            flag = "CHECK" if len(s_notes) > 1 or len(c_notes) > 1 else "ok"
            print(f"\n{name}: {flag}\n  " + "\n  ".join(notes), flush=True)
            print(s[~s.column.isin(KEYS)][["column", "min", "max", "max_null_share"]].round(4).to_string(index=False))
    n, cols = stats(root / "ERA5-Static" / "static.parquet")
    t = table([(n, cols)], n)
    t.to_csv(out / "ERA5-Static.csv", index=False)
    print(f"\nERA5-Static: {n:,} rows\n" + t[t.column != "grid_cell"].round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
