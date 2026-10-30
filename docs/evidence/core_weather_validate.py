"""Check the Core-Weather source folders before merging, from Parquet metadata only (no data read).

    <root>/<source>/hourly|daily/year=YYYY/<date>.parquet   (one folder per source or group, as built)
      -> days per source (against the acquisition days of acquisitions.parquet)
      -> rows per day: the same in every source (the merge is an outer join on cell, acquisition and step)
      -> per column: min, max (row-group statistics), largest share of nulls in a day
      -> <out>/<kind>.csv (one row per source and column) and a printed summary; problems marked CHECK

    python docs/evidence/core_weather_validate.py /data/Major-TOM/Core-Weather <out folder>     (Slurm CPU job)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SOURCES = ["era5"] + [f"era5-g{i}" for i in range(1, 9)] + [f"era5land-edh-g{i}" for i in range(1, 7)] + \
          ["imerg", "cams", "cpc", "s5p"]
KEYS = {"grid_cell", "acquired", "step_h", "step_d"}


def file_stats(path: Path) -> tuple[str, int, dict]:
    """(day, rows, {column: (min, max, nulls)}) of one file, from its footer."""
    md = pq.ParquetFile(path).metadata
    out = defaultdict(lambda: [None, None, 0])
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for i in range(rg.num_columns):
            c = rg.column(i)
            name, st = c.path_in_schema, c.statistics
            if name in KEYS or st is None:
                continue
            o = out[name]
            o[2] += st.null_count or 0
            if st.has_min_max and isinstance(st.min, (int, float)):
                o[0] = st.min if o[0] is None else min(o[0], st.min)
                o[1] = st.max if o[1] is None else max(o[1], st.max)
    return path.stem, md.num_rows, dict(out)


def main(root: str, out: str) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("hourly", "daily"):
        rows_by_day, table = defaultdict(dict), []
        with ProcessPoolExecutor(16) as pool:
            for source in SOURCES:
                files = sorted((root / source / kind).glob("year=*/*.parquet"))
                if not files:
                    continue
                stats = list(pool.map(file_stats, files, chunksize=32))
                columns = defaultdict(lambda: [None, None, 0.0])
                for day, rows, cols in stats:
                    rows_by_day[day][source] = rows
                    for name, (lo, hi, nulls) in cols.items():
                        c = columns[name]
                        c[0] = lo if c[0] is None else (c[0] if lo is None else min(c[0], lo))
                        c[1] = hi if c[1] is None else (c[1] if hi is None else max(c[1], hi))
                        c[2] = max(c[2], nulls / rows if rows else 0.0)
                for name, (lo, hi, share) in columns.items():
                    table.append({"source": source, "column": name, "min": lo, "max": hi, "max_null_share": share})
                print(f"{kind:6} {source:16} {len(files):5} days", flush=True)
        uneven = [d for d, by in rows_by_day.items() if len(set(by.values())) > 1]
        print(f"{kind}: {len(rows_by_day)} days; days whose sources differ in rows: {len(uneven)}"
              + (f" CHECK, e.g. {uneven[0]}: {rows_by_day[uneven[0]]}" if uneven else ""), flush=True)
        pd.DataFrame(table).to_csv(out / f"{kind}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
