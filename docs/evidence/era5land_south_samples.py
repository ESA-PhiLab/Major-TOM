"""Samples for the ERA5-Land fill south of 57.1°S (docs/CLIMATE.md 6.1), from the Core metadata.

    <core>/Core-S2L2A, Core-S2L1C, Core-S1RTC metadata.parquet (the samples Core-Weather was built for)
      -> rows whose cell's lower edge is south of 57.1°S (grid_row_u x 180 / 2004 < -57.1): the cells Earth Data
         Hub's copy misses, and those that straddle 57.1°S (partly covered there)
      -> <out>/south/Core-<name>/metadata.parquet: grid_cell, grid_row_u, grid_col_r, timestamp, product_id
         (stream's samples; the folder names the dataset, as stream's acquisitions.parquet records it)
      -> <out>/overlap/Core-<name>/metadata.parquet: the samples of the first two acquisition times in cells wholly
         between 56.5°S and 57.1°S (rows -635 to -630), which both copies cover: the end-to-end test compares the two

    python docs/evidence/era5land_south_samples.py /data/Major-TOM <out folder>
"""
import sys
from pathlib import Path

import pandas as pd

DLAT = 180 / 2004
COLUMNS = ["grid_cell", "grid_row_u", "grid_col_r", "timestamp", "product_id"]


def main(core: str, out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_parquet(Path(core) / f"Core-{d}" / "metadata.parquet", columns=COLUMNS).assign(dataset=d)
                      for d in ["S2L2A", "S2L1C", "S1RTC"]], ignore_index=True)
    south = rows[rows.grid_row_u * DLAT < -57.1]
    both = rows[rows.grid_row_u.between(-635, -630)].sort_values("timestamp")
    both = both[both.timestamp.isin(both.timestamp.drop_duplicates().head(2))]       # the first two acquisition times
    for name, table in [("south", south), ("overlap", both)]:
        for d, part in table.groupby("dataset"):
            path = out / name / f"Core-{d}" / "metadata.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part[COLUMNS].to_parquet(path, index=False)
    acquisitions = south.drop_duplicates(["grid_cell", "timestamp"])
    print(f"{len(south):,} samples, {len(acquisitions):,} acquisitions, {south.grid_cell.nunique():,} cells south of 57.1°S")
    print(f"overlap test: {both.drop_duplicates(['grid_cell', 'timestamp'])[['grid_cell', 'timestamp']].values.tolist()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
