"""By-cell layout: same values as the monthly files, each cell's months together, row groups never split a cell."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from majortom.release.tables.bycell import by_cell
from majortom.release.tables.layout import repack
from test_release_tables_layout import _monthly          # pytest puts tests/ on the path


def test_by_cell_keeps_values_and_whole_cells(tmp_path):
    _monthly(tmp_path / "raw", n=600)
    tmp = tmp_path / "rel"
    repack(tmp_path / "raw", tmp, workers=2, row_group=64)                      # the published monthly layout
    for f in (tmp / "series").rglob("*.parquet"):                                # a missing value to carry over
        t = pd.read_parquet(f); t.loc[3, "x_t2m"] = np.nan; t.to_parquet(f, index=False, row_group_size=64)
    cells = by_cell(tmp, tmp, workers=2)
    assert cells == len(pd.read_parquet(tmp / "cells.parquet"))
    months = sorted((tmp / "series").rglob("*.parquet"))
    for path in sorted((tmp / "by_cell").glob("band=*.parquet")):
        t = pq.read_table(path).to_pandas()
        t["grid_cell"] = t.grid_cell.astype(str)
        n = len(months)
        assert (t.groupby("grid_cell", sort=False).size() == n).all()                         # every month, once
        md = pq.ParquetFile(path).metadata
        assert all(md.row_group(g).num_rows % n == 0 for g in range(md.num_row_groups))      # whole cells per group
        for f in months:
            y, m = int(f.parent.name.split("=")[1]), int(f.stem)
            src = pd.read_parquet(f).set_index("grid_cell")
            got = t[(t.year == y) & (t.month == m)].set_index("grid_cell")
            for c in ("x_t2m", "x_valid"):
                assert np.array_equal(got[c].to_numpy(), src.loc[got.index, c].to_numpy(), equal_nan=True)
