"""CPC gauge counts: raw files read north-up (plain or gzip), mean over days, 3 x 3 sum, one column added once."""
import gzip

import numpy as np
import pandas as pd

from majortom.cells import Cell
from majortom.climate.gauges import add_gauges, neighbourhood


def _day(folder, stamp, gauges, compress):
    """A raw CPC day: rain (unused) then gnum, row 0 south, as CPC writes it."""
    data = np.stack([np.zeros((360, 720)), np.flipud(gauges)]).astype("<f4").tobytes()
    path = folder / f"PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx.{stamp}{'.gz' if compress else '.RT'}"
    path.write_bytes(gzip.compress(data) if compress else data)


def test_gauges_column(tmp_path):
    raw = tmp_path / "raw" / "2020"
    raw.mkdir(parents=True)
    g = np.zeros((360, 720))
    g[67, 711] = 4                                     # north-up pixel 56.25-56.5°N, 355.5-356°E (Loch Lomond's 3 x 3)
    _day(raw, "20200101", g, compress=True)
    _day(raw, "20200102", 2 * g, compress=False)       # mean over the two days: 6 gauges in that pixel
    cells = pd.DataFrame([(624, -29), (0, 0)], columns=["grid_row_u", "grid_col_r"])
    cells.insert(0, "grid_cell", [Cell(r, c).name for r, c in zip(cells.grid_row_u, cells.grid_col_r)])
    cpc = tmp_path / "CPC-Monthly"
    (cpc / "series" / "year=2020").mkdir(parents=True)
    cells.to_parquet(cpc / "cells.parquet", index=False)
    month = cpc / "series" / "year=2020" / "01.parquet"
    pd.DataFrame({"grid_cell": cells.grid_cell, "year": np.int16(2020), "month": np.int8(1),
                  "cpc_precip": np.float32(1), "cpc_valid": np.float32(1)}).to_parquet(month, index=False)

    assert add_gauges(cpc, tmp_path / "raw", workers=1) == 1
    t = pd.read_parquet(month)
    assert list(t.columns) == ["grid_cell", "year", "month", "cpc_precip", "cpc_gauges", "cpc_valid"]
    assert np.isclose(t.cpc_gauges[0], 6, atol=1e-3)            # Loch Lomond: within the gauge pixel's 3 x 3
    assert t.cpc_gauges[1] == 0                                 # the equator at 0°: no gauges nearby
    add_gauges(cpc, tmp_path / "raw", workers=1)                # a rerun leaves the file as it is
    assert pd.read_parquet(month).equals(t)


def test_neighbourhood_wraps_in_longitude_not_over_the_pole():
    f = np.zeros((360, 720))
    f[0, 0] = 1
    lit = set(zip(*np.nonzero(neighbourhood(f))))
    assert lit == {(0, 0), (0, 1), (0, 719), (1, 0), (1, 1), (1, 719)}
