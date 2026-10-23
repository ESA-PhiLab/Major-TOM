"""majortom.climate.monthly: the series in a fixed cell order, months resumed, climatology (offline, fake source).

The fake field: year * 100 + month in the northern hemisphere, empty (like sea for a land-only source) in the south.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from majortom.cells import cell_at
from majortom.climate.sources import Source
from majortom.climate.weights import Grid

monthly_module = importlib.import_module("majortom.climate.monthly")
FAKE = Source("fake", "-", Grid(west=-180, north=90, step=1, width=360, height=180), {"a": "a"}, {"a": "1"}, "1MS",
              period="month")


@pytest.fixture
def reads(monkeypatch):
    calls = []

    def month_field(src, variable, month):
        calls.append(month)
        f = np.full((180, 360), month.year * 100 + month.month, dtype=float)
        f[90:] = np.nan                                              # south: no data
        return f

    monkeypatch.setitem(monthly_module.SOURCES, "fake", FAKE)
    monkeypatch.setattr(monthly_module, "reader", lambda src: SimpleNamespace(month_field=month_field))
    return calls


def cells():
    places = [(-4.6, 56.1), (10.0, 45.0), (150.0, -33.0)]
    rows = [cell_at(*p) for p in places]
    return pd.DataFrame({"grid_cell": [c.name for c in rows], "grid_row_u": [c.row for c in rows],
                         "grid_col_r": [c.col for c in rows]})


def test_series_keeps_cells_with_data_and_resumes(reads, tmp_path):
    assert monthly_module.monthly("fake", tmp_path, (2020, 2020), cells=cells(), progress=False) == 12
    kept = pd.read_parquet(tmp_path / "cells.parquet")
    assert list(kept.grid_cell) == list(cells().grid_cell[:2])       # the southern cell has no data
    july = pd.read_parquet(tmp_path / "series" / "year=2020" / "07.parquet")
    assert list(july.fake_a) == [202007, 202007] and list(july.month) == [7, 7]
    reads.clear()
    assert monthly_module.monthly("fake", tmp_path, (2020, 2021), cells=cells(), progress=False) == 12
    assert all(m.year == 2021 for m in reads)                        # 2020 not read again


def test_climatology(reads, tmp_path):
    monthly_module.monthly("fake", tmp_path, (2001, 2003), cells=cells(), progress=False)
    rows = monthly_module.climatology(tmp_path, period=(2001, 2003), bands=4)
    assert rows == 2 * 12
    clim = pd.concat(pd.read_parquet(p) for p in sorted((tmp_path / "climatology").glob("*.parquet")))
    march = clim[(clim.month == 3)].iloc[0]
    assert march.fake_a_mean == pytest.approx(200203) and march.fake_a_std == pytest.approx(100)
    assert march.n_years == 3


def test_all_cells_cover_the_grid():
    c = monthly_module.all_cells()
    assert len(c) == c.grid_cell.nunique() > 5_000_000 and (c.grid_cell == "624U_29L").any()
