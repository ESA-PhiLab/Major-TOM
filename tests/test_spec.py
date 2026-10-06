"""Cells and windows against docs/SPEC.md and the official grid (offline)."""
import numpy as np
import pandas as pd
import pytest

from majortom import Grid
from majortom.cells import Cell, cell_at, cells_from_points, cells_in
from majortom.spec import cell_coverage, window

SNOWBIRD = (-111.6556, 40.5829)


def test_snowbird_window_matches_spec_5_1():
    cell = cell_at(*SNOWBIRD, d=10)
    win = window(cell, lattice=60, origin=(0, 0))
    assert cell.name == "451U_946L"
    assert (win.crs, win.left, win.top, win.side) == ("EPSG:32612", 434640, 4494780, 10560)
    assert [win.pixels(r) for r in (10, 20, 60)] == [1056, 528, 176]
    assert cell_coverage(cell, win) == pytest.approx(1.0)


def test_window_side_is_a_multiple_of_twice_the_lattice():
    cell = cell_at(*SNOWBIRD, d=1)
    assert window(cell, lattice=60).side == 1080       # SPEC 5.2: 1.056 km rounded up to 120 m steps
    assert window(cell, lattice=25).side == 1100


def test_names_around_the_equator_and_meridian():
    assert cell_at(0.001, 0.001).name == "0U_0R"
    assert cell_at(-0.001, -0.001).name == "1D_1L"


def test_cells_match_the_official_grid():
    grid = Grid(1000, latitude_range=(-90, 90))        # the default range wraps points beyond it (WS2)
    rng = np.random.default_rng(0)
    lats, lons = rng.uniform(-80, 80, 200), rng.uniform(-179, 179, 200)
    rows, cols = grid.latlon2rowcol(lats, lons)
    ours = [cell_at(lon, lat, 1000).name for lon, lat in zip(lons, lats)]
    assert ours == [f"{r}_{c}" for r, c in zip(rows, cols)]


def test_antimeridian_wraps_to_the_easternmost_column():
    assert cell_at(-178.892, -63.121, 1000).name == "8D_7R"   # odd row: 7R spans 168E to 168W
    assert cell_at(180.0, 10.0, 10) == cell_at(-180.0, 10.0, 10)


def test_cells_from_points_groups_by_cell_and_query():
    points = pd.DataFrame({"lon": [-111.656, -111.655, -111.656], "lat": [40.583, 40.584, 40.583],
                           "class": "ski_resort", "datetime": ["2024-01-01", "2024-01-01", "2024-07-01"]})
    cells = cells_from_points(points, d=10)
    assert list(cells.grid_cell) == ["451U_946L", "451U_946L"]   # same cell, two different queries
    assert list(cells.n_points) == [2, 1]
    assert cells.index.is_unique and "class" in cells


def test_cells_in_covers_the_box():
    cells = cells_in((-111.8, 40.5, -111.6, 40.6), d=10)
    assert "451U_946L" in set(cells.grid_cell)
    assert len(cells) == len(set(cells.grid_cell))


def test_south_pole_cell_is_centred_on_the_pole():
    pole = Cell(-1002, 0, 10)                          # the disc cell at -90 deg (even row count)
    win = window(pole, lattice=60)
    assert win.crs == "EPSG:32761"
    assert (win.left + win.side / 2, win.top - win.side / 2) == (2_000_000, 2_000_000)
