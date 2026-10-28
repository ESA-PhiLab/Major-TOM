"""majortom.climate.weights: cell x source pixel weights and cell means (offline)."""
import math

import numpy as np
import pandas as pd

from majortom.cells import Cell, cell_at
from majortom.climate import Grid, cell_means, weights

ERA5 = Grid(west=-0.125, north=90.125, step=0.25, width=1440, height=721)     # 0-360°, centred pixels
ONE = Grid(west=0, north=90, step=1, width=360, height=180)                   # 1° pixels, edges on whole degrees


def table(cells):
    return pd.DataFrame({"grid_row_u": [c.row for c in cells], "grid_col_r": [c.col for c in cells]})


def sums(w):
    return np.bincount(w.cell, w.weight, minlength=w.n_cells)


def across_lat(edge: float) -> Cell:
    """A cell whose latitude band crosses `edge`."""
    rows = [Cell(r, 0) for r in range(cell_at(0.0, edge).row - 1, cell_at(0.0, edge).row + 2)]
    return next(c for c in rows if c.lat < edge - 1e-9 and edge + 1e-9 < c.lat + c.dlat)


def across_lon(edge: float, lat: float = 45.3) -> Cell:
    """A cell whose longitude band crosses `edge`."""
    c = cell_at(edge, lat)
    cells = [Cell(c.row, c.col + k) for k in (-1, 0, 1)]
    return next(c for c in cells if c.lon < edge - 1e-9 and edge + 1e-9 < c.lon + c.dlon)


def test_rows_sum_to_one():
    cells = [cell_at(lon, lat) for lon, lat in [(-4.6, 56.1), (16.6, -13.5), (149.8, -33.4), (0.0, 0.0), (-179.95, 70.0)]]
    assert np.allclose(sums(weights(table(cells), ERA5)), 1)


def test_cell_inside_one_pixel():
    """10° pixels: a 10 km cell well inside one pixel gets weight 1 on it."""
    grid = Grid(west=0, north=90, step=10, width=36, height=18)
    w = weights(table([cell_at(5.0, 45.0)]), grid)
    assert list(w.pixel) == [4 * 36 + 0] and np.allclose(w.weight, 1)


def test_constant_field():
    cells = table([cell_at(lon, 50.0) for lon in range(-170, 180, 20)])
    value, valid = cell_means(weights(cells, ERA5), np.full((721, 1440), 7.5))
    assert np.allclose(value, 7.5) and np.allclose(valid, 1)


def test_missing_pixel_share():
    """A cell across the 1°E pixel edge, with its western pixel missing: valid is the eastern share."""
    cell = across_lon(1.0)
    field = np.full((180, 360), 4.0)
    field[:, 0] = np.nan
    value, valid = cell_means(weights(table([cell]), ONE), field)
    assert np.isclose(valid[0], (cell.lon + cell.dlon - 1.0) / cell.dlon) and np.isclose(value[0], 4.0)


def test_all_missing():
    field = np.full((180, 360), np.nan)
    value, valid = cell_means(weights(table([cell_at(5.0, 45.0)]), ONE), field)
    assert np.isnan(value[0]) and valid[0] == 0


def test_area_weighting_in_latitude():
    """A cell across the 46°N pixel edge: its weights follow sin(latitude), not degrees.
    (Major TOM row edges fall exactly on 45°N, so 46°N is used.)"""
    cell = across_lat(46.0)
    w = weights(table([cell]), ONE)
    north_row = 90 - 47                                              # the pixel row from 46°N to 47°N
    share = (math.sin(math.radians(cell.lat + cell.dlat)) - math.sin(math.radians(46))) / \
            (math.sin(math.radians(cell.lat + cell.dlat)) - math.sin(math.radians(cell.lat)))
    assert np.isclose(w.weight[w.pixel // 360 == north_row].sum(), share)


def test_wraps_across_180():
    """Cells either side of 180° land on ERA5's columns around 180° (its longitudes run 0-360°)."""
    west, east = cell_at(179.99, 10.0), cell_at(-179.99, 10.0)
    w = weights(table([west, east]), ERA5)
    cols = w.pixel % ERA5.width
    assert set(cols) <= {719, 720, 721} and np.allclose(sums(w), 1)


def test_table_order_kept():
    """Results follow the table's row order, whatever its index."""
    cells = table([cell_at(10.0, 10.0), cell_at(-60.0, -30.0)]).set_axis([7, 3])
    field = np.zeros((721, 1440))
    field[:360, :] = 1.0                                            # northern hemisphere = 1
    value, _ = cell_means(weights(cells, ERA5), field)
    assert value[0] == 1.0 and value[1] == 0.0


def test_land_weighting():
    """A cell half over a land pixel (value 0.3) and half over a sea pixel (stored 0): land-weighted mean is 0.3."""
    cell = across_lon(1.0)
    field = np.full((180, 360), 0.3)
    field[:, 0] = 0.0                                               # western pixel: sea, stored as 0
    land = np.ones((180, 360))
    land[:, 0] = 0.0
    w = weights(table([cell]), ONE)
    plain, _ = cell_means(w, field)
    value, valid = cell_means(w, field, land)
    east = (cell.lon + cell.dlon - 1.0) / cell.dlon
    assert np.isclose(plain[0], 0.3 * east) and np.isclose(value[0], 0.3) and np.isclose(valid[0], east)


def test_category_codes_take_the_dominant_pixel():
    """A cell mostly in a 'snow' (5) pixel and partly in a 'rain' (1) pixel is snow, not 3."""
    from majortom.climate.weights import cell_modes

    cell = across_lon(1.0)
    field = np.full((180, 360), 5.0)
    east = (cell.lon + cell.dlon - 1.0) / cell.dlon
    field[:, 0 if east > 0.5 else 1] = 1.0                          # the smaller part: rain
    value, valid = cell_modes(weights(table([cell]), ONE), field)
    assert value[0] == 5.0 and np.isclose(valid[0], 1)


def test_category_codes_lose_packing_noise():
    """ERA5's 16-bit packing decodes code 3 as 2.9999084: the cell gets exactly 3."""
    from majortom.climate.weights import cell_modes

    cell = across_lon(1.0)
    field = np.full((180, 360), 2.9999084)
    value, _ = cell_modes(weights(table([cell]), ONE), field)
    assert value[0] == 3.0
