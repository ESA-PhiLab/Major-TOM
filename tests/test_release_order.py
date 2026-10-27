"""majortom.release.order: the Hilbert writing order (offline, numpy and pandas only)."""
import numpy as np
import pandas as pd

from majortom.cells import Cell
from majortom.release.order import hilbert, hilbert_order


def test_hilbert_is_a_continuous_curve():
    """Every grid square is visited once, and consecutive positions are neighbouring squares."""
    order = 4
    n = 1 << order
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    lon = (x.ravel() + 0.5) / n * 360 - 180
    lat = (y.ravel() + 0.5) / n * 180 - 90
    d = hilbert(lon, lat, order)
    assert sorted(d) == list(range(n * n))
    path = np.argsort(d)
    steps = np.abs(np.diff(x.ravel()[path])) + np.abs(np.diff(y.ravel()[path]))
    assert (steps == 1).all()


def _samples(cells, dates):
    return pd.DataFrame([dict(grid_cell=c.name, acquired=pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * k))
                         for c in cells for k in range(dates)]).sample(frac=1, random_state=0)   # shuffled


def test_cells_contiguous_and_dates_in_time_order():
    cells = [Cell(r, c) for r in range(400, 410) for c in range(-5, 5)]
    s = _samples(cells, dates=3)
    ordered = s.loc[hilbert_order(s)]
    runs = (ordered["grid_cell"] != ordered["grid_cell"].shift()).sum()
    assert runs == len(cells)                                                     # each cell is one run
    assert ordered.groupby("grid_cell", sort=False)["acquired"].apply(lambda t: t.is_monotonic_increasing).all()
    assert list(hilbert_order(s)) == list(hilbert_order(s))                        # deterministic


def test_runs_of_the_order_are_compact():
    """Any stretch of 100 consecutive cells covers a small area compared with the 60 x 60 block they come from."""
    cells = [Cell(r, c) for r in range(400, 460) for c in range(-30, 30)]
    ordered = _samples(cells, dates=1).loc[lambda t: hilbert_order(t)]
    centre = np.array([Cell(*[int(v[:-1]) * (1 if v[-1] in "UR" else -1) for v in name.split("_")]).centroid
                       for name in ordered["grid_cell"]])
    spans = [np.ptp(centre[i:i + 100], axis=0).sum() for i in range(0, len(centre) - 100, 100)]
    block = np.ptp(centre, axis=0).sum()
    assert np.median(spans) < block / 3
