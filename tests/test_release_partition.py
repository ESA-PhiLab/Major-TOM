"""majortom.release.partition: Hilbert order and size-targeted partitions (offline, numpy and pandas only)."""
import numpy as np
import pandas as pd

from majortom.cells import Cell, cell_at
from majortom.release.partition import hilbert, partition


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


def test_neighbours_are_close_on_the_curve():
    near = hilbert(np.array([2.0, 2.1]), np.array([48.6, 48.6]))
    far = hilbert(np.array([2.0, 140.0]), np.array([48.6, -30.0]))
    assert abs(near[1] - near[0]) < abs(far[1] - far[0])


def _samples(cells, dates=1, size=10):
    rows = [dict(grid_cell=c.name, acquired=pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * k), bytes=size)
            for c in cells for k in range(dates)]
    return pd.DataFrame(rows)


def test_sizes_cells_and_determinism():
    cells = [Cell(r, c) for r in range(400, 420) for c in range(-10, 10)]           # 400 cells near 36°N
    samples = _samples(cells, dates=3, size=7)                                      # 21 bytes per cell
    ids, summary = partition(samples, max_bytes=500)
    assert (summary["bytes"] <= 500).all() and (summary["bytes"].iloc[:-1] > 500 - 21).all()
    assert samples.assign(p=ids).groupby("grid_cell")["p"].nunique().max() == 1   # a cell is never split
    assert ids.equals(partition(samples, max_bytes=500)[0])
    assert summary["samples"].sum() == len(samples)


def test_regional_partitions_are_compact():
    """Partitions of a dense block cover small areas, much smaller than the block."""
    cells = [Cell(r, c) for r in range(400, 460) for c in range(-30, 30)]           # 3,600 cells
    _, summary = partition(_samples(cells), max_bytes=10 * 100)                     # 100 cells per partition
    span = (summary.lon_max - summary.lon_min) + (summary.lat_max - summary.lat_min)
    block = 60 * Cell(430, 0).dlon + 60 * Cell(430, 0).dlat
    assert span.median() < block / 3


def test_a_cell_larger_than_a_partition_is_cut_by_time():
    big = cell_at(10.0, 45.0)
    samples = pd.concat([_samples([big], dates=50, size=10), _samples([cell_at(10.2, 45.0)], size=10)],
                        ignore_index=True)
    ids, summary = partition(samples, max_bytes=100)
    parts = samples.assign(p=ids)[lambda t: t.grid_cell == big.name].groupby("p")["acquired"].agg(["min", "max"])
    assert len(parts) == 5 and (summary["bytes"] <= 100).all()
    assert (parts["min"].iloc[1:].to_numpy() > parts["max"].iloc[:-1].to_numpy()).all()   # consecutive periods
