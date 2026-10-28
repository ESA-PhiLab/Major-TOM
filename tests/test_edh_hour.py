"""Earth Data Hub reader: one hour read straight from a block file equals the mapped block's hour (no network)."""
import numpy as np

from majortom.climate.readers.edh import _drop, _hour


def test_hour_matches_memmap(tmp_path):
    hours, height, width = 24, 7, 11
    path = tmp_path / "0000.f32"
    data = np.memmap(path, np.float32, "w+", shape=(hours, height, width))
    data[:] = np.random.default_rng(0).normal(size=(hours, height, width)).astype(np.float32)
    data[5, 2:4, 3:6] = np.nan                                  # never-written and empty areas read back as written
    data.flush()
    expected = np.array(data)
    del data
    _drop(path)
    for i in (0, 5, hours - 1):
        np.testing.assert_array_equal(_hour(path, i, height, width), expected[i])


def test_sparse_hours_read_as_zero(tmp_path):
    path = tmp_path / "0001.f32"
    data = np.memmap(path, np.float32, "w+", shape=(4, 3, 3))
    data[1] = 7.0                                               # only hour 1 written; the rest stay sparse zeros
    data.flush()
    del data
    assert (_hour(path, 1, 3, 3) == 7.0).all()
    assert (_hour(path, 3, 3, 3) == 0.0).all()
