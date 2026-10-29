"""Earth Data Hub reader: hours read from block files equal the data written (no network, no mappings kept)."""
from types import SimpleNamespace

import numpy as np

from majortom.climate.readers import edh
from majortom.climate.readers.edh import T, Y, X, _chunks, _drop, _hour, _tiled_hour, _write_tile


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


HEIGHT, WIDTH = 120, 250                 # 3 x 3 chunks; the last row and column of chunks are partial (20, 50)


def _dense(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(T, HEIGHT, WIDTH)).astype(np.float32)


def test_tiled_hour_matches_dense(tmp_path):
    dense = _dense()
    path = tmp_path / "0000.t32"
    written = [(0, 0), (1, 2), (2, 1), (2, 2)]                  # includes both partial edges
    filled = np.zeros((HEIGHT, WIDTH), bool)
    with open(path, "wb") as f:
        for cy, cx in written:
            sl = (slice(cy * Y, (cy + 1) * Y), slice(cx * X, (cx + 1) * X))
            _write_tile(f.fileno(), cy, cx, WIDTH, dense[(slice(None), *sl)])
            filled[sl] = True
    assert _chunks(filled) == sorted(written)
    for i in (0, 700, T - 1):
        got = _tiled_hour(path, i, _chunks(filled), HEIGHT, WIDTH)
        np.testing.assert_array_equal(got[filled], dense[i][filled])
        assert (got[~filled] == 0).all()


def test_fetch_then_field_from_fake_store(tmp_path, monkeypatch):
    dense = _dense(2)
    dense[:, 0:Y, X:2 * X] = np.nan                             # chunk (0, 1) is open sea
    source = SimpleNamespace(name="fake", grid=SimpleNamespace(height=HEIGHT, width=WIDTH), accumulated=())
    monkeypatch.setattr(edh, "cache_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(edh, "_open", lambda: {"t2m": dense})  # _read slices it like the zarr array
    monkeypatch.setitem(edh._needed, "fake", {(0, 0), (0, 1), (2, 2)})
    edh._blocks.clear()
    for hour in ("1950-01-01T00", "1950-01-20T13", "1950-03-01T23"):
        i = int((np.datetime64(hour, "h") - edh.START) / np.timedelta64(1, "h"))
        got = edh.field(source, "t2m", hour)
        for cy, cx in ((0, 0), (2, 2)):
            sl = (slice(cy * Y, (cy + 1) * Y), slice(cx * X, (cx + 1) * X))
            np.testing.assert_array_equal(got[sl], dense[i][sl])
        assert np.isnan(got[0:Y, X:2 * X]).all() and np.isnan(got[Y:2 * Y]).all()
    assert (tmp_path / "fake" / "blocks" / "t2m" / "0000.t32.filled.npy").exists()
    assert (0, 1) not in edh._needed["fake"]                    # the sea chunk is never requested again
    edh._blocks.clear()


class _Flaky:
    """A chunk store that fails with `errors` in turn, then returns a chunk."""

    def __init__(self, *errors):
        self.errors = list(errors)

    def __getitem__(self, sel):
        if self.errors:
            raise self.errors.pop(0)
        return np.ones((2, 2), np.float32)


def test_cut_download_is_retried_even_if_its_byte_count_contains_429(monkeypatch):
    import pytest
    monkeypatch.setattr(edh.time, "sleep", lambda s: None)
    cut = OSError("Response payload is not completed: <ContentLengthError: 400, message='Not enough data to "
                  "satisfy content length header (received 93429 of 3242240 bytes).'>")
    assert edh._read(_Flaky(cut, cut), ()).sum() == 4                     # retried twice, then read
    with pytest.raises(OSError, match="429"):
        edh._read(_Flaky(OSError("429, message='Too Many Requests'")), ())   # quota: stop at once
