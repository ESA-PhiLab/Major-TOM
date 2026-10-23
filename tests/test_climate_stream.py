"""majortom.climate.stream: the same tables as series(), each hour read once, restartable (offline, fake source)."""
import importlib

import numpy as np
import pandas as pd
import pytest

from majortom.cells import cell_at
from majortom.climate.series import series
from majortom.climate.sources import Source
from majortom.climate.stream import stream
from majortom.climate.weights import Grid

series_module = importlib.import_module("majortom.climate.series")
stream_module = importlib.import_module("majortom.climate.stream")
FAKE = Source("fake", "-", Grid(west=-180, north=90, step=1, width=360, height=180),
              {"t2m": "t", "tp": "p"}, {"t2m": "K", "tp": "m"}, "1h", accumulated=("tp",))


@pytest.fixture
def reads(monkeypatch):
    """The fake field: hour number + latitude row, so cells and hours are both visible; counts every read."""
    calls = []

    def fake(source, variable, time):
        calls.append((variable, pd.Timestamp(time)))
        hour = pd.Timestamp(time).value // 3_600_000_000_000
        return (hour + np.arange(180, dtype=float))[:, None].repeat(360, axis=1)

    for module in (series_module, stream_module):
        monkeypatch.setitem(module.SOURCES, "fake", FAKE)
        monkeypatch.setattr(module, "field", fake)
    return calls


def samples():
    places = [(-4.6, 56.1), (10.0, 45.0), (-70.0, -30.0)]
    times = ["2021-07-01T11:21:00Z", "2021-07-01T15:02:00Z", "2021-07-03T10:40:00Z", "2021-07-09T03:15:00Z"]
    rows = [(cell_at(*p), t) for p, t in zip(places * 2, times)]
    return pd.DataFrame({"grid_cell": [c.name for c, _ in rows], "grid_row_u": [c.row for c, _ in rows],
                         "grid_col_r": [c.col for c, _ in rows], "timestamp": [t for _, t in rows]})


def read_back(out, kind):
    return pd.concat([pd.read_parquet(p) for p in sorted((out / kind).glob("year=*/*.parquet"))]) \
        .sort_values(["acquired", "grid_cell", f"step_{kind[0]}"]).reset_index(drop=True)


def test_same_tables_as_series(reads, tmp_path):
    hourly, daily = series(samples(), "fake", hourly=3, daily=2)
    stream(samples(), tmp_path, "fake", hourly=3, daily=2)
    for kind, expected in (("hourly", hourly), ("daily", daily)):
        expected = expected.sort_values(["acquired", "grid_cell", f"step_{kind[0]}"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(read_back(tmp_path, kind), expected, check_dtype=False)


def test_each_hour_read_once(reads, tmp_path):
    stream(samples(), tmp_path, "fake", hourly=3, daily=2)
    assert len(reads) == len(set(reads))


def test_restart_skips_finished_days(reads, tmp_path):
    stream(samples(), tmp_path, "fake", hourly=3, daily=2)
    first_run = read_back(tmp_path, "daily")
    last_day = sorted((tmp_path / "daily").glob("year=*/*.parquet"))[-1]
    last_day.unlink()                                               # as if the run stopped before the last day
    (tmp_path / "hourly" / last_day.parent.name / last_day.name).unlink()
    reads.clear()
    stream(samples(), tmp_path, "fake", hourly=3, daily=2)
    pd.testing.assert_frame_equal(read_back(tmp_path, "daily"), first_run)
    assert len(reads) < 24 * 2 * 9                                  # only the last day's window was read again
