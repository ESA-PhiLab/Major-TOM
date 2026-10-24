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


def test_acquisitions_hours_apart_in_one_day(reads, tmp_path):
    """Two cells acquired 23 h apart on one day: each gets its own cell's values, none from a neighbour, no gaps."""
    a, b = cell_at(-4.6, 56.1), cell_at(150.0, -33.0)
    s = pd.DataFrame({"grid_cell": [a.name, b.name], "grid_row_u": [a.row, b.row], "grid_col_r": [a.col, b.col],
                      "timestamp": ["2021-07-01T00:10:00Z", "2021-07-01T23:20:00Z"]})
    hourly, daily = series(s, "fake", hourly=3, daily=2)
    stream(s, tmp_path, "fake", hourly=3, daily=2)
    for kind, expected in (("hourly", hourly), ("daily", daily)):
        got = read_back(tmp_path, kind)
        assert got.notna().all().all()
        expected = expected.sort_values(["acquired", "grid_cell", f"step_{kind[0]}"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_command_line(reads, tmp_path):
    """Two metadata files sharing an acquisition: one table row per acquisition, both samples in the link table."""
    from majortom.climate.__main__ import main

    l2a, l1c = tmp_path / "Core-S2L2A" / "metadata.parquet", tmp_path / "Core-S2L1C" / "metadata.parquet"
    for path, product in ((l2a, "L2A"), (l1c, "L1C")):
        path.parent.mkdir()
        samples().head(2).assign(product_id=product).to_parquet(path)
    assert main(["stream", "--samples", str(l2a), str(l1c), "--source", "fake", "--out", str(tmp_path / "out"),
                 "--hourly", "3", "--daily", "2"]) == 0
    links = pd.read_parquet(tmp_path / "out" / "acquisitions.parquet")
    assert len(links) == 4 and set(links.dataset) == {"Core-S2L2A", "Core-S2L1C"}
    assert len(read_back(tmp_path / "out", "daily")) == 2 * 5               # 2 acquisitions x 5 daily steps


def test_hourly_window_wider_than_daily(reads, tmp_path):
    """--daily 0 covers only +-12 h; the +-24 h hourly window must still be read."""
    hourly, daily = series(samples(), "fake", hourly=24, daily=0)
    stream(samples(), tmp_path, "fake", hourly=24, daily=0)
    expected = hourly.sort_values(["acquired", "grid_cell", "step_h"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(read_back(tmp_path, "hourly"), expected, check_dtype=False)
