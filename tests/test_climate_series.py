"""majortom.climate.series: step alignment and daily statistics, on a fake source (offline).

The fake field's value everywhere is the hour's own number (hours since 1970), so every table value
says exactly which hours it was made from.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from majortom.climate.series import acquisitions, series
from majortom.climate.sources import Source
from majortom.climate.weights import Grid

series_module = importlib.import_module("majortom.climate.series")   # the module, not the function

FAKE = Source("fake", "-", Grid(west=-180, north=90, step=10, width=36, height=18),
              {"t2m": "t", "tp": "p", "sp": "s"}, {"t2m": "K", "tp": "m", "sp": "Pa"}, "1h", accumulated=("tp",))


def hour_number(t) -> float:
    return float(pd.Timestamp(t).value // 3_600_000_000_000)


@pytest.fixture(autouse=True)
def fake_source(monkeypatch):
    monkeypatch.setitem(series_module.SOURCES, "fake", FAKE)
    monkeypatch.setattr(series_module, "field", lambda source, variable, time: np.full((18, 36), hour_number(time)))


def samples(*times):
    return pd.DataFrame({"grid_cell": ["624U_29L"] * len(times), "grid_row_u": 624, "grid_col_r": -29,
                         "timestamp": list(times)})


def test_hourly_step_zero_is_nearest_hour():
    hourly, _ = series(samples("2021-07-14T11:21:19Z", "2021-07-15T11:40:00Z"), "fake", hourly=2, daily=1)
    zero = hourly[hourly.step_h == 0]
    assert list(zero.fake_t2m) == [hour_number("2021-07-14T11:00"), hour_number("2021-07-15T12:00")]
    assert list(hourly.step_h[:5]) == [-2, -1, 0, 1, 2]


def test_daily_windows_centred_on_acquisition():
    """Step 0 holds the 24 stamped hours after acquired - 12 h: 00:00..23:00 for 11:21."""
    _, daily = series(samples("2021-07-14T11:21:19Z"), "fake", hourly=1, daily=1)
    first = hour_number("2021-07-14T00:00")
    by_step = daily.set_index("step_d")
    assert by_step.loc[0, "fake_t2m_mean"] == pytest.approx(first + 11.5)
    assert by_step.loc[0, "fake_t2m_min"] == first and by_step.loc[0, "fake_t2m_max"] == first + 23
    assert by_step.loc[-1, "fake_t2m_min"] == first - 24 and by_step.loc[1, "fake_t2m_max"] == first + 47


def test_accumulations_summed_states_averaged():
    _, daily = series(samples("2021-07-14T11:21:19Z"), "fake", hourly=1, daily=0)
    first = hour_number("2021-07-14T00:00")
    row = daily.iloc[0]
    assert row.fake_tp_sum == pytest.approx(24 * first + 276)               # 0 + 1 + ... + 23 = 276
    assert row.fake_sp_mean == pytest.approx(first + 11.5)
    assert "fake_tp_mean" not in daily and "fake_sp_min" not in daily      # min/max only for EXTREMES
    assert row.fake_valid == pytest.approx(1)


def test_one_row_per_acquisition():
    """The same acquisition in two datasets (L2A and L1C) gives one row per step, not two."""
    acq = acquisitions(samples("2021-07-14T11:21:19Z", "2021-07-14T11:21:19Z"))
    hourly, daily = series(samples("2021-07-14T11:21:19Z", "2021-07-14T11:21:19Z"), "fake", hourly=24, daily=30)
    assert len(acq) == 1 and len(hourly) == 49 and len(daily) == 61


def test_three_hourly_source(monkeypatch):
    """A 3-hourly source (like CAMS): off-step hours are empty and not read; daily means use the 8 steps."""
    three = Source("three", "-", FAKE.grid, {"t2m": "t"}, {"t2m": "K"}, "3h")
    monkeypatch.setitem(series_module.SOURCES, "three", three)
    read = []
    monkeypatch.setattr(series_module, "field",
                        lambda source, variable, time: read.append(time) or np.full((18, 36), hour_number(time)))
    hourly, daily = series(samples("2021-07-14T11:21:19Z"), "three", hourly=2, daily=0)
    assert all(pd.Timestamp(t).hour % 3 == 0 for t in read)
    by_step = hourly.set_index("step_h")["three_t2m"]
    assert np.isnan(by_step[-1]) and by_step[-2] == hour_number("2021-07-14T09:00")   # 10:00 off step, 09:00 on
    first = hour_number("2021-07-14T00:00")
    assert daily.three_t2m_mean.iloc[0] == pytest.approx(first + np.mean([0, 3, 6, 9, 12, 15, 18, 21]))
