"""majortom.climate.days: source days chosen per step, cached means, restart (offline, fake daily source).

The fake source: day d has value d (days since 1970) plus the latitude row, with UTC-offset groups like ERA5-Land
(group = 3-hour steps of 12 - UTC hour; a group-g day starts at 00:00 - g h).
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from majortom.cells import cell_at
from majortom.climate.sources import Source
from majortom.climate.weights import Grid

days_module = importlib.import_module("majortom.climate.days")
H = np.timedelta64(1, "h")


def groups(src, acquired, lon):
    hour = (acquired - acquired.astype("datetime64[D]")) / H
    return (3 * np.round((12 - hour) / 3)).astype(int)


FAKE_READER = SimpleNamespace(
    groups=groups,
    days=lambda src, centres, lon, group: (centres + group * H).astype("datetime64[D]").astype(np.int64),
    period_start=lambda src, days, group: days.astype("datetime64[D]") - group * H,
)
FAKE = Source("fake", "-", Grid(west=-180, north=90, step=1, width=360, height=180), {"a": "a", "b": "b"},
              {"a": "1", "b": "1"}, "1D", period="day", options={"suffix": "_mean"})


@pytest.fixture
def reads(monkeypatch, tmp_path):
    calls = []

    def fake_field(src, variable, key):
        calls.append((variable, key))
        day, group = key
        return (day + np.arange(180, dtype=float))[:, None].repeat(360, axis=1) + (variable == "b") * 0.5

    monkeypatch.setitem(days_module.SOURCES, "fake", FAKE)
    monkeypatch.setattr(days_module, "reader", lambda src: FAKE_READER)
    monkeypatch.setattr(days_module, "field", fake_field)
    monkeypatch.setenv("MAJORTOM_CLIMATE_CACHE", str(tmp_path / "cache"))
    return calls


def samples():
    rows = [(cell_at(-4.6, 56.1), "2021-07-01T11:21:00Z"), (cell_at(150.0, -33.0), "2021-07-01T23:50:00Z"),
            (cell_at(-4.6, 56.1), "2021-07-03T10:40:00Z")]
    return pd.DataFrame({"grid_cell": [c.name for c, _ in rows], "grid_row_u": [c.row for c, _ in rows],
                         "grid_col_r": [c.col for c, _ in rows], "timestamp": [t for _, t in rows]})


def table(out):
    return pd.concat([pd.read_parquet(p) for p in sorted((out / "daily").glob("year=*/*.parquet"))]) \
        .sort_values(["acquired", "step_d"]).reset_index(drop=True)


def test_days_centred_by_group(reads, tmp_path):
    days_module.stream_days(samples(), tmp_path / "out", "fake", daily=2)
    t = table(tmp_path / "out")
    assert len(t) == 3 * 5 and t.notna().all().all()
    first = t[t.acquired == pd.Timestamp("2021-07-01T11:21")].set_index("step_d")
    day0 = np.datetime64("2021-07-01").astype(np.int64)
    row = 180 - 1 - int(np.floor(56.1 + 90))                        # the 1° row holding the cell (north = row 0)
    assert first.loc[0, "fake_a_mean"] == day0 + row and first.loc[1, "fake_a_mean"] == day0 + 1 + row
    assert first.loc[0, "fake_b_mean"] == day0 + row + 0.5
    # 11:21 UTC: offset 12 - 11.35 rounds to group 0, so step 0 is the UTC day from 00:00
    assert first.loc[0, "fake_start"] == pd.Timestamp("2021-07-01T00:00")
    late = t[t.acquired == pd.Timestamp("2021-07-01T23:50")].set_index("step_d")
    # 23:50 UTC: 12 - 23.83 = -11.8 -> group -12, so step 0 is the day starting 2021-07-01 12:00 UTC
    assert late.loc[0, "fake_start"] == pd.Timestamp("2021-07-01T12:00")


def test_each_source_day_read_once_and_restart(reads, tmp_path):
    days_module.stream_days(samples(), tmp_path / "out", "fake", daily=2)
    assert len(reads) == len(set(reads))
    first = table(tmp_path / "out")
    files = sorted((tmp_path / "out" / "daily").glob("year=*/*.parquet"))
    files[-1].unlink()
    reads.clear()
    days_module.stream_days(samples(), tmp_path / "out", "fake", daily=2)
    assert reads == []                                              # cached means reused, nothing read again
    pd.testing.assert_frame_equal(table(tmp_path / "out"), first)


def test_valid_per_variable(reads, monkeypatch, tmp_path):
    """Sentinel-5P-like: variable b has no data, so b_valid is 0 while a_valid is 1."""
    per = Source("fake", "-", FAKE.grid, {"a": "a", "b": "b"}, {"a": "1", "b": "1"}, "1D", period="day",
                 options={"valid_per_variable": True})
    monkeypatch.setitem(days_module.SOURCES, "fake", per)
    real = days_module.field
    monkeypatch.setattr(days_module, "field", lambda src, v, key: real(src, v, key) * (np.nan if v == "b" else 1))
    days_module.stream_days(samples(), tmp_path / "out", "fake", daily=1)
    t = table(tmp_path / "out")
    assert (t.fake_a_valid == 1).all() and (t.fake_b_valid == 0).all() and "fake_valid" not in t


def test_budget_then_continue(reads, tmp_path):
    """A field budget stops early and writes only complete days; the next run finishes the rest."""
    days_module.stream_days(samples(), tmp_path / "out", "fake", daily=2, max_fields=6)
    assert len(reads) == 6
    partial = sorted((tmp_path / "out" / "daily").glob("year=*/*.parquet"))
    days_module.stream_days(samples(), tmp_path / "out", "fake", daily=2)
    assert len(reads) == len(set(reads))
    assert len(sorted((tmp_path / "out" / "daily").glob("year=*/*.parquet"))) > len(partial)
