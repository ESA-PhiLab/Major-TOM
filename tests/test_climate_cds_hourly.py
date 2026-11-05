"""majortom.climate.readers.cds_hourly: ERA5-Land hourly south of 57.1°S from the Climate Data Store (offline).

Requests and file reads are replaced: a span "file" is an empty marker, and a field's value is the hours since
00 UTC (an accumulated total restarting at 01 UTC), so the hourly amount is 1 at every hour.
"""
import numpy as np
import pandas as pd
import pytest

from majortom.climate.readers import cds_hourly
from majortom.climate.sources import ERA5_LAND_GROUPS, SOURCES

SRC = SOURCES["era5-land-cds-south"]


def test_spans_are_weeks_cut_at_month_ends():
    day = pd.Timestamp
    assert cds_hourly.span(day("2017-01-25")) == (day("2017-01-23"), day("2017-01-29"))     # Monday-Sunday
    assert cds_hourly.span(day("2017-01-31")) == (day("2017-01-30"), day("2017-01-31"))     # week runs into February
    assert cds_hourly.span(day("2017-02-01")) == (day("2017-02-01"), day("2017-02-05"))


def test_requests_stay_within_the_accepted_size():
    first, last = pd.Timestamp("2017-01-23"), pd.Timestamp("2017-01-29")
    for group, names in ERA5_LAND_GROUPS.items():
        r = cds_hourly.request(SRC, first, last, group)
        assert len(r["variable"]) * len(r["day"]) * len(r["time"]) <= 1848                     # 8,184 was refused
        assert r["area"] == [-56.5, -180.0, -90.0, 180.0] and r["month"] == "01"
    assert sorted(v for names in ERA5_LAND_GROUPS.values() for v in names) == sorted(SRC.variables)


def test_grid_covers_the_gap_and_cells_that_straddle_it():
    g = SRC.grid
    assert g.north == -56.45 and g.north - g.height * g.step == pytest.approx(-90.05) and g.width * g.step == 360


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setenv("MAJORTOM_CLIMATE_CACHE", str(tmp_path))

    def fetch(source, request, target):
        fetched.append((request["year"], request["month"], request["day"][0], len(request["day"])))
        target.write_bytes(b"")
        return target

    def read(path, variable, when, grid):
        hour = pd.Timestamp(when).hour
        return np.full((grid.height, grid.width), 24.0 if hour == 0 else float(hour), np.float32)

    monkeypatch.setattr(cds_hourly, "_fetch", fetch)
    monkeypatch.setattr(cds_hourly, "_mine", set())
    monkeypatch.setattr(cds_hourly, "read_netcdf", read)
    return tmp_path, fetched


def test_accumulated_totals_become_hourly_amounts(fake_store):
    for hour in ["2017-01-25T00", "2017-01-25T01", "2017-01-25T05", "2017-01-30T00"]:     # 30th: across two spans
        assert np.all(cds_hourly.field(SRC, "tp", pd.Timestamp(hour)) == 1.0)
    assert np.all(cds_hourly.field(SRC, "t2m", pd.Timestamp("2017-01-25T05")) == 5.0)       # a state: as stored


def test_older_spans_are_removed_as_the_run_moves_on(fake_store):
    root, fetched = fake_store
    for day in ["2017-01-10", "2017-01-17", "2017-01-24"]:
        cds_hourly.field(SRC, "t2m", pd.Timestamp(day + "T12"))
    left = sorted(p.name[:13] for p in (root / "era5land" / "hourly").glob("*.nc"))
    assert left == ["2017-01-16_7d", "2017-01-23_7d"] and len(fetched) == 3               # the 9th-15th removed


def test_spans_of_other_jobs_are_left_alone(fake_store):
    root, fetched = fake_store
    other = root / "era5land" / "hourly" / "2016-12-05_7d_g1_-56.5_-180_-90_180.nc"           # e.g. another year's job
    other.parent.mkdir(parents=True)
    other.write_bytes(b"")
    for day in ["2017-01-10", "2017-01-17", "2017-01-24"]:
        cds_hourly.field(SRC, "t2m", pd.Timestamp(day + "T12"))
    left = sorted(p.name[:13] for p in other.parent.glob("*.nc"))
    assert left == ["2016-12-05_7d", "2017-01-16_7d", "2017-01-23_7d"]                     # only its own removed
