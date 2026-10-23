"""majortom.climate.readers: placing fields on the grid, choosing source days (offline)."""
import numpy as np
from affine import Affine

from majortom.climate.readers import earthengine, files, place, queue
from majortom.climate.sources import SOURCES
from majortom.climate.weights import Grid

H = np.timedelta64(1, "h")
GRID = Grid(west=-180, north=90, step=1, width=360, height=180)


def test_place_wraps_longitudes_and_flips_south_up():
    data = np.arange(6, dtype=float).reshape(2, 3)                  # 2 rows x 3 columns
    east = place(data, Affine(1, 0, 178, 0, -1, 10), GRID)          # columns at 178, 179, 180 (= -180)
    assert east[80, 358] == 0 and east[80, 359] == 1 and east[80, 0] == 2
    south_up = place(data[::-1], Affine(1, 0, 0, 0, 1, 8), GRID)    # rows stored from 8°N upwards
    assert south_up[80, 180] == 0 and south_up[81, 180] == 3
    assert np.isnan(east[0, 0])


def test_era5_land_groups_centre_days():
    """12 - UTC hour in 3-hour steps; the group's day centred on the acquisition within 1.5 h."""
    src = SOURCES["era5-land"]
    acquired = np.array(["2021-07-01T11:21", "2021-07-01T23:50", "2021-07-01T02:00", "2021-07-01T05:40"],
                        dtype="datetime64[s]")
    group = queue.groups(src, acquired, np.zeros(4), np.zeros(4))
    assert list(group) == [0, -12, 9, 6]
    day = queue.days(src, acquired[:, None], np.zeros((4, 1)), group[:, None])[:, 0]
    start = queue.period_start(src, day, group)
    centre = start + np.timedelta64(12, "h")
    assert (np.abs((centre - acquired) / H) <= 1.5 + 1e-9).all()


def test_cpc_days_follow_the_end_of_day_map(monkeypatch):
    """A box whose day ends at 12Z: day D runs from 12Z on D-1, so a window centred at 15Z belongs to D+1."""
    src = SOURCES["cpc"]
    eod = np.full((360, 720), 24.0, np.float32)
    eod[int((39.7 + 90) // 0.5), int((-104.9 % 360) // 0.5)] = 12.0          # Denver's box
    monkeypatch.setattr(files, "end_of_day", lambda source: eod)
    group = files.groups(src, np.zeros(2, "datetime64[s]"), np.array([-104.9, 0.0]), np.array([39.7, 0.0]))
    assert list(group) == [-12, 0]
    centres = np.array([["2021-07-01T15:00"], ["2021-07-01T15:00"]], dtype="datetime64[s]")
    day = files.days(src, centres, None, group[:, None])[:, 0]
    assert list(day.astype("datetime64[D]").astype(str)) == ["2021-07-02", "2021-07-01"]
    start = files.period_start(src, day, group)
    assert list(start.astype(str)) == ["2021-07-01T12", "2021-07-01T00"]


def test_s5p_overpass_day():
    """Overpass ~13:30 local: near 150°E (UTC+10) an evening-UTC centre belongs to the next UTC day's mosaic
    only if the local day has moved on; at 0° the UTC day is the local day."""
    src = SOURCES["s5p"]
    centres = np.array([["2021-07-01T11:00"], ["2021-07-01T15:00"]], dtype="datetime64[s]")
    lon = np.array([[0.0], [150.0]])
    day = earthengine.days(src, centres, lon, np.zeros((2, 1), int))[:, 0].astype("datetime64[D]").astype(str)
    # 0°: local 11:00 on 1 July, overpass 13:30 UTC on 1 July; 150°E: local 01:00 on 2 July, overpass 03:30 UTC 2 July
    assert list(day) == ["2021-07-01", "2021-07-02"]
