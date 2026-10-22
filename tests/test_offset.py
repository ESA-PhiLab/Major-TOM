"""Sentinel-2 offset rule and band names per archive (offline)."""
import datetime as dt

import pytest

pytest.importorskip("pystac_client")
import pystac                                                    # noqa: E402
from majortom.build.collections import COLLECTIONS, declared_or_baseline_offset   # noqa: E402


def item(baseline=None, raster_bands=None) -> pystac.Item:
    """A one-asset item with an optional processing baseline and declared raster:bands."""
    it = pystac.Item("x", None, None, dt.datetime(2021, 7, 14, tzinfo=dt.timezone.utc),
                     {} if baseline is None else {"s2:processing_baseline": baseline})
    extra = {} if raster_bands is None else {"raster:bands": raster_bands}
    it.add_asset("red", pystac.Asset("https://example.org/red.tif", extra_fields=extra))
    return it


def test_offset_from_the_baseline_when_not_declared():           # Planetary Computer
    assert declared_or_baseline_offset(item("03.00"), "red") == 0
    assert declared_or_baseline_offset(item("04.00"), "red") == 1000
    assert declared_or_baseline_offset(item("05.10"), "red") == 1000


def test_declared_offset_wins_over_the_baseline():               # Earth Search Collection 1
    declared = [{"scale": 0.0001, "offset": -0.1}]
    assert declared_or_baseline_offset(item("03.00", declared), "red") == 1000
    assert declared_or_baseline_offset(item(None, [{"scale": 0.0001, "offset": 0}]), "red") == 0


def test_no_offset_known():
    assert declared_or_baseline_offset(item(), "red") is None


def test_band_names_are_the_same_from_every_archive():
    s2 = COLLECTIONS["sentinel-2-l2a"]
    assert s2.source("sentinel-2-l2a", "planetary-computer").key("B04") == "B04"
    assert s2.source("sentinel-2-l2a", "earth-search").key("B04") == "red"
    assert s2.source("sentinel-2-l2a", "earth-search").key("SCL") == "scl"
    assert COLLECTIONS["landsat-c2-l2"].source("landsat-c2-l2", "earth-search") is None   # requester-pays there
    assert COLLECTIONS["cop-dem-glo-30"].source("cop-dem-glo-30", "planetary-computer").key("DEM") == "data"


def test_sentinel_1_rtc_is_on_planetary_computer_only():
    s1 = COLLECTIONS["sentinel-1-rtc"]
    assert s1.assets == ("vv", "vh") and s1.clouds is None
    assert s1.archives() == ["planetary-computer"]
