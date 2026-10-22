"""Earth Engine images as items, and its date filter (offline: a made-up image, no Earth Engine call)."""
import pytest

pytest.importorskip("pystac_client")
from majortom.build.backends import earthengine                  # noqa: E402
from majortom.build.collections import COLLECTIONS               # noqa: E402

LANDSAT_5 = {
    "id": "LANDSAT/LT05/C02/T1_L2/LT05_038032_19850831",
    "bands": [{"id": b, "crs": "EPSG:32612", "crs_transform": [30, 0, 299985, 0, -30, 4583415]}
              for b in ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL")],
    "properties": {"system:time_start": 494348400000, "CLOUD_COVER": 3.0, "SPACECRAFT_ID": "LANDSAT_5",
                   "WRS_PATH": 38, "WRS_ROW": 32, "majortom:collection": "LANDSAT/LT05/C02/T1_L2",
                   "system:footprint": {"type": "LinearRing",
                                        "coordinates": [[-112, 40], [-110, 40], [-110, 42], [-112, 42], [-112, 40]]}},
}


def test_image_becomes_an_item_with_our_band_names():
    item = earthengine.to_item(LANDSAT_5, "LANDSAT/LT05/C02/T1_L2")
    assert item.properties["platform"] == "landsat-5" and item.properties["grid:code"] == "WRS2-038032"
    assert item.properties["eo:cloud_cover"] == 3.0 and item.properties["proj:code"] == "EPSG:32612"
    assert item.assets["nir08"].href == "ee://LANDSAT/LT05/C02/T1_L2/LT05_038032_19850831#SR_B4"   # TM band 4
    assert "coastal" not in item.assets                                                               # TM has none
    assert item.assets["red"].extra_fields["raster:bands"] == [{"scale": 2.75e-05, "offset": -0.2}]
    crs, transform = earthengine.header(None, item, "red")
    assert crs.to_epsg() == 32612 and (transform.a, transform.c) == (30, 299985)


def test_sentinel_2_on_earth_engine_has_no_offset():
    from majortom.build.collections import declared_or_baseline_offset

    image = {"id": "COPERNICUS/S2_SR_HARMONIZED/20240829T180911_20240829T181118_T12TVK",
             "bands": [{"id": "B4", "crs": "EPSG:32612", "crs_transform": [10, 0, 399960, 0, -10, 4500000]}],
             "properties": {"system:time_start": 1724955785355, "SPACECRAFT_NAME": "Sentinel-2A", "MGRS_TILE": "12TVK",
                            "PROCESSING_BASELINE": "05.11", "BOA_ADD_OFFSET_B4": -1000}}
    item = earthengine.to_item(image, "COPERNICUS/S2_SR_HARMONIZED")
    assert item.properties["platform"] == "sentinel-2a" and item.properties["grid:code"] == "MGRS-12TVK"
    assert declared_or_baseline_offset(item, "B04") == 0                      # harmonized, not baseline 05.11


def test_dates_include_the_whole_last_day():
    assert earthengine._dates("2024-07-01/2024-08-31") == ("2024-07-01T00:00:00", "2024-09-01T00:00:00")
    start, end = earthengine._dates("2024-08-29T18:08:11+00:00/2024-08-29T18:10:11+00:00")
    assert end.startswith("2024-08-29T18:10:11.001")


def test_earth_engine_sources_name_known_collections():
    from majortom.build.collections import EARTH_ENGINE

    for name, product in COLLECTIONS.items():
        source = product.source(name, "earth-engine")
        if source:
            assert all(cid in EARTH_ENGINE for cid in source.collection.split("+")), name
