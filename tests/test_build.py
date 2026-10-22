"""Scene choice rules of majortom.build (offline)."""
import pandas as pd
import pytest

pytest.importorskip("pystac_client")
from majortom.build import Choice, interval                      # noqa: E402
from majortom.build.collections import COLLECTIONS               # noqa: E402
from majortom.build.search import middle                         # noqa: E402


def test_cloud_range_accepts_clear_or_cloudy_scenes():
    clear, cloudy, any_cloud = Choice(), Choice(cloud=(0.3, 1.0)), Choice(cloud=None)
    assert clear.accepts(0.02, 0) and not clear.accepts(0.4, 0)
    assert cloudy.accepts(0.4, 0) and not cloudy.accepts(0.02, 0)
    assert any_cloud.accepts(0.9, 0)
    assert not clear.accepts(0.0, 0.5)                  # too much no-data, whatever the clouds


def test_cloud_gap_ranks_the_closest_scene_to_the_range():
    cloudy = Choice(cloud=(0.3, 1.0))
    assert cloudy.cloud_gap(0.25) < cloudy.cloud_gap(0.05)


def test_dates_for_a_sample():
    assert interval("2024-07-01", 10) == "2024-06-21T00:00:00/2024-07-11T00:00:00"
    assert interval("2024-06-01/2024-06-30", None) == "2024-06-01/2024-06-30"
    assert middle("2024-06-01/2024-06-03") == pd.Timestamp("2024-06-02", tz="UTC")


def test_every_collection_has_a_time_kind_and_known_resampling():
    for name, collection in COLLECTIONS.items():
        assert collection.time in ("scene", "yearly", "static"), name
        assert collection.resampling in ("nearest", "bilinear"), name


def test_tiles_of_one_pass_are_grouped_even_seconds_apart():
    import datetime as dt
    import pystac
    from majortom.build.search import passes

    def tile(name, platform, seconds):
        when = dt.datetime(2024, 7, 15, 10, 41, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=seconds)
        return pystac.Item(name, None, None, when, {"platform": platform})

    items = [tile("a", "sentinel-2a", 52), tile("b", "sentinel-2b", 50), tile("c", "sentinel-2a", 32),
             tile("d", "sentinel-2a", 3600)]
    assert [[t.id for t in g] for g in passes(items)] == [["a", "c"], ["b"], ["d"]]


def test_saved_sample_is_reused_only_for_the_same_request(tmp_path):
    import json
    from dataclasses import asdict
    from dataclasses import replace
    from majortom.build import ARCHIVES, _same_request
    from majortom.cells import Cell

    pc = ARCHIVES["planetary-computer"]
    cell, choice = Cell(450, -910, 10.0), Choice()
    sidecar = tmp_path / "000000.json"
    sidecar.write_text(json.dumps(dict(grid_cell=cell.name, grid_km=cell.d, choice=asdict(choice),
                                       query=dict(datetime="2024-01-01/2024-12-31", margin=0.056))))
    assert _same_request(sidecar, pc, cell, "2024-01-01/2024-12-31", 0.056, choice)   # no archive field: PC
    assert not _same_request(sidecar, replace(pc, name="other"), cell, "2024-01-01/2024-12-31", 0.056, choice)
    assert not _same_request(sidecar, pc, Cell(450, -909, 10.0), "2024-01-01/2024-12-31", 0.056, choice)
    assert not _same_request(sidecar, pc, cell, "2020-07-02/2020-08-31", 0.056, choice)
    assert not _same_request(sidecar, pc, cell, "2024-01-01/2024-12-31", 0.056, Choice(cloud=(0.3, 1.0)))
    assert not _same_request(tmp_path / "missing.json", pc, cell, None, 0.056, choice)


def test_planetary_computer_is_the_default_archive():
    import inspect
    from majortom.build import ARCHIVES, download

    assert inspect.signature(download).parameters["archive"].default == "planetary-computer"
    assert ARCHIVES["planetary-computer"].stac_url.startswith("https://planetarycomputer.microsoft.com")


def test_unknown_archive_is_refused():
    from majortom.build import download

    with pytest.raises(KeyError, match="unknown archive"):
        download(pd.DataFrame(), "sentinel-2-l2a", archive="nowhere")


def test_l1c_borrows_clouds_from_the_same_pass_of_l2a(monkeypatch):
    import datetime as dt
    import pystac
    from majortom.build import ARCHIVES, scene_cloud, search
    from majortom.build.collections import Source

    def tile(name, code, platform, seconds, cloud=None):
        when = dt.datetime(2024, 8, 29, 18, 23, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=seconds)
        props = {"grid:code": code, "platform": platform, "eo:cloud_cover": cloud}
        return pystac.Item(name, None, None, when, props)

    l2a = [tile("vk", "MGRS-12TVK", "sentinel-2a", 0), tile("vl", "MGRS-12TVL", "sentinel-2a", -15),
           tile("other-satellite", "MGRS-12TVK", "sentinel-2b", 5)]

    class Catalog:
        def search(self, **query):
            return type("Search", (), {"items": lambda self: iter(l2a)})()

    from majortom.build.backends import stac
    monkeypatch.setattr(stac, "catalog", lambda archive: Catalog())
    l1c = [tile("l1c-vk", "MGRS-12TVK", "sentinel-2a", 0, cloud=10), tile("l1c-vl", "MGRS-12TVL", "sentinel-2a", -15,
                                                                           cloud=30)]
    mates = search.same_pass(ARCHIVES["earth-search"], Source("sentinel-2-c1-l2a"), l1c, (0, 0, 1, 1))
    assert [it.id for it in mates] == ["vk", "vl"]                      # same tiles, same satellite
    assert scene_cloud(l1c) == pytest.approx(0.2)                        # fallback: catalogue cloud cover


def test_crs_from_assets_when_the_item_has_none():                  # STAC 1.1 catalogues such as CDSE
    import datetime as dt
    import pystac
    from majortom.build.search import item_crs

    item = pystac.Item("x", None, None, dt.datetime(2024, 8, 29), {})
    item.add_asset("B04_10m", pystac.Asset("s3://eodata/x.jp2", extra_fields={"proj:code": "EPSG:32612"}))
    assert item_crs(item) == "EPSG:32612"


def test_dates_outside_a_source_period_are_refused():
    from majortom.build import download, outside

    assert outside(("2020-01-01", None), "2015-06-01/2015-08-31")
    assert not outside(("2020-01-01", None), "2019-12-01/2020-01-15")             # overlaps
    assert outside(("2015-01-01", "2021-12-31"), "2022-01-01T00:00:00Z/2022-02-01T00:00:00Z")
    assert not outside(None, "1900-01-01/1900-02-01") and not outside(("2020-01-01", None), None)
    with pytest.raises(ValueError, match="hls2-l30 for"):
        download(pd.DataFrame({"row": [0]}), "hls2-l30", "2015-06-01/2015-08-31")


def test_optional_bands_are_saved_only_where_every_tile_has_them():
    import datetime as dt
    import pystac
    from majortom.build import COLLECTIONS, present

    def scene(*keys):
        it = pystac.Item("x", None, None, dt.datetime(2000, 1, 1), {})
        for k in keys:
            it.add_asset(k, pystac.Asset("https://example.org/x.tif"))
        return [it]

    landsat = COLLECTIONS["landsat-c2-l2"]
    source = landsat.source("landsat-c2-l2", "planetary-computer")
    tm = scene("blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel")
    assert present(landsat, source, tm) == list(landsat.assets)                    # Landsat 5: no coastal
    assert present(landsat, source, scene(*tm[0].assets, "coastal"))[-1] == "coastal"


def test_declared_scaling_is_recorded_as_given():
    import datetime as dt
    import pystac
    from majortom.build.collections import declared_scaling

    it = pystac.Item("x", None, None, dt.datetime(2024, 8, 26), {})
    it.add_asset("red", pystac.Asset("x", extra_fields={"raster:bands": [{"scale": 2.75e-05, "offset": -0.2}]}))
    it.add_asset("qa_pixel", pystac.Asset("x", extra_fields={"raster:bands": [{"unit": "bit index"}]}))
    assert declared_scaling(it, "red") == {"scale": 2.75e-05, "offset": -0.2}
    assert declared_scaling(it, "qa_pixel") is None


def test_crs_and_tile_from_an_mgrs_tile_in_the_id():                 # NASA HLS records give no CRS
    import datetime as dt
    import pystac
    from majortom.build.search import item_crs, tile_code

    def item(name):
        return pystac.Item(name, None, None, dt.datetime(2024, 8, 27), {})

    assert item_crs(item("HLS.S30.T12TVK.2024240T181919.v2.0")) == "EPSG:32612"
    assert item_crs(item("HLS.L30.T34HBH.2024240T181919.v2.0")) == "EPSG:32734"       # band H: south
    assert tile_code(item("HLS.S30.T12TVK.2024240T181919.v2.0")) == "MGRS-12TVK"
    assert item_crs(item("Copernicus_DSM_COG_10_N40_00_W112_00_DEM")) is None


def test_tiles_in_far_utm_zones_are_ignored():                      # footprints across 180° match everywhere
    from majortom.build.search import near_zone

    assert near_zone("EPSG:32612", "EPSG:32612") and near_zone("EPSG:32611", "EPSG:32612")
    assert not near_zone("EPSG:32601", "EPSG:32612")
    assert near_zone("EPSG:32660", "EPSG:32601")                       # zones 60 and 1 are neighbours
    assert near_zone("EPSG:4326", "EPSG:32612") and near_zone(None, "EPSG:32612")
