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
    from majortom.build import _same_request
    from majortom.cells import Cell

    cell, choice = Cell(450, -910, 10.0), Choice()
    sidecar = tmp_path / "000000.json"
    sidecar.write_text(json.dumps(dict(grid_cell=cell.name, grid_km=cell.d, choice=asdict(choice),
                                       query=dict(datetime="2024-01-01/2024-12-31", margin=0.056))))
    assert _same_request(sidecar, cell, "2024-01-01/2024-12-31", 0.056, choice)
    assert not _same_request(sidecar, Cell(450, -909, 10.0), "2024-01-01/2024-12-31", 0.056, choice)
    assert not _same_request(sidecar, cell, "2020-07-02/2020-08-31", 0.056, choice)
    assert not _same_request(sidecar, cell, "2024-01-01/2024-12-31", 0.056, Choice(cloud=(0.3, 1.0)))
    assert not _same_request(tmp_path / "missing.json", cell, None, 0.056, choice)
