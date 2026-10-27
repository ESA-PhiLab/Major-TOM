"""majortom.release: rumi profiles and the rumi writer (offline).

Needs rumi and GDAL >= 3.11 (for the COG check); skipped otherwise, as in the project's 3.11 environment.
"""
import numpy as np
import pytest
from affine import Affine

rumi = pytest.importorskip("rumi")
pytest.importorskip("geozl")
rasterio = pytest.importorskip("rasterio")

from majortom.release.profiles import RUMI_PROFILES, profile   # noqa: E402
from majortom.release.rumi import write_rumi                    # noqa: E402

BANDS = ["B02", "B03", "B04", "B08"]
TRANSFORM = Affine(10, 0, 331_560, 0, -10, 1_023_300)


@pytest.fixture
def data():
    """4 bands, 352 px, uint16, smooth like imagery so that lossy bounds save space."""
    rng = np.random.default_rng(0)
    return (np.cumsum(rng.integers(-20, 21, (4, 352, 352)), axis=2) + 2000).clip(0, 65535).astype(np.uint16)


def test_lossless_round_trip(tmp_path, data):
    w = write_rumi(tmp_path / "s.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2022-09-20T10:15:00")
    back = np.asarray(rumi.read(str(w.path), w.header)).reshape(data.shape)
    assert np.array_equal(back, data) and w.error is None and ">" in w.graph
    meta = rumi.info(source=str(w.path), header=w.header)
    assert list(meta.bands) == BANDS and meta.crs == 32654 and tuple(meta.transform) == tuple(TRANSFORM)[:6]
    if tuple(int(v) for v in rasterio.__gdal_version__.split(".")[:2]) >= (3, 11):
        from majortom.release import read
        with read(w.path, w.header, as_="cog") as mem, mem.open() as f:
            assert np.array_equal(f.read(), data)


@pytest.mark.parametrize("name, bound", [("near-lossless-1", 1), ("near-lossless-2", 2)])
def test_near_lossless_within_bound(tmp_path, data, name, bound):
    exact = write_rumi(tmp_path / "exact.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2022-09-20")
    lossy = write_rumi(tmp_path / f"{name}.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2022-09-20", name)
    back = np.asarray(rumi.read(str(lossy.path), lossy.header)).reshape(data.shape)
    assert np.abs(back.astype(int) - data.astype(int)).max() <= bound
    assert lossy.path.stat().st_size < exact.path.stat().st_size


def test_fixed_graph_is_used(tmp_path, data):
    w = write_rumi(tmp_path / "g.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2022-09-20",
                   graph="planar>zigzag>entropy")
    assert w.graph == "planar>zigzag>entropy"


def test_unknown_profile():
    with pytest.raises(ValueError, match="available: " + ", ".join(RUMI_PROFILES)):
        profile("tiny")


def test_all_predictors_are_tried():
    """choose_graph searches every predictor family, not only planar (geozl.profile's default)."""
    import geozl

    from majortom.release.profiles import choose_graph

    rng = np.random.default_rng(1)
    frame = (np.cumsum(np.cumsum(rng.integers(-3, 4, (2, 176, 176)), 1), 2) + 3000).clip(0, 65535).astype(np.uint16)
    planar = {r["graph"] for r in geozl.profile(frame, reps=1)}
    every = {r["graph"] for r in geozl.profile(frame, prior=None, reps=1)}
    assert planar < every
    assert choose_graph([frame], RUMI_PROFILES["lossless"]) in every


def test_fractional_seconds_and_intervals(tmp_path, data):
    """rumi records whole seconds: the time is rounded down; a (start, end) pair is an interval."""
    w = write_rumi(tmp_path / "t.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2023-09-15T16:49:19.024000+00:00")
    assert str(rumi.info(source=str(w.path), header=w.header).time[0]).startswith("2023-09-15 16:49:19")
    y = write_rumi(tmp_path / "y.rumi", data, TRANSFORM, "EPSG:32654", BANDS, ("2020-01-01", "2020-12-31"))
    assert rumi.info(source=str(y.path), header=y.header).time_kind != "instant"


def test_categorical_refuses_lossy(tmp_path, data):
    with pytest.raises(ValueError, match="categorical"):
        write_rumi(tmp_path / "c.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2023-01-01", "near-lossless-1",
                   categorical=True)
    write_rumi(tmp_path / "c.rumi", data, TRANSFORM, "EPSG:32654", BANDS, "2023-01-01", categorical=True)
