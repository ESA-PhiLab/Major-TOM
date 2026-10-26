"""majortom.release: a rumi sample read as an array and as a COG, and converted to COG files (offline).

Needs rumi and GDAL >= 3.11; skipped otherwise (the project's 3.11 environment has neither yet).
"""
import numpy as np
import pytest
from affine import Affine

rumi = pytest.importorskip("rumi")
geozl = pytest.importorskip("geozl")
rasterio = pytest.importorskip("rasterio")
if tuple(int(v) for v in rasterio.__gdal_version__.split(".")[:2]) < (3, 11):
    pytest.skip("the COG profile needs GDAL >= 3.11", allow_module_level=True)

from majortom.release import read, to_cog   # noqa: E402

BANDS = ["B02", "B03", "B04", "B08"]
TRANSFORM = Affine(10, 0, 331_560, 0, -10, 1_023_300)


@pytest.fixture
def sample(tmp_path):
    """A 4-band 352 px uint16 rumi file with known values, CRS and transform."""
    rng = np.random.default_rng(0)
    data = (np.cumsum(rng.integers(-20, 21, (4, 352, 352)), axis=2) + 2000).clip(0, 65535).astype(np.uint16)
    frames = rumi.frames(data, "b (row h) (col w) -> row col (b h w)", tile_size=176)
    for frame in frames:
        frame.compressed = geozl.compress(frame.data, graph=geozl.graph(frame.data, "planar>zigzag>entropy"))
    path, header = rumi.write(str(tmp_path / "s.rumi"), frames, bands=BANDS, time=["2022-09-20"],
                              transform=tuple(TRANSFORM)[:6], crs="EPSG:32654")
    return path, header, data


def test_array(sample):
    path, header, data = sample
    assert np.array_equal(read(path, header), data)


def test_cog_in_memory(sample):
    path, header, data = sample
    with read(path, header, as_="cog", nodata=0, scale=1e-4, offset=-0.1) as mem, mem.open() as f:
        assert np.array_equal(f.read(), data)
        assert f.crs.to_epsg() == 32654 and f.transform == TRANSFORM
        assert list(f.descriptions) == BANDS and f.nodata == 0
        assert f.scales[0] == pytest.approx(1e-4) and f.offsets[0] == pytest.approx(-0.1)
        assert f.tags(ns="IMAGE_STRUCTURE")["INTERLEAVE"] == "TILE" and f.block_shapes[0] == (176, 176)


def test_to_cog_files(sample, tmp_path):
    path, header, data = sample
    (written,) = to_cog([("MT10km_0624U_0029L_20220920_10m", path, header)], tmp_path / "cogs")
    with rasterio.open(written) as f:
        assert np.array_equal(f.read(), data) and f.transform == TRANSFORM
