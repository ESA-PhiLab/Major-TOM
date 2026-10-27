"""majortom.release.groups: a built sample's raster groups by pixel size (offline)."""
import json

import numpy as np
import pytest
from affine import Affine

from majortom.cells import Cell
from majortom.release.groups import groups
from majortom.spec import legacy_window

SIZES = {"B02": 1056, "B03": 1056, "B05": 528, "B01": 176}


def _write(tmp_path, side, sizes):
    arrays = {b: np.full((n, n), i, np.uint16) for i, (b, n) in enumerate(sizes.items())}
    np.savez(tmp_path / "000001.npz", **arrays)
    (tmp_path / "000001.json").write_text(json.dumps(side))
    return tmp_path / "000001.npz"


def test_v2_groups(tmp_path):
    side = dict(bands=list(SIZES), window_m=10_560.0, left=300_000.0, top=5_000_040.0, res=10.0, crs="EPSG:32633",
                grid_cell="500U_10R", grid_km=10.0, nodata=0.0, offset={b: 1000.0 for b in SIZES}, scaling=None,
                query={"margin": 0.056}, resampling="bilinear")
    found, _ = groups(_write(tmp_path, side, SIZES))
    assert [g.name for g in found] == ["10m", "20m", "60m"]
    assert [g.bands for g in found] == [["B02", "B03"], ["B05"], ["B01"]]
    assert found[0].data.shape == (2, 1056, 1056) and found[2].data.shape == (1, 176, 176)
    assert found[1].transform == Affine(20, 0, 300_000, 0, -20, 5_000_040)
    assert found[0].stored_offset == 1000.0 and found[0].nodata == 0.0


def test_v1_groups_use_each_grid(tmp_path):
    """Core v1 windows: each pixel size has its own corner (docs/SPEC.md 7.2), here a southern zone."""
    cell, crs = Cell(-611, -478), "EPSG:32718"
    win = legacy_window(cell, crs)
    t10 = win.transform(10)
    side = dict(bands=["B02", "B05", "B01"], window_m=10_680.0, left=t10.c, top=t10.f, res=10.0, crs=crs,
                grid_cell=cell.name, grid_km=10.0, nodata=0.0, offset=None, scaling=None, query={"margin": "v1"}, resampling="bilinear")
    found, _ = groups(_write(tmp_path, side, {"B02": 1068, "B05": 534, "B01": 178}))
    assert [g.transform for g in found] == [win.transform(10), win.transform(20), win.transform(60)]
    assert found[2].transform.f % 60 == 40                    # Sentinel-2's 60 m grid in a southern zone


def test_mixed_offsets_in_a_group_fail(tmp_path):
    side = dict(bands=["B02", "B03"], window_m=10_560.0, left=0.0, top=0.0, res=10.0, crs="EPSG:32633",
                grid_cell="0U_0R", grid_km=10.0, offset={"B02": 1000.0, "B03": 0.0}, query={},
                resampling="bilinear")
    with pytest.raises(ValueError, match="different offsets"):
        groups(_write(tmp_path, side, {"B02": 1056, "B03": 1056}))


def test_mixed_dtypes_get_their_own_groups(tmp_path):
    """ESA CCI land cover: uint16 and uint8 bands at 300 m become two groups, never cast."""
    side = dict(bands=["lccs_class", "change_count"], window_m=10_800.0, left=0.0, top=0.0, res=300.0,
                crs="EPSG:32615", grid_cell="429U_786L", grid_km=10.0, nodata=0.0, query={}, resampling="nearest")
    np.savez(tmp_path / "s.npz", lccs_class=np.ones((36, 36), np.uint16), change_count=np.ones((36, 36), np.uint8))
    (tmp_path / "s.json").write_text(json.dumps(side))
    found, _ = groups(tmp_path / "s.npz")
    assert {g.name: str(g.data.dtype) for g in found} == {"300m-uint16": "uint16", "300m-uint8": "uint8"}
    assert all(g.categorical for g in found)
