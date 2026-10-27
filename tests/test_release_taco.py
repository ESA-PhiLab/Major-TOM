"""majortom.release.taco: built samples -> partitioned TACO, read back exactly (offline, synthetic samples).

Needs taco-eo[writer], rumi and GDAL >= 3.11; skipped otherwise, as in the project's 3.11 environment.
"""
import json

import numpy as np
import pandas as pd
import pytest

taco = pytest.importorskip("taco")
rumi = pytest.importorskip("rumi")
pytest.importorskip("geozl")

from majortom.cells import Cell                                  # noqa: E402
from majortom.release.taco import release, sample_id             # noqa: E402
from majortom.spec import window                                 # noqa: E402


def _sample(folder, index, cell, when, rng):
    win = window(cell, 60.0)
    t = win.transform(10)
    arrays = {"B02": rng.integers(0, 3000, (1056, 1056)).astype(np.uint16),
              "B05": rng.integers(0, 3000, (528, 528)).astype(np.uint16)}
    np.savez(folder / f"{index:06d}.npz", **arrays)
    side = dict(bands=list(arrays), window_m=win.side, left=t.c, top=t.f, res=10.0, crs=win.crs, grid_cell=cell.name,
                grid_km=10.0, nodata=0.0, offset={"B02": 1000.0, "B05": 1000.0}, scaling=None, query={"margin": 0.056},
                resampling="bilinear", datetime=when, archive="planetary-computer", collection="sentinel-2-l2a",
                items=[f"S2_{index}"], platform="sentinel-2a", processing_baseline="05.10", cloud=0.01,
                cell_coverage=1.0)
    (folder / f"{index:06d}.json").write_text(json.dumps(side))
    return folder / f"{index:06d}.npz"


def test_release_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    build = tmp_path / "build"
    build.mkdir()
    cells = [Cell(500, 10), Cell(500, 11), Cell(-300, 900)]
    paths = [_sample(build, i, c, f"2023-0{m}-15T10:30:00.123456+00:00", rng)
             for i, (c, m) in enumerate((c, m) for c in cells for m in (5, 6))]
    one = paths[0].stat().st_size                                           # ~2.7 MB of random data per sample
    out = release(paths, tmp_path / "out" / "check.zip", id="check", description="test", licenses=["CC-BY-4.0"],
                  providers=["Major TOM"], max_bytes=int(2.2 * one))
    ds = taco.open_dataset(str(out))
    table = ds.sql("SELECT * FROM dataset").to_pandas()
    assert len(table) == 6 and table["source_file"].nunique() >= 2                 # taco-eo cut by size
    # partitions are consecutive runs of the Hilbert order: each file appears once along the order
    from majortom.release.order import hilbert_order
    sides = {sample_id(s["grid_cell"], 10.0, s["datetime"]): s for s in
             (json.loads(p.with_suffix(".json").read_text()) for p in paths)}
    keyed = table.assign(grid_cell=table["id"].map(lambda i: sides[i]["grid_cell"]),
                         acquired=table["id"].map(lambda i: pd.Timestamp(sides[i]["datetime"])))
    files = keyed.loc[hilbert_order(keyed), "source_file"]
    assert (files != files.shift()).sum() == files.nunique()
    for path in paths:
        side = json.loads(path.with_suffix(".json").read_text())
        row = table[table["id"] == sample_id(side["grid_cell"], 10.0, side["datetime"])].iloc[0]
        assert row["majortom:code"] == sample_id(side["grid_cell"], 10.0, None)
        assert row["source:acquired"] == side["datetime"] and row["source:collection"] == "sentinel-2-l2a"
        data = np.load(path)
        back = np.asarray(rumi.read(row["10m.rumi::location"], row["10m.rumi::header"])).reshape(1, 1056, 1056)
        assert np.array_equal(back[0], data["B02"])
    assert taco.validate(str(out)).ok


def test_sample_id():
    assert sample_id("624U_29L", 10.0, "2021-07-14T11:21:19.024+00:00") == "MT10km_0624U_0029L_20210714T112119"
    assert sample_id("104D_845L", 10.0, None) == "MT10km_0104D_0845L"
