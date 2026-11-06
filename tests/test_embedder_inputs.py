"""majortom.embedder inputs: built samples and TACO records embed like Core rows (offline, synthetic, dummy model).

Needs torch, opencv and geopandas (the `embed` extra); the TACO test also needs rumi and taco-eo. Skipped otherwise.
"""
import hashlib
import json

import numpy as np
import pytest
import shapely

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("open_clip")                                  # majortom.embedder imports every model wrapper

from majortom.cells import Cell                                   # noqa: E402
from majortom.embedder import MajorTOM_Embedder                   # noqa: E402
from majortom.embedder.inputs import from_sample, from_taco      # noqa: E402
from majortom.spec import window                                  # noqa: E402


class Mean(torch.nn.Module):
    """Dummy model: the mean of each band, repeated to 4 values."""
    bands, size = ["B02", "B05"], (224, 224)

    def forward(self, x):                                         # (n, bands, h, w) -> (n, 4)
        return x.mean((2, 3)).repeat(1, 2)


def _sample(folder, index=0, cell=Cell(624, -29)):
    win = window(cell, 60.0)
    t = win.transform(10)
    rng = np.random.default_rng(index)
    arrays = {"B02": rng.integers(0, 3000, (1056, 1056)).astype(np.uint16),
              "B05": rng.integers(0, 3000, (528, 528)).astype(np.uint16)}
    np.savez(folder / f"{index:06d}.npz", **arrays)
    side = dict(bands=list(arrays), window_m=win.side, left=t.c, top=t.f, res=10.0, crs=win.crs, grid_cell=cell.name,
                grid_km=10.0, nodata=0.0, offset={"B02": 1000.0, "B05": 1000.0}, scaling=None, query={"margin": 0.056},
                resampling="bilinear", datetime="2023-03-15T10:30:00.123456+00:00", archive="planetary-computer",
                collection="sentinel-2-l2a", items=[f"S2_{index}"], platform="sentinel-2a",
                processing_baseline="05.10", cloud=0.01, cell_coverage=1.0)
    (folder / f"{index:06d}.json").write_text(json.dumps(side))
    return folder / f"{index:06d}.npz", arrays, t


def test_from_sample_stacks_and_georeferences(tmp_path):
    path, arrays, t = _sample(tmp_path)
    image = from_sample(path, ["B02", "B05"])
    assert image.data.shape == (1056, 1056, 2) and image.data.dtype == np.float32
    assert np.array_equal(image.data[..., 0], arrays["B02"])
    assert np.array_equal(image.data[::2, ::2, 1], arrays["B05"])            # 20 m band upsampled by nearest
    assert (image.transform.a, image.transform.c, image.transform.f) == (10.0, t.c, t.f)
    assert (image.timestamp, image.product_id, image.grid_cell) == ("20230315T103000", "S2_0", "624U_29L")


def test_from_sample_names_missing_bands(tmp_path):
    path, _, _ = _sample(tmp_path)
    with pytest.raises(KeyError, match="B04"):
        from_sample(path, ["B04"])


def test_embed_sample_rows_and_checksum(tmp_path):
    path, _, _ = _sample(tmp_path)
    rows = MajorTOM_Embedder(Mean()).embed_sample(path, device="cpu")
    assert len(rows) == 25                                                    # 1056 px in fragments of 224, 10% overlap
    cell = shapely.box(-4.663, 56.048, -4.502, 56.138)
    assert rows.geometry.union_all().intersects(cell) and set(rows.grid_cell) == {"624U_29L"}
    assert (rows.grid_row_u.iloc[0], rows.grid_col_r.iloc[0]) == (624, -29)
    r = rows.iloc[0]
    text = f"{r.geometry}_{r.timestamp}_{r.product_id}_{np.asarray(r.embedding, dtype=np.float64)}"
    assert hashlib.sha256(text.encode()).hexdigest() == r.unique_id


def test_taco_record_matches_sample(tmp_path):
    taco = pytest.importorskip("taco")
    pytest.importorskip("rumi")
    pytest.importorskip("geozl")
    from majortom.release.taco import release

    build = tmp_path / "build"
    build.mkdir()
    path, _, _ = _sample(build)
    out = release([path], tmp_path / "release" / "x.zip", id="x", description="test", licenses=["CC-BY-4.0"],
                  providers=["test"], geoenrich=False, stats=False)
    record = taco.open_dataset(str(out.parent)).sql("SELECT * FROM dataset").to_pandas().iloc[0].to_dict()
    a, b = from_sample(path, ["B02", "B05"]), from_taco(record, ["B02", "B05"])
    assert np.array_equal(a.data, b.data)
    assert (b.transform.a, b.transform.c, b.transform.f) == (a.transform.a, a.transform.c, a.transform.f)
    assert (b.timestamp, b.product_id, b.grid_cell, b.crs) == (a.timestamp, a.product_id, a.grid_cell, a.crs)
