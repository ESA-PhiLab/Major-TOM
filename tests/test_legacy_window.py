"""Core v1's window (docs/SPEC.md 7.1, 7.5): sizes, per-band rounding, and agreement with real Core samples."""
import numpy as np
import pytest
from affine import Affine

from majortom.cells import Cell, cell_at
from majortom.spec import LEGACY_V1, LegacyWindow, correct_core_transform, legacy_window, window


def test_sizes_and_corner():
    cell = cell_at(-4.6, 56.1)
    win = legacy_window(cell, "EPSG:32630")
    assert (win.pixels(10), win.pixels(20), win.pixels(60)) == (1068, 534, 178)
    assert isinstance(window(cell, 60, margin=LEGACY_V1, crs="EPSG:32630"), LegacyWindow)


def test_each_band_on_its_own_grid():
    """The corner is rounded per pixel size, relative to the source grid's origin."""
    win = LegacyWindow("EPSG:32630", 400_003.7, 6_200_044.9, 10_680, origin=(0.0, 0.0))
    assert win.transform(10) == Affine(10, 0, 400_000, 0, -10, 6_200_040)
    assert win.transform(60) == Affine(60, 0, 400_020, 0, -60, 6_200_040)
    # the same rule as the correction of a stored Core origin
    assert win.transform(10) == correct_core_transform(Affine(10, 0, 400_003.7, 0, -10, 6_200_044.9), "Core-S2L2A")


@pytest.mark.network
def test_matches_core_v1_samples():
    """For real Core-S2L2A samples, the legacy window's corner equals the corrected stored origin."""
    import fsspec
    import pyarrow.parquet as pq
    import rasterio
    from fsspec.parquet import open_parquet_file
    from rasterio.io import MemoryFile

    from majortom.cells import Cell

    url = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"
    with fsspec.open(url).open() as f:
        meta = pq.read_table(f, columns=["grid_cell", "parquet_url", "parquet_row"]).to_pandas()

    def stored_b04(row):
        """The B04 geotransform and CRS stored with one Core sample (one row group of its parquet)."""
        with open_parquet_file(row.parquet_url, columns=["B04"], row_groups=[int(row.parquet_row)]) as f:
            table = pq.ParquetFile(f).read_row_group(int(row.parquet_row), columns=["B04"])
        with MemoryFile(table.column(0)[0].as_py()) as m, m.open() as src:
            return src.transform, str(src.crs)

    rows = meta.sample(5, random_state=1)
    for row in rows.itertuples():
        stored, crs = stored_b04(row)
        r, c = row.grid_cell.split("_")
        cell = Cell(int(r[:-1]) * (1 if r[-1] == "U" else -1), int(c[:-1]) * (1 if c[-1] == "R" else -1))
        win = legacy_window(cell, crs)
        assert win.transform(10) == correct_core_transform(stored, "Core-S2L2A"), row.grid_cell


def test_crop_core_to_v2():
    """The crop's transform is the v2 window's, and its pixels are the v1 pixels at the whole-pixel offset."""
    from majortom.spec import cell_crs, crop_core_to_v2, window

    found = 0
    for row in range(-800, 801, 37):
        cell = Cell(row, 11, 10.0)
        crs = cell_crs(cell)
        t = legacy_window(cell, crs).transform(10)
        data = np.arange(1068 * 1068, dtype=np.int64).reshape(1068, 1068)
        stored = Affine(10, 0, t.c + 3.7, 0, -10, t.f - 2.1)            # Core v1 stored the unrounded corner
        out = crop_core_to_v2(data, stored, crs, cell, "Core-S2L2A")
        v2 = window(cell, 60.0, crs=crs)
        if out is None:
            continue
        found += 1
        crop, ct = out
        assert crop.shape == (1056, 1056) and ct == v2.transform(10)
        col, r = round((v2.left - t.c) / 10), round((t.f - v2.top) / 10)
        assert crop[0, 0] == data[r, col]
    assert found > 0


def test_read_tif_corrects_core_transform():
    """read_tif gives the corrected transform for Core samples, the stored one with raw=True."""
    from rasterio.io import MemoryFile
    from majortom.sample import read_tif

    stored = Affine(10, 0, 600_003.7, 0, -10, 5_000_002.1)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=4, height=4, count=1, dtype="uint16", crs="EPSG:32633",
                      transform=stored) as f:
            f.write(np.ones((1, 4, 4), np.uint16))
        raw = mem.read()
    assert read_tif(raw, "Core-S2L2A")[1] == Affine(10, 0, 600_000, 0, -10, 5_000_000)
    assert read_tif(raw, "Core-S2L2A", raw=True)[1] == stored
    assert read_tif(raw, "Core-DEM")[1] == stored
