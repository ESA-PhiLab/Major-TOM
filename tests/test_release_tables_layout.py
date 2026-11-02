"""Publishing layout: every file reordered along the Hilbert curve, values unchanged, small row groups (no network)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from majortom.cells import Cell
from majortom.release.tables.layout import centroids, curve_rank, repack
from majortom.climate.monthly import climatology
from majortom.release.order import hilbert


def _monthly(folder, n=300, years=(2000, 2001)):
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(-900, 900, n), rng.integers(-40, 40, n)
    cells = pd.DataFrame({"grid_cell": [Cell(int(r), int(c)).name for r, c in zip(rows, cols)],
                          "grid_row_u": rows, "grid_col_r": cols}).drop_duplicates("grid_cell").reset_index(drop=True)
    folder.mkdir()
    cells.to_parquet(folder / "cells.parquet", index=False)
    for y in range(years[0], years[1] + 1):
        for m in range(1, 13):
            path = folder / "series" / f"year={y}" / f"{m:02d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"grid_cell": cells.grid_cell, "year": np.int16(y), "month": np.int8(m),
                          "x_t2m": rng.normal(size=len(cells)).astype(np.float32),
                          "x_valid": np.float32(1)}).to_parquet(path, index=False)
    climatology(folder, years)
    return cells


def test_centroids_match_cells():
    for r, c in [(624, -29), (-35, -667), (0, 0), (1001, 3)]:
        lon, lat = centroids(np.array([r]), np.array([c]))
        assert np.allclose((lon[0], lat[0]), Cell(r, c).centroid)


def test_repack_reorders_without_changing_values(tmp_path):
    cells = _monthly(tmp_path / "src")
    bands = len(list((tmp_path / "src" / "climatology").glob("band=*.parquet")))
    assert repack(tmp_path / "src", tmp_path / "dst", workers=2, row_group=64) == 24 + bands
    lon, lat = centroids(cells.grid_row_u.to_numpy(), cells.grid_col_r.to_numpy())
    curve = pd.Series(hilbert(lon, lat), index=cells.grid_cell)
    for rel in ["series/year=2001/07.parquet", *(p.relative_to(tmp_path / "src") for p in
                                                (tmp_path / "src" / "climatology").glob("band=*.parquet"))]:
        old, new = pd.read_parquet(tmp_path / "src" / rel), pd.read_parquet(tmp_path / "dst" / rel)
        assert list(new.columns[:3]) == ["grid_cell", "grid_row_u", "grid_col_r"]
        assert np.all(np.diff(curve.reindex(new.grid_cell).to_numpy()) >= 0)          # along the curve
        keys = ["grid_cell", "month"]
        merged = old.merge(new, on=keys, suffixes=("", "_new"))
        assert len(merged) == len(old) == len(new)
        for c in old.columns.difference(keys):
            assert np.array_equal(merged[c].to_numpy(), merged[f"{c}_new"].to_numpy(), equal_nan=True)
    meta = pq.ParquetFile(tmp_path / "dst" / "series/year=2001/07.parquet").metadata
    assert meta.num_row_groups == -(-len(cells) // 64)
    ordered = pd.read_parquet(tmp_path / "dst" / "cells.parquet")
    assert (curve_rank(cells).reindex(ordered.grid_cell).to_numpy() == np.arange(len(cells))).all()


def test_nan_becomes_null(tmp_path):
    """Missing values are nulls in every file: NaN (e.g. from an Arrow-written climatology) is converted."""
    import pyarrow as pa
    from majortom.release.tables.layout import _nan_as_null

    t = _nan_as_null(pa.table({"x": pa.array([1.0, float("nan"), None]), "n": pa.array([1, 2, 3])}))
    assert t.column("x").null_count == 2 and t.column("x")[0].as_py() == 1.0 and t.column("n").null_count == 0


def test_climatology_rows_by_rank_and_page_index(tmp_path):
    """A cell's 12 months are rows 12 x rank of the band's cells in cells.parquet; page index, no value dictionaries."""
    from majortom.release.tables.layout import relayout_climatology

    _monthly(tmp_path / "src")
    repack(tmp_path / "src", tmp_path / "dst", workers=2, row_group=64)
    order = pd.read_parquet(tmp_path / "dst" / "cells.parquet").grid_cell.tolist()
    for path in (tmp_path / "dst" / "climatology").glob("band=*.parquet"):
        clim = pd.read_parquet(path)
        band = [c for c in order if c in set(clim.grid_cell)]
        assert clim.grid_cell.tolist() == [c for c in band for _ in range(12)]
        assert clim.month.tolist() == list(range(1, 13)) * len(band)
        rg = pq.ParquetFile(path).metadata.row_group(0)
        cols = {rg.column(i).path_in_schema: rg.column(i) for i in range(rg.num_columns)}
        assert cols["x_t2m_mean"].has_offset_index and cols["x_t2m_mean"].dictionary_page_offset is None
        assert cols["grid_cell"].dictionary_page_offset is not None
    before = {p.name: pd.read_parquet(p) for p in (tmp_path / "src" / "climatology").glob("band=*.parquet")}
    published = tmp_path / "dst"                                       # relayout of files written the old way
    for p in (published / "climatology").glob("band=*.parquet"):
        pq.write_table(pq.read_table(p), p, row_group_size=64)
    old = {p.name: pd.read_parquet(p) for p in (published / "climatology").glob("band=*.parquet")}
    relayout_climatology(published, workers=2)
    for name, frame in old.items():
        new = pq.ParquetFile(published / "climatology" / name)
        assert new.metadata.row_group(0).column(4).has_offset_index
        pd.testing.assert_frame_equal(new.read().to_pandas(), frame)
    assert set(before) == set(old)
