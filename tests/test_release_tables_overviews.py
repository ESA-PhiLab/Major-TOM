"""Overview tables: block means equal a direct computation; missing values ignored; empty blocks left out."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from majortom.release.tables.overviews import overviews


def _series(folder, rng):
    rows, cols = rng.integers(-1002, 1002, 400), rng.integers(-2004, 2004, 400)
    t = pd.DataFrame({"grid_cell": [f"c{i}" for i in range(400)], "grid_row_u": rows.astype(np.int16),
                      "grid_col_r": cols.astype(np.int16), "year": np.int16(2020), "month": np.int8(7),
                      "x_t2m": rng.normal(size=400).astype(np.float32), "x_tp": rng.random(400).astype(np.float32)})
    t.loc[::7, "x_t2m"] = np.nan                                        # missing values, some blocks keep x_tp only
    path = folder / "series" / "year=2020" / "07.parquet"
    path.parent.mkdir(parents=True)
    t.to_parquet(path, index=False)
    return t


def test_block_means_match_a_direct_computation(tmp_path):
    t = _series(tmp_path / "src", np.random.default_rng(0))
    assert overviews(tmp_path / "src", tmp_path / "dst", level=2, workers=1) == 1
    got = pq.read_table(tmp_path / "dst" / "overviews" / "year=2020" / "07.parquet").to_pandas()
    t["block_row"], t["block_col"] = (t.grid_row_u.astype(int) + 1002) // 4, (t.grid_col_r.astype(int) + 2004) // 4
    want = t.groupby(["block_row", "block_col"]).agg(x_t2m=("x_t2m", "mean"), x_tp=("x_tp", "mean"),
                                                     n_cells=("x_tp", "count")).reset_index()
    assert list(got.columns) == ["block_row", "block_col", "n_cells", "x_t2m", "x_tp"]
    assert len(got) == len(want)
    merged = want.merge(got, on=["block_row", "block_col"], suffixes=("", "_got"))
    assert (merged.n_cells == merged.n_cells_got).all()
    for c in ("x_t2m", "x_tp"):
        assert np.allclose(merged[c], merged[f"{c}_got"], equal_nan=True, rtol=1e-6)
    assert got[["block_row", "block_col"]].apply(tuple, axis=1).is_monotonic_increasing      # rows in block order


def test_climatology_overviews_match_a_direct_computation(tmp_path):
    from majortom.release.tables.overviews import climatology_overviews

    rng = np.random.default_rng(1)
    rows, cols = rng.integers(-1002, 1002, 300), rng.integers(-2004, 2004, 300)
    t = pd.DataFrame({"grid_cell": np.repeat([f"c{i}" for i in range(300)], 12),
                      "grid_row_u": np.repeat(rows, 12).astype(np.int16), "grid_col_r": np.repeat(cols, 12).astype(np.int16),
                      "month": np.tile(np.arange(1, 13), 300).astype(np.int8),
                      "x_t2m_mean": rng.normal(size=3600).astype(np.float32), "x_t2m_std": rng.random(3600).astype(np.float32),
                      "n_years": np.int16(30)})
    band = (t.grid_row_u.astype(int) + 1002) * 20 // 2004
    for b, part in t.groupby(band):
        path = tmp_path / "src" / "climatology" / f"band={b:02d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
    assert climatology_overviews(tmp_path / "src", tmp_path / "dst", level=2, workers=2) == 12
    got = pq.read_table(tmp_path / "dst" / "climatology_overviews" / "07.parquet").to_pandas()
    july = t[t.month == 7].copy()
    july["block_row"], july["block_col"] = (july.grid_row_u.astype(int) + 1002) // 4, (july.grid_col_r.astype(int) + 2004) // 4
    want = july.groupby(["block_row", "block_col"]).x_t2m_mean.mean().reset_index()
    assert list(got.columns) == ["block_row", "block_col", "n_cells", "x_t2m_mean"]      # normals only
    merged = want.merge(got, on=["block_row", "block_col"], suffixes=("", "_got"))
    assert len(merged) == len(want) == len(got)
    assert np.allclose(merged.x_t2m_mean, merged.x_t2m_mean_got, rtol=1e-6)


def test_static_overview_means_and_most_common_class(tmp_path):
    from majortom.release.tables.overviews import static_overview

    rng = np.random.default_rng(1)
    rows, cols = rng.integers(-20, 20, 3000), rng.integers(-40, 40, 3000)       # dense: several cells per block
    t = pd.DataFrame({"grid_cell": [f"c{i}" for i in range(3000)], "grid_row_u": rows.astype(np.int16),
                      "grid_col_r": cols.astype(np.int16), "era5_z": rng.normal(size=3000).astype(np.float32),
                      "era5_slt": rng.integers(0, 8, 3000).astype(np.float32)})
    t.loc[5, "era5_slt"] = 2.99991                                      # a 16-bit decode error, as in ERA5-Static
    t.loc[::11, "era5_slt"] = np.nan
    (tmp_path / "src").mkdir()
    t.to_parquet(tmp_path / "src" / "static.parquet", index=False)
    n = static_overview(tmp_path / "src", tmp_path / "dst")
    got = pq.read_table(tmp_path / "dst" / "overviews" / "static.parquet").to_pandas()
    assert n == len(got)
    t["block_row"], t["block_col"] = (t.grid_row_u.astype(int) + 1002) // 4, (t.grid_col_r.astype(int) + 2004) // 4
    t["code"] = t.era5_slt.round()
    mode = t.dropna(subset=["code"]).groupby(["block_row", "block_col"]).code.agg(lambda s: s.value_counts().sort_index()
                                                                                  .idxmax())
    want = t.groupby(["block_row", "block_col"]).era5_z.mean().to_frame().join(mode).reset_index()
    merged = want.merge(got, on=["block_row", "block_col"], suffixes=("", "_got"))
    assert len(merged) == len(want) == len(got)
    assert np.allclose(merged.era5_z, merged.era5_z_got, atol=1e-6)
    assert np.array_equal(merged.code.to_numpy(), merged.era5_slt.to_numpy(), equal_nan=True)
