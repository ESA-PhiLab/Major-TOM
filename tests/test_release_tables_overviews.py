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
