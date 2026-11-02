"""majortom.climate.merge: per-source folders joined into one table per kind (offline)."""
import pandas as pd

from majortom.climate.merge import merge


def write(root, source, kind, day, rows):
    path = root / source / kind / f"year={day[:4]}" / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_outer_join_per_day(tmp_path):
    t = pd.Timestamp("2021-07-01T11:21")
    write(tmp_path, "era5", "daily", "2021-07-01", {"grid_cell": ["a", "a"], "acquired": [t, t], "step_d": [0, 1],
                                                    "era5_t2m_mean": [290.0, 291.0]})
    write(tmp_path, "cpc", "daily", "2021-07-01", {"grid_cell": ["a"], "acquired": [t], "step_d": [0], "cpc_precip": [2.0],
                                                   "cpc_start": [pd.Timestamp("2021-06-30T18:00")]})
    write(tmp_path, "era5", "daily", "2021-07-02", {"grid_cell": ["b"], "acquired": [t], "step_d": [0],
                                                    "era5_t2m_mean": [280.0]})
    assert merge(tmp_path, ["era5", "cpc"], tmp_path / "out", kinds=("daily",)) == 2
    day = pd.read_parquet(tmp_path / "out" / "daily" / "year=2021" / "2021-07-01.parquet")
    assert list(day.era5_t2m_mean) == [290.0, 291.0] and day.cpc_precip.iloc[0] == 2.0 and pd.isna(day.cpc_precip.iloc[1])
    other = pd.read_parquet(tmp_path / "out" / "daily" / "year=2021" / "2021-07-02.parquet")
    assert list(other.columns) == list(day.columns) and pd.isna(other.cpc_precip.iloc[0])   # one schema
    assert other.dtypes.equals(day.dtypes)
    assert merge(tmp_path, ["era5", "cpc"], tmp_path / "out", kinds=("daily",)) == 0     # already merged


def test_column_of_an_earlier_source_is_kept(tmp_path):
    """Groups of one source share columns (era5_valid, era5_land): the first group's copy stays, once."""
    t = pd.Timestamp("2021-07-01T11:21")
    keys = {"grid_cell": ["a"], "acquired": [t], "step_d": [0]}
    write(tmp_path, "era5-g1", "daily", "2021-07-01", {**keys, "era5_u100_mean": [3.0], "era5_valid": [1.0], "era5_land": [0.4]})
    write(tmp_path, "era5-g4", "daily", "2021-07-01", {**keys, "era5_p2ww_mean": [1.5], "era5_valid": [0.6], "era5_land": [0.4]})
    merge(tmp_path, ["era5-g1", "era5-g4"], tmp_path / "out", kinds=("daily",))
    day = pd.read_parquet(tmp_path / "out" / "daily" / "year=2021" / "2021-07-01.parquet")
    assert list(day.columns) == ["grid_cell", "acquired", "step_d", "era5_u100_mean", "era5_valid", "era5_land", "era5_p2ww_mean"]
    assert day.era5_valid.iloc[0] == 1.0                                             # era5-g1's


def test_fill_replaces_rows_where_the_fill_covers_more(tmp_path):
    """ERA5-Land south of 57.1°S: rows with no merged values are filled, partly covered rows replaced, others kept."""
    from majortom.climate.merge import fill

    t = pd.Timestamp("2017-01-25T12:00")
    keys = {"grid_cell": ["south", "edge", "north"], "acquired": [t] * 3, "step_d": [0] * 3}
    merged = tmp_path / "merged"
    write(merged.parent, "merged", "daily", "2017-01-25", {**keys, "era5_t2m_mean": [250.0, 270.0, 280.0],
          "era5land_t2m_mean": [None, 271.0, 281.0], "era5land_valid": [None, 0.4, 1.0]})
    write(tmp_path, "south-g1", "daily", "2017-01-25", {"grid_cell": ["south", "edge"], "acquired": [t] * 2,
          "step_d": [0, 0], "era5land_t2m_mean": [249.0, 272.0], "era5land_valid": [1.0, 1.0]})
    assert fill(tmp_path, ["south-g1"], merged, kinds=("daily",)) == 1
    day = pd.read_parquet(merged / "daily" / "year=2017" / "2017-01-25.parquet")
    assert list(day.era5land_t2m_mean) == [249.0, 272.0, 281.0] and list(day.era5land_valid) == [1.0, 1.0, 1.0]
    assert list(day.era5_t2m_mean) == [250.0, 270.0, 280.0] and list(day.grid_cell) == ["south", "edge", "north"]
