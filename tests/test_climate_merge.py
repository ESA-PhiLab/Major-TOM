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
