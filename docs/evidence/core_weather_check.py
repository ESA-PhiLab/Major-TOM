"""ERA5 Core-Weather tables for a few real Core-S2L2A acquisitions, for checking by hand.

    Core-S2L2A metadata -> acquisitions in a box around Loch Lomond, the N nearest in time to a date
      -> majortom.climate.series (ERA5, hourly ±24 h, daily ±30 days)
      -> <out>/hourly.csv, <out>/daily.csv, and a summary printed

    python -u docs/evidence/core_weather_check.py <out> [date] [n]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.climate import series                              # noqa: E402

META = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"
BOX = (-6.0, 55.5, -3.0, 57.0)                                    # west, south, east, north around Loch Lomond


def nearby(date: str, n: int) -> pd.DataFrame:
    with fsspec.open(META).open() as f:
        meta = pq.read_table(f, columns=["grid_cell", "grid_row_u", "grid_col_r", "timestamp", "centre_lat",
                                         "centre_lon"]).to_pandas()
    west, south, east, north = BOX
    box = meta[meta.centre_lon.between(west, east) & meta.centre_lat.between(south, north)].copy()
    box["gap"] = (pd.to_datetime(box.timestamp, format="mixed", utc=True) - pd.Timestamp(date, tz="UTC")).abs()
    return box.nsmallest(n, "gap")


def main(out: Path, date: str, n: int) -> None:
    picked = nearby(date, n)
    print(f"{len(picked)} acquisitions, {picked.timestamp.min()} to {picked.timestamp.max()}")
    start = time.time()
    hourly, daily = series(picked, "era5", workers=32, progress=1000)
    print(f"series: {time.time() - start:.0f} s")
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out / "hourly.csv", index=False)
    daily.to_csv(out / "daily.csv", index=False)
    first = hourly.acquired.iloc[0]
    print(hourly[hourly.acquired == first][["grid_cell", "acquired", "step_h", "era5_t2m", "era5_tp", "era5_u10",
                                            "era5_valid"]].iloc[::6].to_string(index=False))
    print(daily[daily.acquired == first][["step_d", "era5_t2m_mean", "era5_t2m_min", "era5_t2m_max", "era5_tp_sum",
                                          "era5_swvl1_mean", "era5_valid"]].iloc[::5].to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "2021-07-01", int(sys.argv[3]) if len(sys.argv) > 3 else 20)
