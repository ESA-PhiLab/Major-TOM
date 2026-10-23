"""What an ERA5-Land daily-statistics UTC offset means (docs/CLIMATE.md section 8): tested against ERA5 hourly data.

    ERA5-Land t2m daily_mean, April 2021, an area around Loch Lomond, offsets utc+00:00, utc+12:00, utc-06:00
      -> cell 624U_29L per day (dates from the file's "days since" units)
    ERA5 hourly t2m (ARCO) for the same cell, 31 March - 1 May
      -> for each offset o, the 24 h window day d would cover under three hypotheses:
           local midnight  d 00:00 - o      opposite sign  d 00:00 + o      no shift  d 00:00
      -> spread (standard deviation over the 30 days) of ERA5-Land minus the ERA5 window mean
The bias between ERA5-Land (9 km, elevation-adjusted) and ERA5 (31 km) is removed by using the spread, and day-to-day
weather (1-3 K) separates the hypotheses. A single day cannot: overlapping windows differ by tenths of a kelvin.

    python -u docs/evidence/era5land_offset.py <out>
Needs the Climate Data Store token (`python -m majortom.build.credentials cds`) and the dataset's licence accepted.
First run, 2026-10-23: local midnight wins (spread 0.63 K for utc+12:00, 0.59 K for utc-06:00, 0.60 K for utc+00:00;
opposite sign 1.46 / 1.03 K, no shift 0.91 / 0.76 K); requests took 176-503 s.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.build.credentials import cds_token                       # noqa: E402
from majortom.climate import Grid, cell_means, weights                  # noqa: E402
from majortom.climate.sources import SOURCES, field                     # noqa: E402

CELL = pd.DataFrame({"grid_row_u": [624], "grid_col_r": [-29]})        # 624U_29L
AREA = [57, -6, 55.5, -3]                                              # north, west, south, east
OFFSETS = {"utc+00:00": 0, "utc+12:00": 12, "utc-06:00": -6}
DAYS = [f"{d:02d}" for d in range(1, 31)]                              # April 2021


def fetch(offset: str, out: Path) -> tuple[Path, float]:
    import cdsapi

    target = out / f"t2m_{offset.replace(':', '')}.nc"
    start = time.time()
    if not target.exists():
        client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=cds_token(), quiet=True)
        client.retrieve("derived-era5-land-daily-statistics", {
            "variable": ["2m_temperature"], "year": "2021", "month": "04", "day": DAYS,
            "daily_statistic": "daily_mean", "time_zone": offset, "frequency": "1_hourly", "area": AREA}, str(target))
    return target, time.time() - start


def era5land_days(path: Path) -> pd.Series:
    """Cell mean of 624U_29L for each day in the file, indexed by its date."""
    with rasterio.open(f'netcdf:"{path}":t2m') as src:
        tags = src.tags()
        origin = pd.Timestamp(tags["valid_time#units"].removeprefix("days since "))
        days = [origin + pd.Timedelta(days=int(v)) for v in tags["NETCDF_DIM_valid_time_VALUES"].strip("{}").split(",")]
        t = src.transform
        w = weights(CELL, Grid(west=t.c, north=t.f, step=t.a, width=src.width, height=src.height))
        values = []
        for band in range(1, src.count + 1):
            raw = src.read(band)
            data = raw.astype(float) * (src.scales[band - 1] or 1) + (src.offsets[band - 1] or 0)
            if src.nodata is not None:
                data[raw == src.nodata] = np.nan
            values.append(float(cell_means(w, data)[0][0]))
        return pd.Series(values, index=pd.DatetimeIndex(days))


def era5_hourly() -> pd.Series:
    era5 = SOURCES["era5"]
    w = weights(CELL, era5.grid)
    hours = pd.date_range("2021-03-31T00:00", "2021-05-01T23:00", freq="1h")
    with ThreadPoolExecutor(32) as pool:
        values = list(pool.map(lambda h: float(cell_means(w, field(era5, "t2m", h))[0][0]), hours))
    return pd.Series(values, index=hours)


def window_means(hourly: pd.Series, starts: pd.DatetimeIndex) -> np.ndarray:
    return np.array([hourly[s:s + pd.Timedelta(hours=23)].mean() for s in starts])


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(OFFSETS)) as pool:
        fetched = dict(zip(OFFSETS, pool.map(lambda o: fetch(o, out), OFFSETS)))
    hourly = era5_hourly()
    for offset, hours in OFFSETS.items():
        path, seconds = fetched[offset]
        land = era5land_days(path)
        print(f"{offset}: {len(land)} days, request took {seconds:.0f} s")
        for name, shift in [("local midnight", -hours), ("opposite sign", hours), ("no shift", 0)]:
            diff = land.to_numpy() - window_means(hourly, land.index + pd.Timedelta(hours=shift))
            print(f"   {name:15s} day starts d 00:00 {shift:+d} h: spread {diff.std():.2f} K, bias {diff.mean():+.2f} K")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
