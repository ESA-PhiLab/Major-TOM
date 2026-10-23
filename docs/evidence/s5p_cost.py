"""Earth Engine cost of Sentinel-5P cell means: one product, one day, two ways (docs/CLIMATE.md section 7).

    Core-S2L2A metadata (grid row, column, timestamp)
      -> distinct acquisition days; cells needed on a test day (acquisitions within ±30 days)
      -> the day's NO2 mosaic (mean of its orbits)
      -> A: cell means on Earth Engine, reduceRegions over a subset of those cells     (profiled)
      -> B: one 900 x 900 tile of a 0.05° global grid, computePixels                   (profiled)
      -> EECU-seconds per day and product, extrapolated to Core's days x 7 products

    EE_PROJECT=major-tom-climate python docs/evidence/s5p_cost.py [YYYY-MM-DD]
Credentials come from majortom.build.credentials (EE_SERVICE_ACCOUNT_KEY or the saved settings).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import ee
import fsspec
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.build.credentials import earth_engine_settings     # noqa: E402
from majortom.cells import Cell                                  # noqa: E402

META = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"
NO2 = ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density")
S5P_START = "2018-06-28"           # first OFFL NO2 day on Earth Engine (to confirm in the output)
WINDOW = 30                        # days either side of an acquisition (Core-Weather daily table)
SUBSET = 2000                      # cells sent to reduceRegions; scaled up to the day's full count
PRODUCTS = 7                       # NO2, CO, O3, CH4, SO2, HCHO, aerosol index
GRID = 0.05                        # degrees, option B
TILE = 900                         # pixels per side of the measured tile; the globe is 7200 x 3600


def sign_in() -> None:
    project, key_file = earth_engine_settings()
    email = json.loads(Path(key_file).read_text())["client_email"] if key_file else None
    ee.Initialize(ee.ServiceAccountCredentials(email, key_file) if key_file else None, project=project)


def acquisitions() -> pd.DataFrame:
    """grid row, column and acquisition day of every Core-S2L2A sample (three columns only)."""
    with fsspec.open(META).open() as f:
        meta = pq.read_table(f, columns=["grid_row_u", "grid_col_r", "timestamp"]).to_pandas()
    meta["day"] = pd.to_datetime(meta["timestamp"], format="mixed", utc=True).dt.normalize()
    return meta


def cells_on(meta: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Distinct cells whose acquisition lies within ±WINDOW days of `day`: the cells needed that day."""
    near = meta[(meta["day"] - day).abs() <= pd.Timedelta(days=WINDOW)]
    return near.drop_duplicates(["grid_row_u", "grid_col_r"])


def eecu(run) -> tuple[object, str]:
    """Run `run()` under Earth Engine's profiler; returns its result and the profile table as text."""
    out = io.StringIO()
    with ee.profilePrinting(destination=out):
        result = run()
    return result, out.getvalue()


def mosaic(day: pd.Timestamp) -> ee.Image:
    start = ee.Date(day.strftime("%Y-%m-%d"))
    return ee.ImageCollection(NO2[0]).filterDate(start, start.advance(1, "day")).select(NO2[1]).mean()


def boxes(cells: pd.DataFrame) -> ee.FeatureCollection:
    def box(r, c):
        cell = Cell(int(r), int(c))
        return ee.Feature(ee.Geometry.Rectangle([cell.lon, cell.lat, cell.lon + cell.dlon, cell.lat + cell.dlat],
                                                geodesic=False))
    return ee.FeatureCollection([box(r, c) for r, c in zip(cells["grid_row_u"], cells["grid_col_r"])])


def tile(image: ee.Image) -> bytes:
    """One TILE x TILE block of the 0.05° grid over Europe and Africa (west edge 0°, north edge 60°N)."""
    return ee.data.computePixels({
        "expression": image.unmask(-9999),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": TILE, "height": TILE},
                 "affineTransform": {"scaleX": GRID, "shearX": 0, "translateX": 0,
                                     "shearY": 0, "scaleY": -GRID, "translateY": 60},
                 "crsCode": "EPSG:4326"}})


def main(test_day: str | None) -> None:
    meta = acquisitions()
    days = meta["day"].dt.date
    print(f"Core-S2L2A: {len(meta):,} samples, {days.nunique():,} distinct days, {days.min()} to {days.max()}")
    per_day = meta.groupby("day").size()
    print(f"samples per day: median {per_day.median():.0f}, max {per_day.max():,}")

    s5p_days = meta.loc[meta["day"] >= pd.Timestamp(S5P_START, tz="UTC"), "day"]
    needed = pd.Index(sorted({d + pd.Timedelta(days=k) for d in s5p_days.unique() for k in range(-WINDOW, WINDOW + 1)}))
    print(f"days needed for Sentinel-5P (acquisitions from {S5P_START}, ±{WINDOW} days): {len(needed):,}")

    busiest = per_day[per_day.index >= pd.Timestamp(S5P_START, tz="UTC")].idxmax()    # the worst case
    day = pd.Timestamp(test_day, tz="UTC") if test_day else busiest
    cells = cells_on(meta, day)
    print(f"test day {day.date()}: {len(cells):,} cells needed")

    sign_in()
    image = mosaic(day)
    print(f"orbits that day: {ee.ImageCollection(NO2[0]).filterDate(day.strftime('%Y-%m-%d'), (day + pd.Timedelta(days=1)).strftime('%Y-%m-%d')).size().getInfo()}")

    subset = cells.sample(min(SUBSET, len(cells)), random_state=0)
    means, profile_a = eecu(lambda: image.reduceRegions(boxes(subset), ee.Reducer.mean(), scale=1113.2).getInfo())
    valid = sum(f["properties"].get("mean") is not None for f in means["features"])
    print(f"\nA: reduceRegions over {len(subset):,} cells ({valid:,} with values); scale x{len(cells) / len(subset):.1f} "
          f"for the full day\n{profile_a}")

    block, profile_b = eecu(lambda: tile(image))
    print(f"B: one {TILE} x {TILE} tile at {GRID}° ({len(block) / 1e6:.1f} MB); the globe is "
          f"{(360 / GRID) * (180 / GRID) / TILE ** 2:.0f} tiles\n{profile_b}")
    print(f"extrapolate: EECU-s per day x {len(needed):,} days x {PRODUCTS} products / 3600 = EECU-hours")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
