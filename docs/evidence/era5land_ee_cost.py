"""Earth Engine cost of ERA5-Land daily means on days centred by UTC offset (docs/CLIMATE.md section 8, option b).

    ECMWF/ERA5_LAND/HOURLY, the 24 hours of one offset group's day (00:00 - g h ... +23 h), 7 bands, mean
      -> one 900 x 900 tile at 0.1° (computePixels), profiled in the main thread, over land and over sea
      -> EECU-s per tile; the global grid is 4 x 2 tiles; the full build is ~3,000 days x 9 offset groups

    EE_PROJECT=major-tom-climate python -u docs/evidence/era5land_ee_cost.py
"""
from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.climate.readers.earthengine import _ee                    # noqa: E402

BANDS = ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m",
         "surface_pressure", "snow_cover", "volumetric_soil_water_layer_1"]
TILES = {"Europe (land)": (-10, 70), "Pacific (sea)": (-170, 20)}       # west, north of a 90° x 90° tile
DAYS, GROUPS = 3000, 9


def main() -> None:
    ee = _ee()
    start = ee.Date("2021-07-14T23:00:00")                                # group +1: local midnight at UTC+1
    image = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, start.advance(24, "hour")) \
        .select(BANDS).mean().unmask(-9999).toFloat()                    # float32: within the 48 MB request limit
    print("hours in the day:", ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, start.advance(24, "hour"))
          .size().getInfo())
    costs = []
    for name, (west, north) in TILES.items():
        out, t = io.StringIO(), time.time()
        with ee.profilePrinting(destination=out):
            block = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": {
                "dimensions": {"width": 900, "height": 900}, "crsCode": "EPSG:4326",
                "affineTransform": {"scaleX": 0.1, "shearX": 0, "translateX": west,
                                    "shearY": 0, "scaleY": -0.1, "translateY": north}}})
        cost = sum(float(m) for m in re.findall(r"^\s*([0-9.]+)\s", out.getvalue(), re.M))
        costs.append(cost)
        t2m = block["temperature_2m"]
        print(f"{name}: {cost:.1f} EECU-s, {time.time() - t:.1f} s, land share {(t2m != -9999).mean():.2f}")
    per_field = sum(costs) / len(costs) * 8
    total = per_field * DAYS * GROUPS / 3600
    print(f"~{per_field:.0f} EECU-s per global 7-band field; full build ~{total:,.0f} EECU-hours "
          f"({DAYS} days x {GROUPS} groups); with requests cut to each group's area, roughly a third of that")


if __name__ == "__main__":
    main()
