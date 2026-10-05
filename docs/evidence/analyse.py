"""Summarise sampled geotransforms and compare with the notebook's footprint formula."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyproj

sys.path.insert(0, "/home/miko/Projects/asterisk/Major-TOM/MajorTOM")
from grid import Grid  # noqa: E402

g = Grid.__new__(Grid)
g.dist, g.latitude_range, g.longitude_range = 10, (-85, 85), (-180, 180)
g.rows, g.lats = g.get_rows()
row_lat = dict(zip(g.rows, g.lats))


def cell_latlon(cell: str) -> "tuple":
    """Bottom-left lat/lon of a Major TOM 10 km cell like '487U_329R'."""
    r, c = cell.split("_")
    lat = row_lat[r]
    cols, lons = g.subdivide_circumference(lat, return_cols=True)
    return lat, float(lons[list(cols).index(c)])


def frac(v: float, res: float) -> float:
    """Signed offset of v from the nearest multiple of res."""
    return v - res * round(v / res)


recs = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
print(f"{'cell':12} {'tile':6} {'crs':10} {'x%10':>7} {'y%10':>7} {'dx_pred':>8} {'dy_pred':>8} tilezone shape")
stats = []
for r in recs:
    if "error" in r:
        print("ERR", r["grid_cell"], r["error"][:100]); continue
    b = r["B04"]
    tile = re.search(r"_T(\d\d[A-Z]{3})_", r["product_id"]).group(1)
    epsg = int(b["crs"].split(":")[1])
    lat, lon = cell_latlon(r["grid_cell"])
    x0, y0 = pyproj.Transformer.from_crs("EPSG:4326", epsg, always_xy=True).transform(lon, lat)
    px, py = x0 - 340, y0 - 340 + 10680  # notebook: left, top
    same = all(r[k]["c"] == b["c"] and r[k]["f"] == b["f"] for k in ("B11", "B01"))
    stats.append((frac(b["c"], 10), frac(b["f"], 10), b["c"] - px, b["f"] - py,
                  int(tile[:2]) == epsg % 100, b["shape"] == [1068, 1068] and b["a"] == 10 and b["e"] == -10, same))
    print(f"{r['grid_cell']:12} {tile:6} {b['crs']:10} {frac(b['c'],10):7.3f} {frac(b['f'],10):7.3f} "
          f"{b['c']-px:8.3f} {b['f']-py:8.3f} {int(tile[:2])==epsg%100!s:5} {b['shape']} a={b['a']} bandsSameOrigin={same}")

s = np.array(stats, dtype=float)
n = len(s)
print(f"\nN={n}")
print(f"origin x not multiple of 10: {(np.abs(s[:,0])>1e-6).sum()}/{n};  y: {(np.abs(s[:,1])>1e-6).sum()}/{n}")
print(f"max |origin - notebook prediction| x={np.abs(s[:,2]).max():.2e} y={np.abs(s[:,3]).max():.2e}")
print(f"CRS zone == product MGRS zone: {int(s[:,4].sum())}/{n}")
print(f"1068x1068 @ exactly 10 m: {int(s[:,5].sum())}/{n};  B11/B01 share B04 origin: {int(s[:,6].sum())}/{n}")
