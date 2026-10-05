"""Join HF Core-S2L2A metadata to the Elliot sample: CRS agreement, Core coverage in its own CRS,
Core centre_lat/lon definition, and the Elliot y-snap rule."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

from compare import HERE, project, cell_ring, margins

s = pd.read_parquet(HERE / "sample_results.parquet")

# --- y snap rule
for hemi, m in [("N", s["lat"] >= 0), ("S", s["lat"] < 0)]:
    g = s[m]
    print(hemi, "gy mod 60 values:", np.unique(g["gy"] % 60)[:10], " gx mod 60:", np.unique(g["gx"] % 60)[:5],
          " round60(ry)==gy:", np.mean(np.round(g["ry"] / 60) * 60 == g["gy"]),
          " round to 60 grid offset 20/40:",
          [np.mean(np.round((g["ry"] - o) / 60) * 60 + o == g["gy"]) for o in (20, 40)])

core = pq.read_table(HERE / "core_meta.parquet", columns=["grid_cell", "crs", "centre_lat", "centre_lon", "product_id"]).to_pandas()
core = core.drop_duplicates("grid_cell")
core["r"] = core["grid_cell"].str.extract(r"^(\d+)([UD])").apply(lambda x: int(x[0]) * (1 if x[1] == "U" else -1), axis=1)
core["c"] = core["grid_cell"].str.extract(r"_(\d+)([RL])$").apply(lambda x: int(x[0]) * (1 if x[1] == "R" else -1), axis=1)
j = s.merge(core.rename(columns={"crs": "core_crs"}), on=["r", "c"], how="inner")
print("joined", len(j))
j["core_tile"] = j["product_id"].str.extract(r"_T(\d\d[A-Z]{3})_")[0]
print("CRS match Core vs Elliot:", np.mean(j["core_crs"] == j["crs"]), " tile match:", np.mean(j["core_tile"] == j["tile"]))

# Core centre definition: centre of BL-anchored window in Core CRS, back to lat/lon
bx, by = project(j, j["lon"].to_numpy(), j["lat"].to_numpy(), crs_col="core_crs")
lon_c = np.full(len(j), np.nan); lat_c = np.full(len(j), np.nan)
for crs, idx in j.groupby("core_crs").indices.items():
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_c[idx], lat_c[idx] = tr.transform(bx[idx] + 5000, by[idx] + 5000)
mlat, mlon = j["lat"] + 180 / 2004 / 2, j["lon"] + j["dlon"] / 2
print("centre_lat - window-centre lat |max| (m):", np.nanmax(np.abs(j["centre_lat"] - lat_c)) * 111000,
      " centre_lon diff max deg:", np.nanmax(np.abs(((j["centre_lon"] - lon_c) + 180) % 360 - 180)))
print("centre vs latlon midpoint (m) median/p99:",
      np.nanpercentile(np.abs(j["centre_lat"] - mlat) * 111000, [50, 99]))

# Core coverage in its own CRS
rlon, rlat = cell_ring(j)
px, py = project(j, rlon, rlat, crs_col="core_crs")
m = margins((bx - 340, by - 340, bx + 10340, by + 10340), px, py)
print("Core own-CRS contained:", np.mean(m.min(1) >= 0), " p1 min margin:", np.percentile(m.min(1), 1))
same = (j["core_crs"] == j["crs"]).to_numpy()
print("  same-CRS subset contained:", np.mean(m.min(1)[same] >= 0), " diff-CRS subset:", np.mean(m.min(1)[~same] >= 0),
      " Elliot contained (all joined):", j["ell_snap60_ok"].mean())
print("E in C (same CRS subset):", j.loc[same, "e_in_c"].mean())
j["alat"] = pd.cut(np.abs(j["lat"]), [0, 20, 40, 55, 65, 75, 85], include_lowest=True)
j["core_own_ok"] = m.min(1) >= 0
print(j.groupby("alat", observed=True).agg(n=("id", "size"), core_own_ok=("core_own_ok", "mean"),
                                           ell_ok=("ell_snap60_ok", "mean"), e_in_c=("e_in_c", "mean"),
                                           crs_match=("crs", lambda v: np.mean(v == j.loc[v.index, "core_crs"]))).round(3))
