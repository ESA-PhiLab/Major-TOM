"""Diagnose v2 (centroid, 1056 px, 60 m snap, index CRS) misses on Core cells.

1. cause breakdown, 2. plain-zone CRS rule, 3. minimal window side, 4. neighbour coverage.
"""
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from compare import HERE, load_index, cell_ring, margins
from core_cells import to_crs

DLAT = 180 / 2004
SIZES_PX = [1056, 1068, 1080, 1104, 1152]


def exception_zone(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """UTM exception zone number (Norway/Svalbard) or 0 when the regular rule applies."""
    z = np.zeros(len(lat), dtype=int)
    nor = (lat >= 56) & (lat < 64) & (lon >= 3) & (lon < 12)
    z[nor] = 32
    sv = (lat >= 72) & (lat < 84)
    for lo, hi, zz in [(0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)]:
        z[sv & (lon >= lo) & (lon < hi)] = zz
    return z


def snap60(x: np.ndarray, y: np.ndarray, south: np.ndarray):
    """Snap to the Sentinel-2 60 m lattice (northing offset 40 m in the southern hemisphere)."""
    oy = np.where(south, 40.0, 0.0)
    return np.round(x / 60) * 60, np.round((y - oy) / 60) * 60 + oy


def required_side(df: pd.DataFrame, crs_col: str, rlon: np.ndarray, rlat: np.ndarray) -> np.ndarray:
    """Side (m) of the smallest square centred on the 60 m-snapped centroid that contains the cell."""
    four = pd.Series(["EPSG:4326"] * len(df))
    mlat = (df.lat + DLAT / 2).to_numpy()
    mlon = (df.lon + df.dlon / 2).to_numpy()
    cx, cy = to_crs(four, df[crs_col], mlon, mlat)
    cx, cy = snap60(cx, cy, df[crs_col].str.startswith("EPSG:327").to_numpy())
    px, py = to_crs(four, df[crs_col], rlon, rlat)
    half = np.maximum(np.abs(px - cx[:, None]), np.abs(py - cy[:, None])).max(1)
    return 2 * half


def neighbour_cover(miss: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """For each missed cell: is the sliver outside its own window covered by neighbours' v2 windows?"""
    key = idx.set_index(["r", "c"])[["crs", "gx", "gy"]]
    look = key.to_dict("index")
    out = []
    for row in miss.itertuples():
        tr = Transformer.from_crs("EPSG:4326", row.v2_crs, always_xy=True)
        rlon, rlat = cell_ring(pd.DataFrame({"lon": [row.lon], "lat": [row.lat], "dlon": [row.dlon]}), 17)
        px, py = tr.transform(rlon[0], rlat[0])
        cell = shapely.Polygon(np.c_[px, py])
        own = shapely.box(row.gx, row.gy - 10560, row.gx + 10560, row.gy)
        sliver = cell.difference(own)
        wins, skipped, missing, wins_reproj = [], 0, 0, []
        for dr in (-1, 0, 1):
            r2 = row.r + dr
            lat2 = r2 * DLAT
            nc = int(np.ceil(2 * np.pi * 6378.137 * np.cos(np.radians(lat2)) / 10))
            d2 = 360 / nc
            lo, hi = int(np.floor((row.lon - 0.5 * row.dlon) / d2)), int(np.floor((row.lon + 1.5 * row.dlon) / d2))
            for c2 in range(lo, hi + 1):
                c2w = ((c2 + nc // 2) % nc) - nc // 2
                if dr == 0 and c2w == row.c:
                    continue
                n = look.get((r2, c2w))
                if n is None:
                    missing += 1
                    continue
                box = shapely.box(n["gx"], n["gy"] - 10560, n["gx"] + 10560, n["gy"])
                if n["crs"] == row.v2_crs:
                    wins.append(box)
                    wins_reproj.append(box)
                else:
                    skipped += 1
                    t2 = Transformer.from_crs(n["crs"], row.v2_crs, always_xy=True)
                    bx, by = np.array(box.exterior.coords).T
                    s = np.linspace(0, 1, 17)
                    ex = np.concatenate([bx[i] + s * (bx[i + 1] - bx[i]) for i in range(4)])
                    ey = np.concatenate([by[i] + s * (by[i + 1] - by[i]) for i in range(4)])
                    wins_reproj.append(shapely.Polygon(np.c_[t2.transform(ex, ey)]))
        rem = sliver.difference(shapely.union_all(wins)) if wins else sliver
        rem2 = sliver.difference(shapely.union_all(wins_reproj)) if wins_reproj else sliver
        out.append(dict(grid_cell=row.grid_cell, sliver_m2=sliver.area, uncovered_same_m2=rem.area,
                        uncovered_reproj_m2=rem2.area, n_skipped=skipped, n_missing=missing))
    return pd.DataFrame(out)


def main() -> None:
    df = pd.read_parquet(HERE / "core_cells_full.parquet")
    idx = load_index()
    df = df.merge(idx[["r", "c", "tile"]], on=["r", "c"], how="left")
    mlat, mlon = (df.lat + DLAT / 2).to_numpy(), (df.lon + df.dlon / 2).to_numpy()
    pz = (np.floor((mlon + 180) / 6) % 60 + 1).astype(int)
    df["plain_crs"] = np.where(mlat < 0, "EPSG:327", "EPSG:326") + pd.Series(pz).map("{:02d}".format)
    v2z = df.v2_crs.str[-2:].astype(int).to_numpy()
    exz = exception_zone(mlat, mlon)
    df["cause"] = "iii_regular"
    df.loc[df.tile.str[:2].astype(int).to_numpy() != v2z, "cause"] = "ii_tile_mismatch"
    df.loc[(exz > 0) & (v2z == exz) & (exz != pz), "cause"] = "i_exception"
    dcm = (mlon - (-183 + 6 * v2z) + 180) % 360 - 180
    df["gamma_v2"] = np.degrees(np.arctan(np.tan(np.radians(dcm)) * np.sin(np.radians(mlat))))

    # v2 missed area / metres in the index CRS
    rlon, rlat = cell_ring(df, 9)
    four = pd.Series(["EPSG:4326"] * len(df))
    qx, qy = to_crs(four, df.v2_crs, rlon, rlat)
    x0, y0 = df.gx.to_numpy().astype(float), df.gy.to_numpy() - 10560.0
    m2 = margins((x0, y0, x0 + 10560, y0 + 10560), qx, qy)
    df["v2_miss_m"] = np.clip(-m2.min(1), 0, None)
    cellp = shapely.polygons(np.stack([np.c_[qx, qx[:, :1]], np.c_[qy, qy[:, :1]]], -1))
    win = shapely.box(x0, y0, x0 + 10560, y0 + 10560)
    df["v2_miss_area_pct"] = 100 * shapely.area(shapely.difference(cellp, win)) / shapely.area(cellp)

    # 1. cause breakdown
    miss = df[~df.v2_contains]
    print("v2 misses", len(miss), "of", len(df))
    print("all cells per cause:", df.cause.value_counts().to_dict())
    print(miss.groupby("cause").agg(n=("grid_cell", "size"), samples=("n_samples", "sum"),
                                    g_min=("gamma_v2", lambda v: np.abs(v).min()), g_max=("gamma_v2", lambda v: np.abs(v).max()),
                                    lat_min=("lat", lambda v: np.abs(v).min()),
                                    area_med=("v2_miss_area_pct", "median"), area_p95=("v2_miss_area_pct", lambda v: v.quantile(.95)),
                                    area_max=("v2_miss_area_pct", "max"), m_max=("v2_miss_m", "max")).round(3).to_string())
    print("fail rate per cause (%):", (100 * (~df.v2_contains).groupby(df.cause).mean()).round(3).to_dict())

    # 2 + 3. required side in index CRS and plain-zone CRS
    df["side_index"] = required_side(df, "v2_crs", rlon, rlat)
    df["side_plain"] = required_side(df, "plain_crs", rlon, rlat)
    df["side_best"] = np.minimum(df.side_index, df.side_plain)
    print("consistency: side_index>10560 vs ~v2_contains agree:", np.mean((df.side_index > 10560) == ~df.v2_contains))
    m = ~df.v2_contains
    ne = m & (df.cause != "i_exception")
    print("non-exception misses:", ne.sum(), " fixed by plain-zone CRS (side_plain<=10560):",
          int((df.side_plain[ne] <= 10560).sum()), " plain==index CRS among them:", int((df.plain_crs[ne] == df.v2_crs[ne]).sum()))
    ex = m & (df.cause == "i_exception")
    print("exception misses:", ex.sum(), " fixed by plain zone:", int((df.side_plain[ex] <= 10560).sum()),
          " plain-zone tiles used:", df.loc[ex, "plain_crs"].value_counts().head(8).to_dict())
    df["exc_region"] = exz > 0
    for name, sub in [("all", df), ("exception cause", df[df.cause == "i_exception"]), ("exception region", df[df.exc_region])]:
        for col in ["side_index", "side_best"]:
            v = sub[col] / 10
            print(f"{name:17s} {col:10s} n={len(sub)} side px p99={np.percentile(v, 99):.1f} p99.9={np.percentile(v, 99.9):.1f} "
                  f"max={v.max():.1f} | % missing at", {s: round(100 * np.mean(v > s), 4) for s in SIZES_PX})

    # 4. neighbour coverage
    nb = neighbour_cover(df[m], idx)
    print("neighbour check n", len(nb), " cells with any different-CRS neighbour:", (nb.n_skipped > 0).sum(),
          " neighbours missing from index (cells):", (nb.n_missing > 0).sum())
    same_only = nb[nb.n_skipped == 0]
    print("same-CRS-only cells:", len(same_only), " sliver covered (<1 m2 left):", (same_only.uncovered_same_m2 < 1).sum())
    print("all cells, reprojected neighbours: covered", (nb.uncovered_reproj_m2 < 1).sum(), "/", len(nb),
          " max uncovered m2", nb.uncovered_reproj_m2.max().round(1))
    df.drop(columns=["alat_band", "dcm_band"], errors="ignore").to_parquet(HERE / "v2_misses_cells.parquet")
    nb.merge(df[["grid_cell", "cause", "lat", "lon"]], on="grid_cell").to_csv(HERE / "v2_misses_neighbours.csv", index=False)
    print(nb.merge(df[["grid_cell", "cause"]], on="grid_cell").groupby("cause").apply(
        lambda g: pd.Series(dict(n=len(g), covered=(g.uncovered_reproj_m2 < 1).sum(), skipped_cells=(g.n_skipped > 0).sum()))))


if __name__ == "__main__":
    main()
