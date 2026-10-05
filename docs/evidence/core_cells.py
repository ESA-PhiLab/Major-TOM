"""v1 (Core: bottom-left anchor, 1068 px, product CRS) vs v2 (centroid, 1056 px, 60 m snap, Elliot CRS)
on the actual Core-S2L2A cells. All geometry evaluated in each sample's v1 (product) CRS.

Computes per unique (grid_cell, crs) pair, then maps back to samples for per-sample counts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

from compare import HERE, add_cell_geometry, load_index, cell_ring, margins

NDENSE = 9


def signed(code: pd.Series, pat: str, pos: str) -> pd.Series:
    """Parse '123U'-style codes into signed ints."""
    p = code.str.extract(pat)
    return p[0].astype(int) * np.where(p[1] == pos, 1, -1)


def to_crs(src: pd.Series, dst: pd.Series, x: np.ndarray, y: np.ndarray):
    """Transform arrays (rows aligned with src/dst, extra trailing dims ok) from src CRS to dst CRS."""
    ox, oy = x.copy(), y.copy()
    key = src.astype(str) + "|" + dst.astype(str)
    for k, idx in pd.Series(np.arange(len(key))).groupby(key.values).indices.items():
        a, b = k.split("|")
        if a == b:
            continue
        tr = Transformer.from_crs(a, b, always_xy=True)
        ox[idx], oy[idx] = tr.transform(x[idx], y[idx])
    return ox, oy


def square_ring(x0: np.ndarray, y0: np.ndarray, size: float, n: int = NDENSE):
    """Densified axis-aligned square ring from bottom-left (x0, y0)."""
    s = np.linspace(0, 1, n)[:-1]
    u = np.concatenate([s, np.ones_like(s), 1 - s, np.zeros_like(s)])
    v = np.concatenate([np.zeros_like(s), s, np.ones_like(s), 1 - s])
    return x0[:, None] + u * size, y0[:, None] + v * size


def polys(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Array of shapely polygons from ring coordinate arrays (N, K)."""
    coords = np.stack([x, y], -1)
    return shapely.polygons(np.concatenate([coords, coords[:, :1]], 1))


def main() -> None:
    core = pq.read_table(HERE / "core_meta.parquet", columns=["grid_cell", "crs"]).to_pandas()
    print("samples", len(core), "unique cells", core.grid_cell.nunique())
    pairs = core.groupby(["grid_cell", "crs"]).size().rename("n_samples").reset_index()
    print("unique (cell, crs) pairs", len(pairs), " cells with >1 CRS", (pairs.grid_cell.value_counts() > 1).sum())
    pairs["r"] = signed(pairs.grid_cell, r"^(\d+)([UD])", "U")
    pairs["c"] = signed(pairs.grid_cell, r"_(\d+)([RL])$", "R")

    idx = load_index()[["r", "c", "crs", "gx", "gy"]].rename(columns={"crs": "v2_crs"})
    df = pairs.rename(columns={"crs": "v1_crs"}).merge(idx, on=["r", "c"], how="left")
    miss = df[df.v2_crs.isna()]
    print("pairs missing from index:", len(miss), " row range", miss.r.min(), miss.r.max(), " sample", miss.grid_cell.head(5).tolist())
    print("  |r| hist", np.histogram(np.abs(miss.r), bins=[0, 500, 800, 900, 950, 1000])[0])
    df = add_cell_geometry(df.dropna(subset=["v2_crs"]).reset_index(drop=True))

    # convergence in v1 CRS at the cell midpoint
    zone = df.v1_crs.str[-2:].astype(int)
    mlat, mlon = df.lat + 180 / 2004 / 2, df.lon + df.dlon / 2
    df["dcm"] = (mlon - (-183 + 6 * zone) + 180) % 360 - 180
    df["gamma_deg"] = np.degrees(np.arctan(np.tan(np.radians(df.dcm)) * np.sin(np.radians(mlat))))

    # v1 window in v1 CRS (origin rounded to 10 m, as pixels were read)
    four = pd.Series(["EPSG:4326"] * len(df))
    bx, by = to_crs(four, df.v1_crs, df.lon.to_numpy(), df.lat.to_numpy())
    v1x, v1y = np.round((bx - 340) / 10) * 10, np.round((by - 340) / 10) * 10
    v1 = shapely.box(v1x, v1y, v1x + 10680, v1y + 10680)

    # true cell in v1 CRS
    rlon, rlat = cell_ring(df, NDENSE)
    px, py = to_crs(four, df.v1_crs, rlon, rlat)
    cell = polys(px, py)
    m1 = margins((v1x, v1y, v1x + 10680, v1y + 10680), px, py)
    df["v1_contains"] = m1.min(1) >= 0
    df["v1_miss_m"] = np.clip(-m1.min(1), 0, None)
    df["v1_miss_area_pct"] = 100 * shapely.area(shapely.difference(cell, v1)) / shapely.area(cell)

    # v2 window: in its own CRS (containment) and transformed into v1 CRS (crop / overlap / offsets)
    v2x0, v2y0 = df.gx.to_numpy().astype(float), df.gy.to_numpy() - 10560.0
    qx, qy = to_crs(four, df.v2_crs, rlon, rlat)
    m2 = margins((v2x0, v2y0, v2x0 + 10560, v2y0 + 10560), qx, qy)
    df["v2_contains"] = m2.min(1) >= 0
    sx, sy = square_ring(v2x0, v2y0, 10560)
    sx, sy = to_crs(df.v2_crs, df.v1_crs, sx, sy)
    v2 = polys(sx, sy)
    df["crs_differs"] = df.v1_crs != df.v2_crs
    df["v2_in_v1_geom"] = shapely.within(v2, v1)
    df["croppable"] = df.v2_in_v1_geom & ~df.crs_differs
    df["dx"], df["dy"] = sx[:, 0] - v1x, sy[:, 0] - v1y
    df["overlap_frac"] = shapely.area(shapely.intersection(v1, v2)) / shapely.area(v2)

    df["alat_band"] = pd.cut(np.abs(df.lat), np.arange(0, 91, 10), right=False)
    df["dcm_band"] = pd.cut(np.abs(df.dcm), [0, 1, 2, 3, 4, 180], right=False)
    df.drop(columns=["alat_band", "dcm_band"]).to_parquet(HERE / "core_cells_full.parquet")
    cols = ["grid_cell", "lat", "lon", "gamma_deg", "v1_contains", "v2_contains", "croppable",
            "crs_differs", "dx", "dy", "overlap_frac", "v1_crs", "v2_crs", "n_samples",
            "v1_miss_m", "v1_miss_area_pct"]
    df[cols].to_csv(HERE / "core_cells_v1_v2.csv", index=False, float_format="%.4f")

    agg = dict(cells=("grid_cell", "nunique"), samples=("n_samples", "sum"),
               v1_fail=("v1_contains", lambda v: 100 * (~v).mean()),
               v1_miss_area_max=("v1_miss_area_pct", "max"), v1_miss_m_max=("v1_miss_m", "max"),
               v2_fail=("v2_contains", lambda v: 100 * (~v).mean()),
               crop=("croppable", lambda v: 100 * v.mean()),
               crs_diff=("crs_differs", lambda v: 100 * v.mean()),
               dx_med=("dx", "median"), dx_p95=("dx", lambda v: np.percentile(np.abs(v), 95)),
               dy_med=("dy", "median"), dy_p95=("dy", lambda v: np.percentile(np.abs(v), 95)),
               ovl_med=("overlap_frac", lambda v: 100 * v.median()),
               ovl_p5=("overlap_frac", lambda v: 100 * np.percentile(v, 5)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df["all"] = "all"
    for by in ["all", "alat_band", "dcm_band"]:
        print(df.groupby(by, observed=True).agg(**agg).round(2).to_string(), "\n")
    w = df.n_samples
    print("per-sample: v1 fail", int(w[~df.v1_contains].sum()), "/", int(w.sum()),
          f"({100 * w[~df.v1_contains].sum() / w.sum():.2f}%)",
          " croppable", int(w[df.croppable].sum()), f"({100 * w[df.croppable].sum() / w.sum():.2f}%)")
    print("geom v2-in-v1 regardless of CRS:", 100 * df.v2_in_v1_geom.mean())
    print("v1 miss area pct among failures: median", df.loc[~df.v1_contains, "v1_miss_area_pct"].median(),
          " p95", df.loc[~df.v1_contains, "v1_miss_area_pct"].quantile(.95))


if __name__ == "__main__":
    main()
