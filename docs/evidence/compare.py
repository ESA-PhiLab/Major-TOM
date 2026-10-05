"""Compare Core (bottom-left anchored, 1068 px) vs Elliot (centroid anchored, 1056 px) windows.

Reads the source.coop index (local copy), reconstructs Major TOM cells analytically,
and measures cell coverage + mutual containment in the Elliot-assigned UTM CRS.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

HERE = Path(__file__).parent
R_EQ, D = 6378.137, 10.0
NR = int(np.ceil(np.pi * R_EQ / D))
DLAT = 180.0 / NR
NSAMP = 400_000
rng = np.random.default_rng(0)


def load_index() -> pd.DataFrame:
    """Load needed columns of the Elliot index and parse ids to signed row/col."""
    cols = ["id", "majortom:crs", "majortom:mgrs_tile", "majortom:geotransform",
            "majortom:geotransform_raw", "geometry"]
    t = pq.read_table(HERE / "global.parquet", columns=cols)
    df = pd.DataFrame({"id": t["id"].to_numpy(zero_copy_only=False),
                       "crs": t["majortom:crs"].to_numpy(zero_copy_only=False),
                       "tile": t["majortom:mgrs_tile"].to_numpy(zero_copy_only=False)})
    gt = np.array(t["majortom:geotransform"].combine_chunks().flatten()).reshape(-1, 6)
    gr = np.array(t["majortom:geotransform_raw"].combine_chunks().flatten()).reshape(-1, 6)
    df["gx"], df["gy"], df["rx"], df["ry"] = gt[:, 0], gt[:, 3], gr[:, 0], gr[:, 3]
    df["px_x"], df["px_y"] = gr[:, 1], gr[:, 5]
    pts = shapely.from_wkb(t["geometry"].to_numpy(zero_copy_only=False))
    df["clon"], df["clat"] = shapely.get_x(pts), shapely.get_y(pts)
    p = df["id"].str.extract(r"MT10km_(\d+)([UD])_(\d+)([RL])")
    df["r"] = p[0].astype(int) * np.where(p[1] == "U", 1, -1)
    df["c"] = p[2].astype(int) * np.where(p[3] == "R", 1, -1)
    return df


def add_cell_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Bottom-left lat/lon and cell spacing from the Major TOM grid rules."""
    df["lat"] = df["r"] * DLAT
    nc = np.ceil(2 * np.pi * R_EQ * np.cos(np.radians(df["lat"])) / D)
    df["dlon"] = 360.0 / nc
    df["lon"] = df["c"] * df["dlon"]
    return df


def project(df: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, crs_col: str = "crs"):
    """Project lon/lat arrays (rows aligned with df, extra trailing dims ok) into each row's CRS."""
    x = np.full(lon.shape, np.nan)
    y = np.full(lon.shape, np.nan)
    for crs, idx in df.groupby(crs_col).indices.items():
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x[idx], y[idx] = tr.transform(lon[idx], lat[idx])
    return x, y


def cell_ring(df: pd.DataFrame, n: int = 9):
    """Densified lat/lon rectangle of each cell, shape (N, 4*(n-1))."""
    s = np.linspace(0, 1, n)[:-1]
    u = np.concatenate([s, np.ones_like(s), 1 - s, np.zeros_like(s)])
    v = np.concatenate([np.zeros_like(s), s, np.ones_like(s), 1 - s])
    lon = df["lon"].to_numpy()[:, None] + u[None] * df["dlon"].to_numpy()[:, None]
    lat = df["lat"].to_numpy()[:, None] + v[None] * DLAT
    return lon, lat


def margins(win: tuple, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Margins (left, right, bottom, top) in m of cell ring inside window (L, B, R, T)."""
    L, B, Rr, T = win
    return np.stack([px.min(1) - L, Rr - px.max(1), py.min(1) - B, T - py.max(1)], 1)


def main() -> None:
    df = add_cell_geometry(load_index())
    print("rows", len(df))
    zone = df["crs"].str[-2:].astype(int)
    df["cm"] = -183 + 6 * zone
    dl = (df["lon"] + df["dlon"] / 2 - df["cm"] + 180) % 360 - 180
    df["dcm"] = dl
    df["gamma"] = np.degrees(np.arctan(np.tan(np.radians(dl)) * np.sin(np.radians(df["lat"] + DLAT / 2))))

    # --- centroid definition
    mlat, mlon = df["lat"] + DLAT / 2, df["lon"] + df["dlon"] / 2
    dlon_c = ((df["clon"] - mlon + 180) % 360) - 180
    print("centroid - latlon midpoint: |dlat| max", np.abs(df["clat"] - mlat).max(),
          "|dlon| max", np.abs(dlon_c).max())
    print("pixel sizes", df["px_x"].unique()[:5], df["px_y"].unique()[:5])

    s = df.sample(NSAMP, random_state=0).reset_index(drop=True)
    cx, cy = project(s, s["clon"].to_numpy(), s["clat"].to_numpy())
    print("raw x0 - (cx-5280): abs max", np.nanmax(np.abs(s["rx"] - (cx - 5280))),
          " raw y0 - (cy+5280): abs max", np.nanmax(np.abs(s["ry"] - (cy + 5280))))
    for name, f in [("round60", lambda v: np.round(v / 60) * 60), ("floor60", lambda v: np.floor(v / 60) * 60)]:
        print(name, "x match", np.mean(f(s["rx"]) == s["gx"]), "y match", np.mean(f(s["ry"]) == s["gy"]))
    snapdx, snapdy = s["gx"] - s["rx"], s["gy"] - s["ry"]
    print("snap shift x range", snapdx.min(), snapdx.max(), " y", snapdy.min(), snapdy.max())

    # --- CRS assignment
    bl_zone = (np.floor((s["lon"] + 180) / 6) % 60 + 1).astype(int)
    c_zone = (np.floor((s["clon"] + 180) / 6) % 60 + 1).astype(int)
    ez = s["crs"].str[-2:].astype(int)
    print("Elliot zone == centroid lon-zone:", np.mean(ez == c_zone), " == BL lon-zone:", np.mean(ez == bl_zone))
    print("Elliot tile zone == crs zone:", np.mean(s["tile"].str[:2].astype(int) == ez))

    # --- windows in Elliot CRS
    bx, by = project(s, s["lon"].to_numpy(), s["lat"].to_numpy())
    rlon, rlat = cell_ring(s)
    px, py = project(s, rlon, rlat)
    core_raw = (bx - 340, by - 340, bx + 10340, by + 10340)
    cx10, cy10 = np.round((bx - 340) / 10) * 10, np.round((by - 340) / 10) * 10
    core_snap = (cx10, cy10, cx10 + 10680, cy10 + 10680)
    ell_raw = (s["rx"].to_numpy(), s["ry"].to_numpy() - 10560, s["rx"].to_numpy() + 10560, s["ry"].to_numpy())
    ell_snap = (s["gx"].to_numpy(), s["gy"].to_numpy() - 10560, s["gx"].to_numpy() + 10560, s["gy"].to_numpy())
    res = {}
    for name, w in [("core_raw", core_raw), ("core_snap10", core_snap), ("ell_raw", ell_raw), ("ell_snap60", ell_snap)]:
        m = margins(w, px, py)
        res[name] = m
        s[f"{name}_min"] = m.min(1)
        s[f"{name}_ok"] = m.min(1) >= 0
        print(f"{name}: contained {np.mean(m.min(1) >= 0):.4f}; margin pctl(1,50,99) L/R/B/T:",
              [np.percentile(m[:, i], [1, 50, 99]).round(0).tolist() for i in range(4)],
              "min", m.min(0).round(0).tolist())
    # Elliot inside Core (snapped / snapped)
    e, c = ell_snap, core_snap
    d = np.stack([e[0] - c[0], c[2] - e[2], e[1] - c[1], c[3] - e[3]], 1)
    s["e_in_c"] = d.min(1) >= 0
    print("Elliot(snap60) inside Core(snap10):", s["e_in_c"].mean(),
          " per-side fail L/R/B/T:", (d < 0).mean(0).round(4).tolist())
    print("margins E in C pctl(1,50,99):", [np.percentile(d[:, i], [1, 50, 99]).round(0).tolist() for i in range(4)])
    er = ell_raw
    d2 = np.stack([er[0] - core_raw[0], core_raw[2] - er[2], er[1] - core_raw[1], core_raw[3] - er[3]], 1)
    print("Elliot(raw) inside Core(raw):", (d2.min(1) >= 0).mean())
    # origin offsets (bottom-left corners)
    ox, oy = e[0] - c[0], e[1] - c[1]
    s["ox"], s["oy"] = ox, oy
    print("origin offset Elliot-Core (BL corners) x pctl", np.percentile(ox, [0, 1, 50, 99, 100]).round(0),
          " y", np.percentile(oy, [0, 1, 50, 99, 100]).round(0))

    # --- stratified tables
    s["alat"] = pd.cut(np.abs(s["lat"]), [0, 20, 40, 55, 65, 75, 85], include_lowest=True)
    s["adcm"] = pd.cut(np.abs(s["dcm"]), [0, 1, 2, 3, 4, 6, 180], include_lowest=True)
    agg = dict(n=("id", "size"), gamma_med=("gamma", lambda v: np.median(np.abs(v))),
               core_ok=("core_snap10_ok", "mean"), core_min_p1=("core_snap10_min", lambda v: np.percentile(v, 1)),
               ell_ok=("ell_snap60_ok", "mean"), ell_min_p1=("ell_snap60_min", lambda v: np.percentile(v, 1)),
               e_in_c=("e_in_c", "mean"), ox_med=("ox", "median"), oy_med=("oy", "median"))
    pd.set_option("display.width", 250)
    print(s.groupby("alat", observed=True).agg(**agg).round(3))
    print(s.groupby("adcm", observed=True).agg(**agg).round(3))
    # ~2000-cell stratified subsample
    strat = s.groupby(["alat", "adcm"], observed=True, group_keys=False).apply(lambda g: g.sample(min(len(g), 60), random_state=0))
    print("stratified n", len(strat))
    print(strat.groupby(["alat", "adcm"], observed=True).agg(**agg).round(2).to_string())
    s.drop(columns=["alat", "adcm"]).to_parquet(HERE / "sample_results.parquet")


if __name__ == "__main__":
    main()
