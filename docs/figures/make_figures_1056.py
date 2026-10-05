"""Figures: behaviour of 1056 px centroid-anchored windows at their extremes.

Inputs come from the research scripts in docs/evidence (not committed, regenerate there):
  v2_misses_cells.parquet  one row per Core-S2L2A cell with its v2 window (gx, gy, v2_crs)
  global.parquet           ELLIOT index (source.coop/major-tom/index), for neighbouring windows

Usage:
  srun --cpus-per-task=8 --mem=32G python make_figures_1056.py --data <dir with inputs>
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin

R_KM = 6378.137
N_ROWS = 2004                      # ceil(pi * R / 10 km)
DLAT = 180 / N_ROWS
SIDE_M = 10560                     # 1056 px at 10 m
OUT = Path(__file__).parent
CAUSE_STYLE = {"i_exception": ("Exception zone (Norway/Svalbard)", "#c0392b"),
               "ii_tile_mismatch": ("Index tile/CRS mismatch", "#e67e22"),
               "iii_regular": ("Regular zone", "#2980b9")}


# ---------- grid geometry ----------

def parse_id(ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Signed (row, col) from ids like 'MT10km_0012D_0034L' or '12D_34L'."""
    parts = ids.str.replace("MT10km_", "", regex=False).str.split("_", expand=True)
    sign = lambda s, pos: np.where(s.str[-1] == pos, 1, -1) * s.str[:-1].astype(int)
    return sign(parts[0], "U").to_numpy(), sign(parts[1], "R").to_numpy()


def cell_lonlat(r: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bottom-left lat, lon and column width (deg) of cells, per the grid.py construction."""
    lat = r * DLAT
    n_cols = np.ceil(2 * math.pi * R_KM * np.cos(np.radians(lat)) / 10)
    dlon = 360 / n_cols
    return lat, c * dlon, dlon


def cell_polygon(lat: float, lon: float, dlon: float, crs: str, n: int = 33) -> shapely.Polygon:
    """Lat/lon cell rectangle, densified and projected to `crs`."""
    s = np.linspace(0, 1, n)
    u = np.r_[s, np.ones(n), s[::-1], np.zeros(n)]
    v = np.r_[np.zeros(n), s, np.ones(n), s[::-1]]
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shapely.Polygon(np.c_[tr.transform(lon + u * dlon, lat + v * DLAT)])


def window_polygon(gx: float, gy: float, src: str, dst: str, n: int = 17) -> shapely.Polygon:
    """1056 px window with top-left (gx, gy) in `src`, reprojected (densified) to `dst`."""
    box = shapely.box(gx, gy - SIDE_M, gx + SIDE_M, gy)
    if src == dst:
        return box
    box = shapely.segmentize(box, SIDE_M / n)
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(box, lambda xy: np.c_[tr.transform(xy[:, 0], xy[:, 1])])


def cell_area_km2(lat: np.ndarray, dlon: np.ndarray) -> np.ndarray:
    """Spherical area of lat/lon cells."""
    return R_KM**2 * np.radians(dlon) * np.abs(np.sin(np.radians(lat + DLAT)) - np.sin(np.radians(lat)))


# ---------- data ----------

def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = pd.read_parquet(data / "v2_misses_cells.parquet")
    t = pq.read_table(data / "global.parquet", columns=["id", "majortom:crs", "majortom:geotransform"])
    gt = np.array(t["majortom:geotransform"].combine_chunks().flatten()).reshape(-1, 6)
    idx = pd.DataFrame({"id": t["id"].to_numpy(zero_copy_only=False),
                        "crs": t["majortom:crs"].to_numpy(zero_copy_only=False),
                        "gx": gt[:, 0], "gy": gt[:, 3]})
    idx["r"], idx["c"] = parse_id(idx["id"])
    idx["lat"], idx["lon"], idx["dlon"] = cell_lonlat(idx["r"].to_numpy(), idx["c"].to_numpy())
    return core, idx


def neighbourhood(idx: pd.DataFrame, focus: pd.Series, rows: int = 2, cols: float = 2.5) -> pd.DataFrame:
    """Index cells within `rows` rows and `cols` column widths of the focus cell."""
    near = idx[(idx.r - focus.r).abs() <= rows]
    return near[(near.lon - focus.lon).abs() <= cols * focus.dlon]


# ---------- figures ----------

def sliver_zoom(f: pd.Series, pad: float = 400.0) -> tuple[float, float, float, float] | None:
    """Bounds (x0, y0, x1, y1) around the largest part of the cell outside its own window."""
    cell = cell_polygon(f.lat, f.lon, f.dlon, f.v2_crs)
    own = shapely.box(f.gx, f.gy - SIDE_M, f.gx + SIDE_M, f.gy)
    parts = [g for g in getattr(cell.difference(own), "geoms", [cell.difference(own)]) if not g.is_empty]
    if not parts:
        return None
    return max(parts, key=lambda g: g.area).buffer(pad).bounds


def add_inset(ax, box: tuple, draw) -> None:
    """Zoomed inset in the upper-left of `ax`, redrawn with `draw(inset_ax)`, framed on `box`."""
    ins = ax.inset_axes([0.03, 0.55, 0.42, 0.42])
    draw(ins)
    x0, y0, x1, y1 = box
    ins.set_xlim(x0, x1); ins.set_ylim(y0, y1); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    ax.indicate_inset_zoom(ins, edgecolor="0.3")

def pick_cells(core: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Typical mid-latitude cell, worst regular-zone miss, worst exception-zone miss."""
    ok = core[core.v2_contains & (core.lat.between(44, 46))]
    typical = ok.iloc[(ok.gamma_v2.abs() - ok.gamma_v2.abs().median()).abs().argmin()]
    miss = core[~core.v2_contains]
    reg = miss[miss.cause == "iii_regular"].sort_values("v2_miss_m").iloc[-1]
    exc = miss[miss.cause == "i_exception"].sort_values("v2_miss_m").iloc[-1]
    return [("Typical (45°)", typical), ("Worst regular zone", reg), ("Worst exception zone", exc)]


def fig_cells(core: pd.DataFrame, idx: pd.DataFrame) -> None:
    """Focus cell, its window, neighbours' windows, missed sliver and uncovered hole."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (title, f) in zip(axes, pick_cells(core)):
        crs = f.v2_crs
        cell = cell_polygon(f.lat, f.lon, f.dlon, crs)
        own = shapely.box(f.gx, f.gy - SIDE_M, f.gx + SIDE_M, f.gy)
        near = neighbourhood(idx, f)
        others = [window_polygon(n.gx, n.gy, n.crs, crs) for n in near.itertuples()
                  if not (n.r == f.r and n.c == f.c)]
        near_cells = [cell_polygon(n.lat, n.lon, n.dlon, crs) for n in near.itertuples()]
        other_crs = [n.crs for n in near.itertuples() if not (n.r == f.r and n.c == f.c)]
        sliver = cell.difference(own)
        hole = sliver.difference(shapely.union_all(others)) if others else sliver

        def draw(a, near_cells=near_cells, others=others, other_crs=other_crs,
                 sliver=sliver, hole=hole, cell=cell, own=own, crs=crs):
            for c in near_cells:
                a.plot(*c.exterior.xy, color="0.75", lw=0.6)
            for w, wc in zip(others, other_crs):
                a.plot(*w.exterior.xy, color="#27ae60" if wc == crs else "#8e44ad", lw=0.8, ls="--")
            for geom, col in [(sliver, "#f5b7b1"), (hole, "#c0392b")]:
                for g in getattr(geom, "geoms", [geom]):
                    if not g.is_empty:
                        a.fill(*g.exterior.xy, color=col, lw=0)
            a.plot(*cell.exterior.xy, color="k", lw=1.6)
            a.plot(*own.exterior.xy, color="#2980b9", lw=2)

        draw(ax)
        pad = 2500
        x0, y0, x1, y1 = own.buffer(pad).bounds
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_title(f"{title}: {f.grid_cell} ({f.lat:.1f}°, {f.lon:.1f}°)\n"
                     f"{crs}, γ = {f.gamma_v2:.2f}°, miss = {f.v2_miss_m:.0f} m "
                     f"({f.v2_miss_area_pct:.2f}% of cell)", fontsize=10)
        ax.ticklabel_format(style="plain", useOffset=False); ax.tick_params(labelsize=7)
        zoom = sliver_zoom(f)
        if zoom:
            add_inset(ax, zoom, draw)
    handles = [plt.Line2D([], [], color="k", lw=1.6, label="Cell (lat/lon)"),
               plt.Line2D([], [], color="#2980b9", lw=2, label="1056 px window"),
               plt.Line2D([], [], color="#27ae60", ls="--", label="Neighbour window, same CRS"),
               plt.Line2D([], [], color="#8e44ad", ls="--", label="Neighbour window, other CRS (reprojected)"),
               plt.Rectangle((0, 0), 1, 1, color="#f5b7b1", label="Cell outside own window"),
               plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="Not covered by any window")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("1056 px centroid-anchored windows: typical and extreme cells (UTM metres)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "fig1_cells_1056.png", dpi=150)


def fig_map(core: pd.DataFrame) -> None:
    """Where 1056 px windows miss part of their cell, by cause."""
    miss = core[~core.v2_contains]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [2, 1, 1]})
    views = [("All Core cells", (-180, 180, -90, 90)),
             ("Arctic", (-30, 60, 55, 85)),
             ("Antarctic", (-180, 180, -86, -60))]
    for ax, (title, (x0, x1, y0, y1)) in zip(axes, views):
        ax.scatter(core.lon, core.lat, s=0.05, c="0.85", rasterized=True)
        for cause, (label, col) in CAUSE_STYLE.items():
            m = miss[miss.cause == cause]
            ax.scatter(m.lon, m.lat, s=1.5, c=col, label=f"{label} ({len(m):,})", rasterized=True)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_title(title)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    axes[0].legend(loc="lower left", fontsize=8, markerscale=6)
    fig.suptitle(f"Core cells whose 1056 px window misses part of the cell: "
                 f"{len(miss):,} of {len(core):,} ({100 * len(miss) / len(core):.2f}%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_miss_map_1056.png", dpi=150)


def fig_tradeoff(core: pd.DataFrame) -> None:
    """Omission vs duplication as a function of window size."""
    sizes = np.arange(1008, 1201, 12)                     # only sizes whose corner lands on the 60 m lattice
    area = cell_area_km2(core.lat.to_numpy(), core.dlon.to_numpy())
    dup = [100 * (((s * 0.01) ** 2) / area).mean() - 100 for s in sizes]
    pz = core.plain_crs.str[-2:].astype(int)              # plain UTM zone number
    no_tiles = (core.lat >= 72) & pz.isin([32, 34, 36])   # Svalbard: no S2 tiles in the plain zone
    side_avail = np.where(no_tiles, core.side_index, core.side_best)
    miss_index = [100 * (core.side_index > s * 10).mean() for s in sizes]
    miss_best = [100 * (core.side_best > s * 10).mean() for s in sizes]
    miss_avail = [100 * (side_avail > s * 10).mean() for s in sizes]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    curves = [ax1.plot(sizes, miss_index, color="#c0392b", label="Cells not contained (index CRS)")[0],
              ax1.plot(sizes, miss_avail, color="#c0392b", ls=":", label="Cells not contained (best CRS with S2 tiles)")[0],
              ax1.plot(sizes, miss_best, color="#c0392b", ls="--", alpha=0.6,
                       label="Cells not contained (best CRS, ignores tile availability)")[0]]
    ax1.set_yscale("symlog", linthresh=0.01); ax1.set_ylabel("% of Core cells not contained")
    ax1.set_xlabel("Window side (px at 10 m, multiples of 12)")
    ax2 = ax1.twinx()
    curves += ax2.plot(sizes, dup, color="#2980b9", label="Window area beyond cell (≈ duplication)")
    ax2.set_ylabel("% extra area per sample", color="#2980b9")
    for s in (1056, 1068, 1104, 1152):
        ax1.axvline(s, color="0.8", lw=0.8); ax1.text(s, 30, str(s), rotation=90, fontsize=8, va="top")
    ax1.legend(curves, [c.get_label() for c in curves], fontsize=8, loc="upper right")
    ax1.set_title("Omission vs duplication: Major TOM window size trade-off (Core-S2L2A cells)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_tradeoff.png", dpi=150)


def coverage_count(focus: pd.Series, idx: pd.DataFrame, res: float = 30.0) -> tuple[np.ndarray, tuple, list]:
    """Number of windows covering each point around the focus cell, in the focus CRS."""
    crs = focus.v2_crs
    near = neighbourhood(idx, focus, rows=4, cols=5)
    cells = [cell_polygon(n.lat, n.lon, n.dlon, crs) for n in near.itertuples()]
    wins = [window_polygon(n.gx, n.gy, n.crs, crs) for n in near.itertuples()]
    inner = neighbourhood(idx, focus, rows=2, cols=2.5)
    x0, y0, x1, y1 = shapely.union_all([cell_polygon(n.lat, n.lon, n.dlon, crs)
                                        for n in inner.itertuples()]).bounds
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    tf = from_origin(x0, y1, res, res)
    count = rasterize([(g, 1) for g in wins], out_shape=(h, w), transform=tf,
                      merge_alg=MergeAlg.add,
                      dtype="uint8")
    return count, (x0, x1, y0, y1), cells


def fig_coverage(core: pd.DataFrame, idx: pd.DataFrame) -> None:
    """Omission (0) and duplication (2+) around the extreme cells."""
    picks = pick_cells(core)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout="constrained")
    cmap = matplotlib.colors.ListedColormap(["#c0392b", "#ecf0f1", "#aed6f1", "#5dade2", "#1f618d"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 9.5], cmap.N)
    for ax, (title, f) in zip(axes, picks):
        count, (x0, x1, y0, y1), cells = coverage_count(f, idx)
        im = ax.imshow(count, extent=(x0, x1, y0, y1), cmap=cmap, norm=norm, interpolation="nearest")
        for c in cells:
            ax.plot(*c.exterior.xy, color="k", lw=0.4)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        zoom = sliver_zoom(f)
        if zoom:
            def draw(a, count=count, ext=(x0, x1, y0, y1), cells=cells):
                a.imshow(count, extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
                for c in cells:
                    a.plot(*c.exterior.xy, color="k", lw=0.8)
            add_inset(ax, zoom, draw)
        share = {k: 100 * (count == k).mean() for k in (0, 1)}
        ax.set_title(f"{title}: {f.grid_cell}\nnot covered {share[0]:.2f}%, covered once {share[1]:.0f}%, "
                     f"2+ times {100 - share[0] - share[1]:.0f}%", fontsize=10)
        ax.ticklabel_format(style="plain", useOffset=False); ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3, 5], shrink=0.7)
    cb.ax.set_yticklabels(["0 (omission)", "1", "2", "3", "4+"]); cb.set_label("Windows covering point")
    fig.suptitle("Coverage count of 1056 px windows around the cells in Fig. 1 (black: cell outlines)")
    fig.savefig(OUT / "fig4_coverage_1056.png", dpi=150)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    core, idx = load(args.data)
    fig_cells(core, idx)
    fig_map(core)
    fig_tradeoff(core)
    fig_coverage(core, idx)


if __name__ == "__main__":
    main()
