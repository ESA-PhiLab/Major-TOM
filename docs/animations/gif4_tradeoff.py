"""GIF 4: omission vs duplication: what window size trades away.

Left: two 3x3-cell patches coloured by how many centroid-anchored windows cover each point
(yellow 0 = omission, mint 1, indigo tints 2 and 3+ = duplication). One typical patch (45°N)
and the worst case (81°N, Svalbard, where UTM zones 33X and 35X meet). Right: the same trade-off
for all Core-S2L2A cells, from tradeoff_curve.csv.
Loop: rest at 1056 -> grow to 1152 -> shrink to 960 -> back to 1056 (seam inside the rest).

Usage (from docs/animations):
  python gif4_tradeoff.py --export-curve PATH/v2_misses_cells.parquet    # once: writes tradeoff_curve.csv
  srun --cpus-per-task=4 --mem=16G python gif4_tradeoff.py [--review DIR]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin

from geometry import (DLAT_10KM, R_KM, Cell, cell_polygon, cells_near, transformer, utm_epsg, reproject,
                      window_centroid)
from style import (HERE, INDIGO, INDIGO_TINT, INK, INK_SOFT, MINT, PAPER, YELLOW, ease, mix, new_figure, phase,
                   render_loop, text, use_brand_fonts)

CURVE = HERE / "tradeoff_curve.csv"
SIZES = list(range(960, 1153, 12))
UP, DOWN, BACK = list(range(1068, 1153, 12)), list(range(1140, 959, -12)), list(range(972, 1057, 12))
SCHEDULE = UP + [1152] + DOWN + [960] + BACK        # 1056 -> 1152 (hold) -> 960 (hold) -> 1056
T0, T1 = 0.05, 0.95                       # size changes happen between these loop times
RES_M = 30                                # coverage raster resolution
PATCH_SPECS = [  # title, centre lat, centre lon (a cell centre; its zone is the display CRS), axes rect
    ("Typical: 45°N", 45.04, 13.5, [0.03, 0.22, 0.28, 0.56]),
    ("Worst case: 81°N, Svalbard zone edge", 81.06, 20.96, [0.33, 0.22, 0.28, 0.56]),
]
COLORS = [YELLOW, MINT, INDIGO_TINT, INDIGO]  # windows covering a point: 0, 1, 2, 3+


# ---------- curve data (exported once from the research table) ----------

def export_curve(parquet: Path) -> None:
    """Per window size: % of Core cells not fully covered, and mean extra area per sample."""
    d = pd.read_parquet(parquet, columns=["lat", "dlon", "plain_crs", "side_index", "side_best"])
    zone = d.plain_crs.str[-2:].astype(int)
    no_tiles = (d.lat >= 72) & zone.isin([32, 34, 36])                # Svalbard: no S2 tiles in the plain zone
    side_m = np.where(no_tiles, d.side_index, d.side_best)            # smallest window, best CRS with S2 tiles
    area_km2 = (R_KM ** 2 * np.radians(d.dlon)
                * np.abs(np.sin(np.radians(d.lat + DLAT_10KM)) - np.sin(np.radians(d.lat))))
    rows = [(s, 100 * (side_m > s * 10).mean(), 100 * ((s * 0.01) ** 2 / area_km2).mean() - 100) for s in SIZES]
    pd.DataFrame(rows, columns=["size_px", "cells_not_covered_pct", "extra_area_pct"]).round(4).to_csv(CURVE, index=False)
    print(f"wrote {CURVE.name}")


# ---------- patches ----------

@dataclass
class Patch:
    title: str
    rect: list[float]
    crs: str                                   # display CRS: the centre cell's UTM zone
    windows: list[tuple[Cell, str]]            # every cell whose window can reach the patch, with its CRS
    cells: list[shapely.Polygon]               # the 3x3 displayed cells, in the display CRS
    mask: np.ndarray                           # raster pixels inside the displayed cells
    transform: object                          # raster georeference
    extent: tuple[float, float, float, float]  # raster bounds in km relative to the patch centre
    origin: tuple[float, float]


def build_patch(title: str, lat: float, lon: float, rect: list[float]) -> Patch:
    """Cells, CRSs and raster grid for one patch."""
    crs = utm_epsg(lat, lon)
    shown = [cell_polygon(c, crs) for c in cells_near(lat, lon, rows=1, half_width=1.5)]
    windows = [(c, utm_epsg(c.lat + c.dlat / 2, c.lon + c.dlon / 2))
               for c in cells_near(lat, lon, rows=2, half_width=2.5)]
    x0, y0, x1, y1 = shapely.union_all(shown).buffer(400).bounds
    shape = (int((y1 - y0) / RES_M), int((x1 - x0) / RES_M))
    transform = from_origin(x0, y1, RES_M, RES_M)
    mask = rasterize([(p, 1) for p in shown], out_shape=shape, transform=transform, dtype="uint8") > 0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    extent = ((x0 - cx) / 1e3, (x1 - cx) / 1e3, (y0 - cy) / 1e3, (y1 - cy) / 1e3)
    return Patch(title, rect, crs, windows, shown, mask, transform, extent, (cx, cy))


def coverage_rgb(patch: Patch, size_px: float) -> np.ndarray:
    """RGB image: how many windows of this size cover each point of the displayed cells."""
    shapes = [(reproject(window_centroid(c, crs, size_px), crs, patch.crs), 1) for c, crs in patch.windows]
    count = rasterize(shapes, out_shape=patch.mask.shape, transform=patch.transform,
                      merge_alg=MergeAlg.add, dtype="uint8")
    palette = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in COLORS + [PAPER]], dtype=np.uint8)
    category = np.where(patch.mask, np.minimum(count, 3), 4)
    return palette[category]


def draw_patch(fig, patch: Patch, size_px: float) -> None:
    ax = fig.add_axes(patch.rect)
    ax.imshow(coverage_rgb(patch, size_px), extent=patch.extent, interpolation="antialiased")
    cx, cy = patch.origin
    for poly in patch.cells:
        x, y = poly.exterior.xy
        ax.plot((np.asarray(x) - cx) / 1e3, (np.asarray(y) - cy) / 1e3, color=INK_SOFT, lw=0.5)  # thin: holes sit on borders
    zones = {crs for _, crs in patch.windows}
    if len(zones) > 1:                                                 # mark the UTM zone edge
        lat = np.linspace(80.5, 81.6, 50)
        x, y = transformer("EPSG:4326", patch.crs).transform(np.full_like(lat, 21.0), lat)
        ax.plot((x - cx) / 1e3, (y - cy) / 1e3, color=INK_SOFT, lw=1.2, ls=(0, (4, 3)))
        ax.text(0.03, 0.97, "zone edge (21°E)", transform=ax.transAxes, fontsize=10, color=INK_SOFT,
                va="top", bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
    ax.set_xlim(*patch.extent[:2])
    ax.set_ylim(*patch.extent[2:])
    ax.set_aspect("equal")
    ax.axis("off")


# ---------- charts ----------

def draw_chart(fig, rect: list[float], curve: pd.DataFrame, column: str, size_px: float, symlog: bool) -> None:
    """Small line chart of one curve column with a marker at the current size."""
    ax = fig.add_axes(rect)
    y = np.interp(size_px, curve.size_px, curve[column])
    ax.plot(curve.size_px, curve[column], color=INK, lw=1.8)
    ax.scatter(curve.size_px, curve[column], s=10, color=INK, zorder=3)
    ax.axvline(size_px, color=INDIGO_TINT, lw=1.5, zorder=1)
    ax.scatter([size_px], [y], s=110, color=INDIGO, edgecolors=PAPER, linewidths=1.5, zorder=4)
    if symlog:
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_yticks([0, 0.1, 1, 10, 100], ["0%", "0.1%", "1%", "10%", "100%"])
        ax.set_ylim(0, 100)
    else:                                                              # negative: window smaller than its cell
        ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=1)
        ax.set_yticks([-10, 0, 10, 20, 30], ["-10%", "0%", "10%", "20%", "30%"])
        ax.set_ylim(-10, 35)
    ax.set_xticks([960, 1008, 1056, 1104, 1152])
    ax.set_xlim(950, 1162)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelsize=11)


# ---------- timeline ----------

def size_at(t: float) -> float:
    """Window size at loop time t: eased glide at the start of each schedule slot, then hold."""
    if t < T0 or t >= T1:
        return 1056.0
    x = (t - T0) / (T1 - T0) * len(SCHEDULE)
    k = int(x)
    previous = SCHEDULE[k - 1] if k > 0 else 1056
    return previous + ease((x - k) / 0.3) * (SCHEDULE[k] - previous)


def slot_time(k: int) -> float:
    """Loop time at which schedule slot k starts."""
    return T0 + k * (T1 - T0) / len(SCHEDULE)


def draw(t: float, patches: list[Patch], curve: pd.DataFrame):
    """Frame at loop time t in [0, 1)."""
    size = size_at(t)
    valid = int(round(size / 12) * 12)
    row = curve.set_index("size_px").loc[valid]

    fig = new_figure()
    text(fig, 0.04, 0.925, f"Window: {valid} px ({valid / 100:.2f} km)", size=26, weight=500, va="center")
    for patch in patches:
        draw_patch(fig, patch, size)
        text(fig, patch.rect[0] + patch.rect[2] / 2, 0.81, patch.title, size=14, weight=500, ha="center")
    for i, (label, color) in enumerate(zip(["not covered", "covered once", "twice", "3+ times"], COLORS)):
        x = 0.05 + i * 0.14
        text(fig, x, 0.165, "■", size=18, color=color, va="center")
        text(fig, x + 0.02, 0.165, label, size=13, weight=400, va="center")

    draw_chart(fig, [0.70, 0.54, 0.26, 0.20], curve, "cells_not_covered_pct", size, symlog=True)
    draw_chart(fig, [0.70, 0.18, 0.26, 0.20], curve, "extra_area_pct", size, symlog=False)
    text(fig, 0.70, 0.835, "Omission: Core cells not fully covered", size=14, weight=500)
    text(fig, 0.70, 0.785, f"{row.cells_not_covered_pct:.2f}%", size=20, weight=500, color=INDIGO)
    text(fig, 0.70, 0.475, "Window area beyond its cell (≈ duplication)", size=14, weight=500)
    text(fig, 0.70, 0.425, f"{row.extra_area_pct:+.1f}%", size=20, weight=500, color=INDIGO)

    top = slot_time(len(UP) + 1)                       # descent starts
    low = slot_time(len(UP) + 1 + len(DOWN) + 1)       # final ascent starts
    back = slot_time(len(SCHEDULE))
    captions = [
        (1 - phase(t, 0.02, 0.05) + phase(t, 0.95, 0.98),
         "1056 px: about 12% overlap, and slivers left in 0.3% of cells, mostly in Svalbard."),
        (phase(t, T0 + 0.01, T0 + 0.04) - phase(t, top - 0.03, top),
         "Bigger windows: holes close, but neighbouring samples overlap more."),
        (phase(t, top + 0.005, top + 0.035) - phase(t, low - 0.03, low),
         "Smaller windows: less overlap, but holes open up between samples."),
        (phase(t, low + 0.005, low + 0.035) - phase(t, back - 0.03, back), "Back to the 1056 px default."),
    ]
    for level, caption in captions:
        text(fig, 0.5, 0.055, caption, size=17, weight=400, ha="center", level=level)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-curve", type=Path, help="write tradeoff_curve.csv from v2_misses_cells.parquet and exit")
    p.add_argument("--review", type=Path, help="also save a contact sheet of a few frames to this folder")
    args = p.parse_args()
    if args.export_curve:
        export_curve(args.export_curve)
        return
    use_brand_fonts()
    curve = pd.read_csv(CURVE)
    patches = [build_patch(*spec) for spec in PATCH_SPECS]
    render_loop(lambda t: draw(t, patches, curve), seconds=18, name="gif4_tradeoff", review_dir=args.review,
                review_ts=(0.02, 0.27, 0.45, 0.66, 0.75, 0.97))


if __name__ == "__main__":
    main()
