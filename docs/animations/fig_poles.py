"""Static slide: Major TOM 10 km cells at both poles, with a hypothetical 1056 px instrument.

Cells of the last rows around each pole (official row/column rule), windows of 1056 px centred on
each cell's lat/lon midpoint (the v2 rule) in polar stereographic coordinates (UPS), since UTM does not
exist here. Colours: how many windows cover each 60 m pixel (yellow 0, mint 1, indigo 2+).
Caveats are written on the slide. Output: out/poles_1056.png

Usage (from docs/animations):  srun --cpus-per-task=4 --mem=16G python fig_poles.py
"""
from __future__ import annotations

import textwrap

import numpy as np
import shapely
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin

from geometry import cell_10km, cell_polygon, window_centroid
from style import INDIGO, INDIGO_TINT, INK, INK_SOFT, MINT, OUT, PAPER, YELLOW, new_figure, text, use_brand_fonts

HALF_KM, RES_M = 32, 60
POLES = [  # title, CRS, rows nearest the pole first
    ("North pole", "EPSG:32661", list(range(1001, 994, -1))),
    ("South pole", "EPSG:32761", list(range(-1002, -995))),
]
CAVEATS = [
    ("Not in the default grid", "Grid() stops at ±85° (latitude_range); these rows exist only on request."),
    ("No UTM", "UTM ends at 84°N and 80°S. Windows here use polar stereographic (UPS) coordinates."),
    ("No Sentinel-2", "Systematic imaging stops at 82.8°N; Antarctica only on request. Hence: hypothetical."),
    ("Cells turn into wedges", "North: 7 wedge cells share the pole (row 1001U). South: one disc cell, 1002D."),
    ("Midpoint ≠ centre", "The v2 window sits on the cell's lat/lon midpoint: for the south disc, 5 km off the pole."),
]


def pole_cells(rows: list[int]):
    """Every cell of the given rows (a full circle of columns each)."""
    cells = []
    for r in rows:
        n = round(360 / cell_10km(r, 0).dlon)
        cells += [cell_10km(r, c) for c in range(n)]
    return cells


def coverage(cells, crs: str) -> np.ndarray:
    """Windows covering each pixel of a ±HALF_KM square around the pole, -1 outside the shown cells."""
    size = int(2 * HALF_KM * 1000 / RES_M)
    transform = from_origin(2e6 - HALF_KM * 1000, 2e6 + HALF_KM * 1000, RES_M, RES_M)   # UPS: pole at (2e6, 2e6)
    windows = [window_centroid(c, crs, snap=False) for c in cells]
    count = rasterize([(w, 1) for w in windows], out_shape=(size, size), transform=transform,
                      merge_alg=MergeAlg.add, dtype="uint8")
    polys = [shapely.make_valid(cell_polygon(c, crs, n=48)) for c in cells]
    inside = rasterize([(p, 1) for p in polys], out_shape=(size, size), transform=transform, dtype="uint8") > 0
    return np.where(inside, np.minimum(count, 2), -1), polys


def draw_pole(fig, rect, title: str, crs: str, rows: list[int]) -> dict:
    cells = pole_cells(rows)
    cat, polys = coverage(cells, crs)
    colors = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in [YELLOW, MINT, INDIGO_TINT, PAPER]],
                      dtype=np.uint8)
    ax = fig.add_axes(rect)
    ax.imshow(colors[np.where(cat < 0, 3, cat)], extent=(-HALF_KM, HALF_KM, -HALF_KM, HALF_KM))
    for p in polys:
        for g in getattr(p, "geoms", [p]):
            if g.geom_type == "Polygon":
                x, y = g.exterior.xy
                ax.plot((np.asarray(x) - 2e6) / 1e3, (np.asarray(y) - 2e6) / 1e3, color=INK_SOFT, lw=0.5)
    ax.plot([0], [0], marker="+", color=INK, ms=14, mew=2)
    for r in rows[:3]:                                                # name the innermost rows
        cell = cell_10km(r, 0)
        n = round(360 / cell.dlon)
        dist_km = (90 - abs(cell.lat + cell.dlat / 2)) * 111.2          # ring's middle, from the pole
        ax.text(0.8, -dist_km, f"{abs(r)}{'U' if r > 0 else 'D'}: {n} cell{'s' * (n > 1)}", fontsize=10,
                color=INK, ha="left", va="center", bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2))
    ax.set_xlim(-HALF_KM, HALF_KM)
    ax.set_ylim(-HALF_KM, HALF_KM)
    ax.set_aspect("equal")
    ax.axis("off")
    shown = cat >= 0
    stats = {"uncovered_km2": float((cat == 0).sum() * RES_M ** 2 / 1e6),
             "uncovered_pct": float(100 * (cat == 0).sum() / shown.sum())}
    x = rect[0] + rect[2] / 2
    text(fig, x, rect[1] + rect[3] + 0.035, title, size=17, weight=500, ha="center", va="center")
    text(fig, x, rect[1] + rect[3] + 0.006, f"{crs} (UPS) · ±{HALF_KM} km around the pole", size=11,
         color=INK_SOFT, ha="center", va="center")
    text(fig, x, rect[1] - 0.03, f"not covered: {stats['uncovered_km2']:.1f} km² ({stats['uncovered_pct']:.2f}% of shown area)",
         size=12, weight=400, ha="center", va="center")
    return stats


def main() -> None:
    use_brand_fonts()
    fig = new_figure()
    text(fig, 0.03, 0.935, "Major TOM at the poles", size=24, weight=500, va="center")
    text(fig, 0.03, 0.885, "10 km cells with a hypothetical instrument and 1056 px windows", size=14,
         color=INK_SOFT, va="center")
    stats = [draw_pole(fig, [0.02 + i * 0.315, 0.15, 0.3, 0.62], title, crs, rows)
             for i, (title, crs, rows) in enumerate(POLES)]
    for i, (label, color) in enumerate(zip(["not covered", "covered once", "twice or more"], [YELLOW, MINT, INDIGO_TINT])):
        text(fig, 0.05 + i * 0.16, 0.055, "■", size=18, color=color, va="center")
        text(fig, 0.07 + i * 0.16, 0.055, label, size=12, weight=400, va="center")
    text(fig, 0.665, 0.80, "Caveats", size=17, weight=500, va="center")
    for i, (head, body) in enumerate(CAVEATS):
        y = 0.73 - i * 0.125
        text(fig, 0.665, y, f"{i + 1}  {head}", size=13, weight=500, va="center", color=INDIGO)
        text(fig, 0.683, y - 0.045, textwrap.fill(body, 50), size=11, va="center", linespacing=1.25)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "poles_1056.png", dpi=200, facecolor=PAPER)
    print("wrote out/poles_1056.png", stats)


if __name__ == "__main__":
    main()
