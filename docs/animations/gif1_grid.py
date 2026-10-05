"""GIF 1: how the Major TOM grid is defined (rows in latitude, points per row, cells).

Left: the official Major TOM grid at a toy spacing d = 500 km. Right: a plain lat/lon grid
with the same rows and a fixed number of points per row, for comparison.
Loop: rest -> rows -> points -> one cell highlighted -> take down -> rest (seam inside the rest).

Usage (from docs/animations):
  srun --cpus-per-task=4 --mem=16G python gif1_grid.py [--review DIR]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from geometry import Row, grid_rows, official_grid, plain_rows
from style import (INDIGO, INK, INK_FAINT, INK_SOFT, PAPER, mix, new_figure, phase, render_loop,
                   text, use_brand_fonts)

D_KM = 1000                             # toy spacing so points are visible; Major TOM uses 10 km
LON0, LAT0 = 10.0, 35.0                 # view centre and tilt (deg); static, so held frames compress
HIGHLIGHT = ("2U", "1R")                # row and column of the highlighted cell
LABELLED_ROWS = ("2U", "1U", "0U", "1D", "2D")
FOOTER = ("Shown at d = 1,000 km so points are visible. "
          "Major TOM uses d = 10 km: 2,004 rows pole to pole, about 5 million points.")
CAPTIONS = [  # (fade in from, to, fade out from, to), text
    ((0.08, 0.11, 0.27, 0.30), "Rows: equal steps in latitude, d apart. "
                               "Named 0U, 1U, … northwards and 1D, 2D, … southwards."),
    ((0.31, 0.34, 0.52, 0.55), "Points: each row is split into circumference ÷ d steps, rounded up, "
                               "so neighbours stay about d apart."),
    ((0.56, 0.59, 0.71, 0.74), "Cells: each point anchors one cell, named by its row and column."),
]


@dataclass
class Scene:
    rows: list[Row]                                   # Major TOM rows
    plain: list[Row]                                  # comparison rows
    coast: np.ndarray                                 # land outlines, (lon, lat) with NaN breaks
    cell: tuple[float, float, float, float]           # highlighted cell: lon, lat, dlon, dlat


def load_scene() -> Scene:
    """Official toy grid, its plain counterpart, land outlines and the highlighted cell."""
    rows = grid_rows(official_grid(D_KM))
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    land = shapely.unary_union(world.geometry.values)
    rings = [np.asarray(p.exterior.coords) for p in getattr(land, "geoms", [land])]
    coast = np.concatenate([np.vstack([r, [np.nan, np.nan]]) for r in rings])
    i = [r.name for r in rows].index(HIGHLIGHT[0])
    r = rows[i]
    cell = (r.lons[r.cols.index(HIGHLIGHT[1])], r.lat, 360 / len(r.lons), rows[i + 1].lat - r.lat)
    return Scene(rows, plain_rows(rows), coast, cell)


def project(lon: np.ndarray, lat: np.ndarray, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic view of a unit globe centred on (lon0, LAT0); far-side points become NaN."""
    lon, lat = np.radians(lon), np.radians(lat)
    l0, p0 = np.radians(lon0), np.radians(LAT0)
    x = np.cos(lat) * np.sin(lon - l0)
    y = np.cos(p0) * np.sin(lat) - np.sin(p0) * np.cos(lat) * np.cos(lon - l0)
    front = np.sin(p0) * np.sin(lat) + np.cos(p0) * np.cos(lat) * np.cos(lon - l0) > 0
    return np.where(front, x, np.nan), np.where(front, y, np.nan)


def wave(progress: float, k: np.ndarray, width: float = 2.0) -> np.ndarray:
    """Per-row visibility 0..1; rows switch on outwards from the equator as progress goes 0 -> 1."""
    return np.clip((progress * (k.max() + width) - k) / width, 0, 1)


def draw_globe(ax, rows: list[Row], coast: np.ndarray, lon0: float,
               rows_on: float, points_on: float, dim: float, labels: bool) -> None:
    """One globe: outline, land, rows and points at the given build-up levels."""
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    a = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(a), np.sin(a), color=INK, lw=1.4, zorder=1)
    ax.plot(*project(coast[:, 0], coast[:, 1], lon0), color=INK_FAINT, lw=0.8, zorder=1)

    k = np.array([int(r.name[:-1]) for r in rows])              # rows away from the equator
    ring = np.linspace(-180, 180, 361)
    for r, lvl in zip(rows, wave(rows_on, k)):
        if lvl > 0:
            ax.plot(*project(ring, np.full_like(ring, r.lat), lon0),
                    color=mix(PAPER, INK_SOFT, lvl), lw=0.9, zorder=2)
            if labels and r.name in LABELLED_ROWS:
                x, y = project(ring, np.full_like(ring, r.lat), lon0)
                i = np.nanargmin(x)                                 # leftmost visible point of the row
                ax.text(-1.04, y[i], r.name, fontsize=12, fontweight=400, ha="right", va="center",
                        color=mix(PAPER, INK_SOFT, lvl))

    xs, ys, sizes = [], [], []
    for r, lvl in zip(rows, wave(points_on, k)):
        if lvl > 0:
            x, y = project(r.lons, np.full_like(r.lons, r.lat), lon0)
            keep = ~np.isnan(x)
            xs.append(x[keep])
            ys.append(y[keep])
            sizes.append(np.full(keep.sum(), 24 * lvl))
    if xs:
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=np.concatenate(sizes),
                   color=mix(INK, PAPER, 0.45 * dim), linewidths=0, zorder=3)


def draw_cell(ax, cell: tuple[float, float, float, float], lon0: float, level: float) -> None:
    """Outline of one cell, its anchor point and its name."""
    lon, lat, dlon, dlat = cell
    s = np.linspace(0, 1, 30)
    u = np.r_[s, np.ones(30), s[::-1], np.zeros(30)]
    v = np.r_[np.zeros(30), s, np.ones(30), s[::-1]]
    ax.plot(*project(lon + u * dlon, lat + v * dlat, lon0), color=mix(PAPER, INK, level), lw=2.8, zorder=4)
    x, y = project(np.array([lon]), np.array([lat]), lon0)
    ax.scatter(x, y, s=130 * level, color=INDIGO, linewidths=0, zorder=5)
    ax.text(x[0] - 0.04, y[0] - 0.04, "_".join(HIGHLIGHT), fontsize=18, fontweight=500, ha="right",
            va="top", color=mix(PAPER, INDIGO, level), zorder=5,
            bbox=dict(boxstyle="square,pad=0.35", facecolor=PAPER, edgecolor="none"))


def draw(t: float, scene: Scene):
    """Frame at loop time t in [0, 1)."""
    lon0 = LON0
    rows_on = phase(t, 0.08, 0.30) - phase(t, 0.82, 0.92)       # rows grow out, later retract
    points_on = phase(t, 0.30, 0.55) - phase(t, 0.75, 0.85)     # points pop in row by row, later go
    cell_on = phase(t, 0.55, 0.60) - phase(t, 0.70, 0.75)       # one anchor and its cell
    counts_on = phase(t, 0.45, 0.50) - phase(t, 0.75, 0.80)

    fig = new_figure()
    left, right = fig.add_axes([0.04, 0.17, 0.44, 0.72]), fig.add_axes([0.52, 0.17, 0.44, 0.72])
    draw_globe(left, scene.rows, scene.coast, lon0, rows_on, points_on, dim=cell_on, labels=True)
    draw_globe(right, scene.plain, scene.coast, lon0, rows_on, points_on, dim=0.0, labels=False)
    if cell_on > 0:
        draw_cell(left, scene.cell, lon0, cell_on)

    eq = next(r for r in scene.rows if r.name == "0U")
    top = scene.rows[-1]
    text(fig, 0.26, 0.93, "Major TOM grid", size=20, weight=500, ha="center")
    text(fig, 0.74, 0.93, "Plain lat/lon grid", size=20, weight=500, color=INK_SOFT, ha="center")
    text(fig, 0.26, 0.145, f"{len(eq.lons)} points on the equator → {len(top.lons)} at {top.lat:.0f}°N",
         size=14, ha="center", level=counts_on)
    text(fig, 0.74, 0.145, f"{len(eq.lons)} points on every row: crowded near the poles",
         size=14, ha="center", color=INK_SOFT, level=counts_on)
    for (a, b, c, d), caption in CAPTIONS:
        text(fig, 0.5, 0.075, caption, size=17, weight=400, ha="center",
             level=phase(t, a, b) - phase(t, c, d))
    text(fig, 0.5, 0.025, FOOTER, size=11, color=INK_SOFT, ha="center")
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review", type=Path, help="also save a contact sheet of a few frames to this folder")
    args = p.parse_args()
    use_brand_fonts()
    scene = load_scene()
    render_loop(lambda t: draw(t, scene), seconds=12, name="gif1_grid", review_dir=args.review,
                review_ts=(0.04, 0.2, 0.42, 0.65, 0.8, 0.95))


if __name__ == "__main__":
    main()
