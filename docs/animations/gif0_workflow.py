"""GIF 0: the whole Major TOM workflow, from grid to AI-ready dataset.

Six stations in a row; a sample (indigo square) travels left to right and each station lights up
as it arrives. The all-lit hold doubles as a static slide.
Loop: rest (all grey) -> travel -> hold -> dim -> rest (seam inside the rest).

Usage (from docs/animations):
  srun --cpus-per-task=4 --mem=16G python gif0_workflow.py [--review DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from geometry import grid_rows, official_grid
from gif1_grid import project                          # GIF 1's globe view, so both grids look alike
from style import (INDIGO, INDIGO_TINT, INK, INK_FAINT, INK_SOFT, MINT, PAPER, YELLOW, ease, mix, new_figure,
                   phase, render_loop, text, use_brand_fonts)

XS = [1.45, 4.05, 6.65, 9.25, 11.85, 14.45]          # station centres (figure is 16 x 9 units)
Y = 5.2
STATIONS = [
    ("Major TOM grid", "points about 10 km apart;\nevery cell has a name"),
    ("Sample footprint", "1056 px window, centred,\nsnapped to the 60 m grid"),
    ("Archive query", "find the scene: Hugging Face,\nAWS, Planetary Computer, CDSE"),
    ("Read + cloud mask", "read only the window's pixels;\nflag clouds"),
    ("rumi + GeoZL", "stateless raster storage,\nframes compressed one by one"),
    ("TACO dataset", "samples + parquet metadata:\nAI-ready, cloud-native"),
]
ARRIVE = [0.06 + 0.11 * i for i in range(6)]
ICON_ROWS = grid_rows(official_grid(2500))            # 9 rows, 17 points on the equator
ICON_LON0 = 10.0        # loop time each station lights up


def tone(color: str, lit: float) -> str:
    """Grey (unlit) to full colour (lit)."""
    return mix(INK_FAINT, color, lit)


def fill(color: str, lit: float) -> str:
    """Pale (unlit) to full fill colour (lit)."""
    return mix(mix(PAPER, INK_FAINT, 0.35), color, lit)


def icon_grid(ax, x, lit):
    """Small globe of the official grid at 2,500 km spacing (same view as GIF 1)."""
    ax.add_patch(Circle((x, Y), 0.9, facecolor=PAPER, edgecolor=tone(INK, lit), lw=1.6))
    ring = np.linspace(-180, 180, 181)
    for row in ICON_ROWS:
        rx, ry = project(ring, np.full_like(ring, row.lat), ICON_LON0)
        ax.plot(x + 0.9 * rx, Y + 0.9 * ry, color=tone(INK_SOFT, lit), lw=0.6)
        px, py = project(row.lons, np.full_like(row.lons, row.lat), ICON_LON0)
        keep = ~np.isnan(px)
        ax.scatter(x + 0.9 * px[keep], Y + 0.9 * py[keep], s=9, color=tone(INK, lit), linewidths=0, zorder=3)


def icon_footprint(ax, x, lit):
    a = np.radians(6)
    corners = np.array([[-0.62, -0.62], [0.62, -0.62], [0.62, 0.62], [-0.62, 0.62]])
    rot = corners @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    ax.add_patch(Polygon(rot + [x, Y], fill=True, facecolor=fill(MINT, lit), edgecolor=tone(INK, lit), lw=1.6))
    ax.add_patch(Rectangle((x - 0.72, Y - 0.72), 1.44, 1.44, fill=False, edgecolor=tone(INDIGO, lit), lw=2.2))
    ax.scatter([x], [Y], s=28, color=tone(INDIGO, lit), zorder=4)


def icon_query(ax, x, lit):
    for i, dx in enumerate([-0.35, -0.15, 0.05]):                          # stack of candidate scenes
        ax.add_patch(Rectangle((x + dx - 0.45, Y - 0.55 + 0.2 * i), 0.9, 0.9,
                               facecolor=fill(INDIGO_TINT, lit * (0.4 + 0.3 * i)), edgecolor=tone(INK, lit), lw=1.1))
    ax.add_patch(Circle((x + 0.45, Y + 0.3), 0.32, facecolor=PAPER, edgecolor=tone(INK, lit), lw=2.2, zorder=4))
    ax.plot([x + 0.68, x + 0.95], [Y + 0.07, Y - 0.2], color=tone(INK, lit), lw=3, zorder=4)


def icon_read(ax, x, lit):
    ax.add_patch(Rectangle((x - 0.9, Y - 0.9), 1.8, 1.8, facecolor=fill(mix(PAPER, INK_FAINT, 0.4), 1),
                           edgecolor=tone(INK_SOFT, lit), lw=1.1))
    cells = 4
    for i in range(cells):
        for j in range(cells):
            cloudy = (i, j) in {(0, 3), (1, 3), (0, 2)}
            ax.add_patch(Rectangle((x - 0.5 + 0.25 * i, Y - 0.5 + 0.25 * j), 0.25, 0.25,
                                   facecolor=fill(YELLOW if cloudy else MINT, lit), edgecolor=PAPER, lw=0.8))
    ax.add_patch(Rectangle((x - 0.5, Y - 0.5), 1.0, 1.0, fill=False, edgecolor=tone(INDIGO, lit), lw=2.2))


def icon_rumi(ax, x, lit):
    for i in range(3):
        for j in range(3):                                                 # independently compressed frames
            s = 0.42 - 0.12 * ((i + j) % 2)
            cx, cy = x - 0.6 + 0.6 * i, Y - 0.6 + 0.6 * j
            ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, facecolor=fill(INDIGO, lit), edgecolor="none"))
    ax.add_patch(Rectangle((x - 0.9, Y - 0.9), 1.8, 1.8, fill=False, edgecolor=tone(INK, lit), lw=1.4, ls=(0, (3, 2))))


def icon_taco(ax, x, lit):
    ax.add_patch(FancyBboxPatch((x - 0.9, Y - 0.8), 1.8, 1.5, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=PAPER, edgecolor=tone(INK, lit), lw=1.6))
    ax.add_patch(Rectangle((x - 0.9, Y + 0.7), 0.7, 0.2, facecolor=tone(INK, lit), edgecolor="none"))   # folder tab
    for i in range(3):
        for j in range(2):
            ax.add_patch(Rectangle((x - 0.75 + 0.33 * i, Y - 0.05 + 0.36 * j), 0.27, 0.27,
                                   facecolor=fill([MINT, INDIGO_TINT, YELLOW][(i + j) % 3], lit), edgecolor="none"))
    for k in range(3):                                                     # metadata table rows
        ax.plot([x + 0.28, x + 0.78], [Y + 0.4 - 0.25 * k] * 2, color=tone(INK_SOFT, lit), lw=1.6)
    ax.plot([x - 0.75, x + 0.78], [Y - 0.55] * 2, color=tone(INK_SOFT, lit), lw=1.0)


ICONS = [icon_grid, icon_footprint, icon_query, icon_read, icon_rumi, icon_taco]


def draw(t: float):
    """Frame at loop time t in [0, 1)."""
    dim = phase(t, 0.88, 0.96)
    fig = new_figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    text(fig, 0.04, 0.915, "From grid to AI-ready samples", size=28, weight=500, va="center")
    text(fig, 0.04, 0.855, "The Major TOM workflow", size=15, color=INK_SOFT, va="center")

    for i, (x, (title, desc)) in enumerate(zip(XS, STATIONS)):
        lit = phase(t, ARRIVE[i], ARRIVE[i] + 0.03) * (1 - dim)
        ICONS[i](ax, x, lit)
        text(fig, x / 16, 3.75 / 9, title, size=16, weight=500, ha="center", va="center",
             color=mix(INK_SOFT, INK, lit))
        text(fig, x / 16, 3.05 / 9, desc, size=11.5, ha="center", va="center", color=INK_SOFT,
             linespacing=1.3)
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 1.05, Y), (XS[i + 1] - 1.05, Y), arrowstyle="-|>",
                                         mutation_scale=14, color=tone(INK_SOFT, lit), lw=1.4))
            leg = phase(t, ARRIVE[i] + 0.03, ARRIVE[i + 1])                    # sample travelling
            if 0 < leg < 1 and dim == 0:
                px = x + 1.05 + ease(leg) * (XS[i + 1] - x - 2.1)
                ax.add_patch(Rectangle((px - 0.12, Y + 0.18), 0.24, 0.24, facecolor=INDIGO, edgecolor="none"))

    group = phase(t, ARRIVE[4], ARRIVE[4] + 0.03) * (1 - dim)              # bracket: AI-ready format
    ax.plot([XS[4] - 1.0, XS[4] - 1.0, XS[5] + 1.0, XS[5] + 1.0], [2.25, 2.1, 2.1, 2.25],
            color=mix(PAPER, INDIGO, group), lw=1.5)
    text(fig, (XS[4] + XS[5]) / 32, 1.8 / 9, "AI-ready format", size=13, weight=500, color=INDIGO, ha="center",
         va="center", level=group)
    ax.plot([XS[0] - 1.0, XS[0] - 1.0, XS[3] + 1.0, XS[3] + 1.0], [2.25, 2.1, 2.1, 2.25],
            color=mix(PAPER, INK_SOFT, phase(t, ARRIVE[3], ARRIVE[3] + 0.03) * (1 - dim)), lw=1.5)
    text(fig, (XS[0] + XS[3]) / 32, 1.8 / 9, "majortom.build", size=13, weight=500, color=INK_SOFT, ha="center",
         va="center", level=phase(t, ARRIVE[3], ARRIVE[3] + 0.03) * (1 - dim))
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review", type=Path, help="also save a contact sheet of a few frames to this folder")
    args = p.parse_args()
    use_brand_fonts()
    render_loop(draw, seconds=12, name="gif0_workflow", review_dir=args.review,
                review_ts=(0.02, 0.2, 0.42, 0.62, 0.8, 0.93))


if __name__ == "__main__":
    main()
