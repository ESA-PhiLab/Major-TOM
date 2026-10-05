"""GIF 2: same grid, different products: bottom-left vs centroid anchoring, both at 1056 px.

A cell of row 902U (81°N) slides east from the central meridian of UTM zone 33X (15°E) to the
zone edge (21°E). Its lat/lon outline rotates against the UTM pixel grid, up to about 6°, because
Svalbard's zones are twice the usual width. Both panels use the same 1056 px window; only the
anchor differs. The 60 m snapping is left out so the motion stays smooth (it moves windows <= 30 m).
Loop: rest -> windows grow out of their anchors -> sweep out -> hold -> sweep back -> windows
shrink back -> rest (seam inside the rest).

Usage (from docs/animations):
  srun --cpus-per-task=4 --mem=16G python gif2_anchoring.py [--review DIR]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from shapely import affinity

from geometry import Cell, cell_10km, cell_polygon, transformer, window_bottom_left, window_centroid
from style import (INDIGO, INK, INK_FAINT, INK_SOFT, MINT, PAPER, YELLOW, ease, mix, new_figure, phase,
                   render_loop, text, use_brand_fonts)

CRS = "EPSG:32633"                     # UTM zone 33 north (Svalbard zone 33X spans 9-21°E)
ROW, CM = 902, 15.0                    # row 902U is at 81.0°N; zone central meridian
BASE = cell_10km(ROW, 0)
DL_MAX = 6.0 - BASE.dlon / 2           # stop when the cell's east edge reaches the zone edge (21°E)
HALF = 6.4                             # half-width of each panel (km)
PANELS = [([0.05, 0.17, 0.42, 0.66], "bottom-left", "Anchored at the grid point (bottom-left corner)"),
          ([0.53, 0.17, 0.42, 0.66], "centre", "Anchored at the cell centre")]
FOOTER = ("UTM zone 33X (EPSG:32633). Both windows 1056 px = 10.56 km. Grey lines: UTM grid directions. "
          "60 m snapping left out here so the motion stays smooth.")


def cell_at(dl: float) -> Cell:
    """The cell of row 902U whose centre lies `dl` degrees east of the zone's central meridian."""
    return BASE._replace(lon=CM + dl - BASE.dlon / 2)


def rotation(cell: Cell) -> float:
    """Angle (deg) between the cell's bottom edge and the UTM x axis."""
    (x0, x1), (y0, y1) = transformer("EPSG:4326", CRS).transform(
        [cell.lon, cell.lon + cell.dlon], [cell.lat, cell.lat])
    return abs(math.degrees(math.atan2(y1 - y0, x1 - x0)))


def local_km(geom, origin: tuple[float, float]):
    """UTM metres -> km relative to `origin` (the cell centre), so the cell stays in place."""
    return affinity.affine_transform(geom, [1e-3, 0, 0, 1e-3, -origin[0] * 1e-3, -origin[1] * 1e-3])


def fill(ax, geom, color: str) -> None:
    """Fill a polygon or multipolygon; empty geometries are skipped."""
    for g in getattr(geom, "geoms", [geom]):
        if not g.is_empty and g.area > 0:
            ax.fill(*g.exterior.xy, color=color, lw=0, zorder=2)


def draw_panel(ax, cell: Cell, origin: tuple[float, float], kind: str, build: float, fills: float) -> float:
    """One panel; returns the % of the cell outside the (full-size) window."""
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_aspect("equal")
    ax.axis("off")
    for v in range(-6, 7):                                            # UTM grid directions
        ax.axvline(v, color=INK_FAINT, lw=0.6, zorder=0)
        ax.axhline(v, color=INK_FAINT, lw=0.6, zorder=0)

    poly = local_km(cell_polygon(cell, CRS), origin)
    if kind == "bottom-left":
        full = window_bottom_left(cell, CRS, snap=False)
        anchor = transformer("EPSG:4326", CRS).transform(cell.lon, cell.lat)
    else:
        full = window_centroid(cell, CRS, snap=False)
        anchor = origin
    full = local_km(full, origin)
    anchor_x, anchor_y = (anchor[0] - origin[0]) * 1e-3, (anchor[1] - origin[1]) * 1e-3
    missed_pct = 100 * poly.difference(full).area / poly.area

    if fills > 0:
        fill(ax, poly.intersection(full), mix(PAPER, MINT, fills))
        fill(ax, poly.difference(full), mix(PAPER, YELLOW, fills))
    ax.plot(*poly.exterior.xy, color=INK, lw=2.0, zorder=3)
    if build > 0.01:
        grown = affinity.scale(full, xfact=build, yfact=build, origin=(anchor_x, anchor_y))   # grows out of the anchor
        ax.plot(*grown.exterior.xy, color=INDIGO, lw=2.6, zorder=4)
        ax.scatter([anchor_x], [anchor_y], s=80, color=mix(PAPER, INDIGO, min(1.0, 4 * build)),
                   edgecolors=PAPER, linewidths=1.5, zorder=5)
    return missed_pct


def draw(t: float):
    """Frame at loop time t in [0, 1)."""
    build = phase(t, 0.05, 0.17) - phase(t, 0.87, 0.95)       # windows grow out of anchors, later shrink back
    fills = phase(t, 0.17, 0.22) - phase(t, 0.82, 0.87)       # covered (mint) / missed (yellow) areas
    sweep = phase(t, 0.20, 0.50) - phase(t, 0.60, 0.84)       # out to the zone edge and back
    dl = DL_MAX * sweep
    cell = cell_at(dl)
    origin = transformer("EPSG:4326", CRS).transform(cell.lon + cell.dlon / 2, cell.lat + cell.dlat / 2)
    gamma = rotation(cell)

    fig = new_figure()
    text(fig, 0.5, 0.935, f"Row 902U, 81°N   ·   {dl:.1f}° east of the zone's central meridian   ·   "
                          f"rotation {gamma:.1f}°", size=16, weight=400, ha="center")
    for rect, kind, title in PANELS:
        missed = draw_panel(fig.add_axes(rect), cell, origin, kind, build, fills)
        x = rect[0] + rect[2] / 2
        text(fig, x, 0.865, title, size=18, weight=500, ha="center")
        text(fig, x - 0.085, 0.13, "■", size=16, color=YELLOW, level=fills * ease(missed / 0.05), va="center")
        text(fig, x - 0.06, 0.13, f"misses {missed:.2f}% of the cell", size=15, level=fills, va="center")

    gate = phase(t, 0.21, 0.24) - phase(t, 0.80, 0.83)        # while the cell is moving
    captions = [
        (phase(t, 0.05, 0.08) - phase(t, 0.17, 0.20), "Same cell, same 1056 px window. Only the anchor differs."),
        (gate * (1 - ease((gamma - 2.6) / 0.3)),
         "Away from the zone's central meridian, the lat/lon cell rotates against the UTM pixel grid."),
        (gate * ease((gamma - 3.0) / 0.3), "Rotation beyond 3° only happens in the extra-wide Norway/Svalbard zones."),
    ]
    for level, caption in captions:
        text(fig, 0.5, 0.065, caption, size=17, weight=400, ha="center", level=level)
    text(fig, 0.5, 0.02, FOOTER, size=11, color=INK_SOFT, ha="center")
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review", type=Path, help="also save a contact sheet of a few frames to this folder")
    args = p.parse_args()
    use_brand_fonts()
    render_loop(draw, seconds=14, name="gif2_anchoring", review_dir=args.review,
                review_ts=(0.02, 0.15, 0.35, 0.55, 0.72, 0.9))


if __name__ == "__main__":
    main()
