"""Major TOM geometry for the tutorial animations.

Grid: the official implementation in MajorTOM/grid.py (see `official_grid`, `utm_epsg`).
10 km cells: the same row/column rule as grid.py, computed per cell (`cell_10km`), because
building the full 10 km grid object takes minutes. `check_cells` compares the two.
Windows: the two anchoring algorithms compared in GIF 2 (`window_bottom_left`, `window_centroid`).
"""
from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import shapely
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[2]
R_KM = 6378.137                                   # equatorial radius used by grid.py
N_ROWS_10KM = math.ceil(math.pi * R_KM / 10)      # 2004 rows pole to pole
DLAT_10KM = 180 / N_ROWS_10KM


# ---------- official grid code ----------

@lru_cache(maxsize=1)
def _grid_module():
    """MajorTOM/grid.py loaded from its file.

    `import MajorTOM` also imports torch through the dataset module. After the WS1 packaging
    work this becomes `import majortom.grid`.
    """
    spec = importlib.util.spec_from_file_location("mt_grid", REPO / "MajorTOM" / "grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_grid(d_km: float):
    """`Grid(d_km)` from the official MajorTOM/grid.py (fast for toy spacings, slow at 10 km)."""
    return _grid_module().Grid(d_km)


def utm_epsg(lat: float, lon: float) -> str:
    """UTM zone at a point, with the Norway/Svalbard exceptions, from grid.py's own rule."""
    return f"EPSG:{_grid_module().get_utm_zone_from_latlng([lat, lon])}"


class Row(NamedTuple):
    """One grid row and its points."""
    name: str            # e.g. "5U" (5 rows north of the equator) or "3D"
    lat: float           # latitude of the row (deg)
    lons: np.ndarray     # longitudes of its points (deg), west to east
    cols: list[str]      # column names, e.g. "2R"; empty for the comparison grid


def grid_rows(grid) -> list[Row]:
    """Rows of an official grid, south to north."""
    return [Row(name, float(lat), g.geometry.x.to_numpy(), g.col.tolist())
            for name, lat, g in zip(grid.rows, grid.lats, grid.points_by_row)]


def plain_rows(rows: list[Row]) -> list[Row]:
    """Same rows, but every row gets the equator's number of points: a plain lat/lon grid."""
    n = max(len(r.lons) for r in rows)
    lons = (np.arange(n) - n // 2) * 360 / n          # equal longitude steps, 0° included
    return [Row(r.name, r.lat, lons, []) for r in rows]


# ---------- 10 km cells ----------

class Cell(NamedTuple):
    """A Major TOM cell: its grid point (bottom-left corner) and its size in degrees."""
    name: str
    lon: float
    lat: float
    dlon: float
    dlat: float


def cell_10km(r: int, c: int) -> Cell:
    """10 km cell from signed row/column indices (r > 0 north, c > 0 east), as in grid.py.

    Rows are 180/2004 deg apart; a row at latitude phi has ceil(2 pi R cos(phi) / 10 km)
    columns, equally spaced from 0 deg longitude.
    """
    lat = r * DLAT_10KM
    n_cols = math.ceil(2 * math.pi * R_KM * math.cos(math.radians(lat)) / 10)
    dlon = 360 / n_cols
    name = f"{abs(r)}{'U' if r >= 0 else 'D'}_{abs(c)}{'R' if c >= 0 else 'L'}"
    return Cell(name, c * dlon, lat, dlon, DLAT_10KM)


def cells_near(lat: float, lon: float, rows: int, half_width: float) -> list[Cell]:
    """10 km cells in `rows` rows either side of `lat`, centred within half_width cells of `lon`."""
    r0 = round(lat / DLAT_10KM)
    out = []
    for r in range(r0 - rows, r0 + rows + 1):
        dlon = cell_10km(r, 0).dlon                      # rows have different column widths
        c0 = round((lon - dlon / 2) / dlon)              # column whose centre is nearest to lon
        span = math.ceil(half_width) + 1
        for c in range(c0 - span, c0 + span + 1):
            cell = cell_10km(r, c)
            if abs(cell.lon + dlon / 2 - lon) <= half_width * dlon:
                out.append(cell)
    return out


def check_cells(samples: list[tuple[int, int]]) -> str:
    """Compare `cell_10km` with grid.py's own row/column code for a few (row, col) pairs."""
    G = _grid_module().Grid
    g = G.__new__(G)                                   # skip __init__: it builds every 10 km point
    g.dist, g.latitude_range, g.longitude_range = 10, (-90, 90), (-180, 180)
    rows, lats = g.get_rows()
    worst = 0.0
    for r, c in samples:
        cell = cell_10km(r, c)
        i = list(rows).index(cell.name.split("_")[0])
        cols, lons = g.subdivide_circumference(lats[i], return_cols=True)
        j = list(cols).index(cell.name.split("_")[1])
        worst = max(worst, abs(lats[i] - cell.lat), abs(lons[j] - cell.lon))
    return f"cell_10km vs grid.py on {len(samples)} cells: max difference {worst:.2e} deg"


# ---------- product windows ----------

@lru_cache(maxsize=None)
def transformer(src: str, dst: str) -> Transformer:
    """Cached coordinate transformer (x/y order, i.e. lon/lat for EPSG:4326)."""
    return Transformer.from_crs(src, dst, always_xy=True)


def snap60(x: float, y: float, crs: str) -> tuple[float, float]:
    """Nearest point of the Sentinel-2 60 m lattice (northings offset by 40 m in the south)."""
    oy = 40.0 if crs.startswith("EPSG:327") else 0.0
    return round(x / 60) * 60, round((y - oy) / 60) * 60 + oy


def cell_polygon(cell: Cell, crs: str, n: int = 24) -> shapely.Polygon:
    """Cell outline (the lat/lon rectangle, densified) projected to `crs`."""
    s = np.linspace(0, 1, n, endpoint=False)
    u = np.r_[s, np.ones(n), 1 - s, np.zeros(n)]
    v = np.r_[np.zeros(n), s, np.ones(n), 1 - s]
    x, y = transformer("EPSG:4326", crs).transform(cell.lon + u * cell.dlon, cell.lat + v * cell.dlat)
    return shapely.Polygon(np.c_[x, y])


def window_bottom_left(cell: Cell, crs: str, size_px: float = 1056, snap: bool = True) -> shapely.Polygon:
    """Core-style anchoring: grid point (the cell's bottom-left corner) minus the margin, square up-right."""
    side = size_px * 10.0
    margin = (side - 10_000) / 2
    x, y = transformer("EPSG:4326", crs).transform(cell.lon, cell.lat)
    x0, y0 = (x - margin, y - margin)
    if snap:
        x0, y0 = snap60(x0, y0, crs)
    return shapely.box(x0, y0, x0 + side, y0 + side)


def window_centroid(cell: Cell, crs: str, size_px: float = 1056, snap: bool = True) -> shapely.Polygon:
    """v2 anchoring: the cell's lat/lon midpoint, projected, snapped to 60 m, square centred on it."""
    side = size_px * 10.0
    cx, cy = transformer("EPSG:4326", crs).transform(cell.lon + cell.dlon / 2, cell.lat + cell.dlat / 2)
    if snap:
        cx, cy = snap60(cx, cy, crs)
    return shapely.box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def reproject(poly: shapely.Polygon, src: str, dst: str, step_m: float = 500) -> shapely.Polygon:
    """Polygon from `src` to `dst`, densified first so straight edges bend correctly."""
    if src == dst:
        return poly
    tr = transformer(src, dst)
    return shapely.transform(shapely.segmentize(poly, step_m),
                             lambda xy: np.c_[tr.transform(xy[:, 0], xy[:, 1])])


if __name__ == "__main__":
    # self-test: the per-cell rule must match grid.py exactly (cells used by the GIFs + a few others)
    print(check_cells([(902, 0), (902, 36), (901, 35), (903, 37), (501, 106), (500, 105),
                       (0, 0), (0, -1000), (-795, 532), (-885, -171)]))
