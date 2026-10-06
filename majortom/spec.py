"""Sample windows: from a cell to a square window on a pixel lattice (docs/SPEC.md section 3).

    cell -> CRS (UTM zone, UPS near the poles) -> anchor (centroid, snapped to the lattice) -> window
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer

from .cells import Cell
from .grid import get_utm_zone_from_latlng

MARGIN = 0.056                 # default window side: (1 + MARGIN) * d, i.e. 10,560 m at d = 10 km
SOUTH_FALSE_NORTHING = 10_000_000
UPS_FALSE_ORIGIN = 2_000_000             # UPS puts the pole at (2,000 km, 2,000 km)


def cell_crs(cell: Cell) -> str:
    """UTM zone of the cell centroid (with the Norway/Svalbard exceptions); UPS beyond 84°N / 80°S."""
    lon, lat = cell.centroid
    if lat > 84:
        return "EPSG:32661"
    if lat < -80:
        return "EPSG:32761"
    return f"EPSG:{get_utm_zone_from_latlng([lat, lon])}"


def default_origin(crs: str, lattice: float) -> tuple[float, float]:
    """A lattice point when the source has no native grid in this CRS.

    In southern UTM zones northings start at 10,000 km; the lattice stays aligned with the equator,
    as Sentinel-2's does (y = 40 mod 60). In UPS the lattice passes through the pole.
    """
    if crs in ("EPSG:32661", "EPSG:32761"):
        return UPS_FALSE_ORIGIN % lattice, UPS_FALSE_ORIGIN % lattice
    south_utm = crs.startswith("EPSG:327")
    return 0.0, (SOUTH_FALSE_NORTHING % lattice if south_utm else 0.0)


@dataclass(frozen=True)
class Window:
    """A square window: CRS, top-left corner and side, all in metres."""
    crs: str
    left: float
    top: float
    side: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.left, self.top - self.side, self.left + self.side, self.top

    def pixels(self, res: float) -> int:
        """Pixels per side at resolution `res` (the side is a whole multiple of every resolution)."""
        return round(self.side / res)

    def transform(self, res: float) -> Affine:
        return Affine(res, 0.0, self.left, 0.0, -res, self.top)


def window(cell: Cell, lattice: float, origin: tuple[float, float] | None = None,
           margin: float = MARGIN, crs: str | None = None) -> Window:
    """The window for `cell` on a lattice of spacing `lattice` metres through the point `origin`.

    Anchor: the cell centroid (the pole itself for a cell around a pole), moved to the nearest lattice
    point. Side: the smallest multiple of 2 * lattice that is at least (1 + margin) * d, so both edges
    fall on the lattice.
    """
    crs = crs or cell_crs(cell)
    x0, y0 = origin if origin is not None else default_origin(crs, lattice)
    lon, lat = cell.centroid
    if cell.dlon >= 360:                              # a single cell around a pole
        lon, lat = 0.0, math.copysign(90.0, cell.lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    x = x0 + lattice * round((x - x0) / lattice)
    y = y0 + lattice * round((y - y0) / lattice)
    side = 2 * lattice * math.ceil((1 + margin) * cell.d * 1000 / (2 * lattice) - 1e-9)
    return Window(crs, x - side / 2, y + side / 2, side)


def cell_polygon(cell: Cell, crs: str, n: int = 32) -> shapely.Polygon:
    """The cell's outline in `crs`, with edges densified so that curved edges are followed."""
    s = np.linspace(0, 1, n, endpoint=False)
    u = np.r_[s, np.ones(n), 1 - s, np.zeros(n)]
    v = np.r_[np.zeros(n), s, np.ones(n), 1 - s]
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        cell.lon + u * cell.dlon, cell.lat + v * cell.dlat)
    return shapely.make_valid(shapely.Polygon(np.c_[x, y]))


def cell_coverage(cell: Cell, win: Window) -> float:
    """Share of the cell's area inside the window (docs/SPEC.md section 6.3)."""
    poly = cell_polygon(cell, win.crs)
    return min(1.0, float(poly.intersection(shapely.box(*win.bounds)).area / poly.area))
