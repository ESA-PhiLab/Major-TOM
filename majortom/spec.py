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


LEGACY_V1 = "v1"                         # margin value selecting Core v1's window rule (docs/SPEC.md 7.1, 7.5)
V1_PAD, V1_SIDE = 340.0, 10_680.0        # Core v1: the grid point minus 340 m, 10,680 m wide (1068 px at 10 m)


@dataclass(frozen=True)
class LegacyWindow(Window):
    """Core v1's window: its corner is not on any lattice, so each band is read on its own pixel grid at the
    corner rounded to that grid (docs/SPEC.md 7.2), exactly as Core v1 was. `origin` is a point of the source's
    pixel grids (Sentinel-2: a multiple of 60 m)."""
    origin: tuple[float, float] = (0.0, 0.0)

    def transform(self, res: float) -> Affine:
        x0, y0 = self.origin
        return Affine(res, 0.0, x0 + res * round((self.left - x0) / res),
                      0.0, -res, y0 + res * round((self.top - y0) / res))


def legacy_window(cell: Cell, crs: str, origin: tuple[float, float] | None = None) -> LegacyWindow:
    """Core v1's window (docs/SPEC.md 7.1): the cell's grid point (south-west corner) projected into `crs`,
    minus 340 m, 10,680 m wide; 1068 / 534 / 178 px at 10 / 20 / 60 m."""
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(cell.lon, cell.lat)
    return LegacyWindow(crs, x - V1_PAD, y - V1_PAD + V1_SIDE, V1_SIDE,
                        origin if origin is not None else default_origin(crs, 60.0))


def window(cell: Cell, lattice: float, origin: tuple[float, float] | None = None,
           margin: float | str = MARGIN, crs: str | None = None) -> Window:
    """The window for `cell` on a lattice of spacing `lattice` metres through the point `origin`.

    `margin="v1"` (LEGACY_V1) gives Core v1's window instead (legacy_window), for samples that line up pixel
    for pixel with existing Core v1 samples.

    Anchor: the cell centroid (the pole itself for a cell around a pole), moved to the nearest lattice
    point. Side: the smallest multiple of 2 * lattice that is at least (1 + margin) * d, so both edges
    fall on the lattice.
    """
    crs = crs or cell_crs(cell)
    if margin == LEGACY_V1:
        return legacy_window(cell, crs, origin)
    x0, y0 = origin if origin is not None else default_origin(crs, lattice)
    lon, lat = cell.centroid
    if cell.dlon >= 360:                              # a single cell around a pole
        lon, lat = 0.0, math.copysign(90.0, cell.lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    x = x0 + lattice * round((x - x0) / lattice)
    y = y0 + lattice * round((y - y0) / lattice)
    side = 2 * lattice * math.ceil((1 + margin) * cell.d * 1000 / (2 * lattice) - 1e-9)
    return Window(crs, x - side / 2, y + side / 2, side)


CORE_NATIVE = ("Core-S2L2A", "Core-S2L1C", "Core-S1RTC")   # pixels copied unchanged from the source


def correct_core_transform(transform: Affine, dataset: str) -> Affine:
    """The grid a Core sample's pixels were read on (docs/reports/2026-10-22-core-pixel-match.md).

    Core stored the unrounded origin; native datasets were read at that origin rounded to the
    band's own pixel grid. Core-DEM was resampled, so its stored transform is kept.
    """
    if dataset not in CORE_NATIVE:
        return transform
    rx, ry = transform.a, -transform.e
    return Affine(rx, 0.0, rx * round(transform.c / rx), 0.0, -ry, ry * round(transform.f / ry))


def core_dataset(url: str) -> str | None:
    """The Core dataset a parquet or sample URL belongs to (e.g. Core-S2L2A), from the Hugging Face path."""
    import re

    found = re.search(r"/(Core-[A-Za-z0-9]+)/", url)
    return found.group(1) if found else None


def crop_core_to_v2(data: np.ndarray, transform: Affine, crs: str, cell: Cell, dataset: str,
                    margin: float = MARGIN) -> tuple[np.ndarray, Affine] | None:
    """Cut this spec's (v2) window out of one band of a Core v1 sample (docs/SPEC.md 7.3), or None.

    The stored transform is corrected first (correct_core_transform); the v2 window is taken in the sample's
    CRS on the band's own pixel grid. None when the v2 window is not inside the sample (about half of Core
    cells, more at high latitudes): those need a new read from the source product.
    """
    t = correct_core_transform(transform, dataset)
    res = t.a
    win = window(cell, lattice=60.0, margin=margin, crs=crs)             # Sentinel-2: a 60 m lattice in every zone
    col = (win.left - t.c) / res
    row = (t.f - win.top) / res
    n = win.pixels(res)
    if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
        raise ValueError(f"v2 window not on the sample's pixel grid (offsets {col}, {row})")
    col, row = int(round(col)), int(round(row))
    height, width = data.shape[-2:]
    if col < 0 or row < 0 or col + n > width or row + n > height:
        return None
    return data[..., row:row + n, col:col + n], win.transform(res)


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
