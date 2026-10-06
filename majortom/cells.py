"""Major TOM cells at any grid spacing, computed straight from the grid formulas (docs/SPEC.md section 2).

Only the cells asked for are computed, so this is fast at any spacing; `Grid` builds every point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

R_KM = 6378.137                                  # equatorial radius used by grid.py
QUERY_COLUMNS = ("datetime", "days", "margin")   # optional per-sample query columns (see majortom.build)


def n_rows(d: float) -> int:
    """Number of rows from pole to pole at grid spacing `d` km."""
    return math.ceil(math.pi * R_KM / d)


@dataclass(frozen=True)
class Cell:
    """One Major TOM cell: signed row (north > 0) and column (east > 0) at grid spacing `d` km.

    The cell is named after its south-west corner, the grid point: rows 0U, 1U, ... northwards and
    1D, 2D, ... southwards; columns 0R, 1R, ... eastwards and 1L, 2L, ... westwards.
    """
    row: int
    col: int
    d: float = 10.0

    @property
    def dlat(self) -> float:
        return 180 / n_rows(self.d)

    @property
    def lat(self) -> float:
        """Latitude of the south edge (the grid point)."""
        return self.row * self.dlat

    @property
    def n_cols(self) -> int:
        """Columns in this cell's row; they run from -(n_cols // 2) to n_cols - 1 - n_cols // 2."""
        circumference = 2 * math.pi * R_KM * math.cos(math.radians(self.lat))
        return max(math.ceil(circumference / self.d), 1)

    @property
    def dlon(self) -> float:
        return 360 / self.n_cols

    @property
    def lon(self) -> float:
        """Longitude of the west edge (the grid point)."""
        return self.col * self.dlon

    @property
    def centroid(self) -> tuple[float, float]:
        """(lon, lat) of the cell's midpoint in latitude and longitude."""
        return self.lon + self.dlon / 2, self.lat + self.dlat / 2

    @property
    def name(self) -> str:
        row = f"{abs(self.row)}{'U' if self.row >= 0 else 'D'}"
        col = f"{abs(self.col)}{'R' if self.col >= 0 else 'L'}"
        return f"{row}_{col}"


def cell_at(lon: float, lat: float, d: float = 10.0) -> Cell:
    """The cell containing a point. Near 180° a column may wrap: in a row with an odd number of
    columns, the easternmost cell continues across the antimeridian."""
    row = math.floor(lat / (180 / n_rows(d)))
    n = Cell(row, 0, d).n_cols
    col = (math.floor(lon / (360 / n)) + n // 2) % n - n // 2
    return Cell(row, col, d)


def _table(cells: list[Cell]) -> pd.DataFrame:
    """One row per cell: name, signed row and column, spacing, and the centroid as lon/lat."""
    return pd.DataFrame({
        "grid_cell": [c.name for c in cells],
        "row": [c.row for c in cells],
        "col": [c.col for c in cells],
        "grid_km": [c.d for c in cells],
        "lon": [c.centroid[0] for c in cells],
        "lat": [c.centroid[1] for c in cells],
    })


def cells_in(bbox: tuple[float, float, float, float], d: float = 10.0) -> pd.DataFrame:
    """Every cell that intersects a lon/lat box (west, south, east, north). Does not cross 180°."""
    west, south, east, north = bbox
    dlat = 180 / n_rows(d)
    cells = []
    for row in range(math.floor(south / dlat), math.floor(north / dlat) + 1):
        dlon = Cell(row, 0, d).dlon
        cells += [Cell(row, col, d) for col in range(math.floor(west / dlon), math.floor(east / dlon) + 1)]
    return _table(cells)


def cells_from_points(points: pd.DataFrame, d: float = 10.0) -> pd.DataFrame:
    """The cells containing `points` (columns `lon`, `lat`), one row per cell and query.

    Points in the same cell become one sample, unless they carry different per-sample queries
    (`datetime`, `days`, `margin`). Other columns are kept from the first point of each cell;
    the point's own position moves to `point_lon`, `point_lat`, and `lon`, `lat` become the cell
    centroid. `n_points` counts the points per sample.
    """
    cells = [cell_at(lon, lat, d) for lon, lat in zip(points["lon"], points["lat"])]
    table = points.rename(columns={"lon": "point_lon", "lat": "point_lat"}).reset_index(drop=True)
    table = pd.concat([_table(cells), table], axis=1)
    keys = ["grid_cell"] + [c for c in QUERY_COLUMNS if c in table]
    grouped = table.groupby(keys, sort=False, dropna=False)
    out = grouped.first().reset_index()
    out["n_points"] = grouped.size().to_numpy()
    return out
