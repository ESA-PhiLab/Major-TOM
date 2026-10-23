"""Cell values from a latitude/longitude source grid (docs/CLIMATE.md section 3).

    cells (grid_row_u, grid_col_r)  +  source grid
      -> per cell: overlap with each pixel, in longitude x in sin(latitude)   (exact area on a sphere)
      -> Weights: (cell, pixel, weight) triplets; a cell's weights sum to 1 where the grid covers it
    field (one time step, NaN where missing)
      -> cell_means: weighted mean over valid pixels, and the share of the cell that is valid

Major TOM cells and source pixels are both latitude/longitude boxes, so the overlap separates into
a longitude part and a latitude part; no polygon intersection is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cells import Cell


@dataclass(frozen=True)
class Grid:
    """A regular latitude/longitude grid as a source stores it: row 0 is the northernmost.

    Longitudes wrap: a grid running 0-360° (ERA5) and one running -180-180° serve the same cells.
    """
    west: float          # west edge of column 0 (ERA5: -0.125, pixels centred on the grid lines)
    north: float         # north edge of row 0  (ERA5: 90.125)
    step: float          # pixel size in degrees
    width: int
    height: int


@dataclass(frozen=True)
class Weights:
    """Sparse cells x pixels matrix as triplets; pixel = row * width + column of the source grid."""
    cell: np.ndarray     # index into the cells table
    pixel: np.ndarray
    weight: np.ndarray
    n_cells: int


def _lat_parts(lat0: float, lat1: float, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Source rows a latitude band [lat0, lat1] overlaps, and each overlap in sin(latitude)."""
    first = max(math.floor((grid.north - lat1) / grid.step), 0)
    last = min(math.ceil((grid.north - lat0) / grid.step), grid.height)
    rows = np.arange(first, last)
    top = np.minimum(grid.north - rows * grid.step, lat1)
    bottom = np.maximum(grid.north - (rows + 1) * grid.step, lat0)
    return rows, np.sin(np.radians(top)) - np.sin(np.radians(bottom))


def _lon_parts(lon0: np.ndarray, dlon: float, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Source columns each cell [lon0, lon0 + dlon] overlaps (cells x span), and each overlap in degrees."""
    span = math.ceil(dlon / grid.step) + 1
    first = np.floor((lon0 - grid.west) / grid.step).astype(int)
    cols = first[:, None] + np.arange(span)
    west = grid.west + cols * grid.step
    overlap = np.clip(np.minimum(west + grid.step, (lon0 + dlon)[:, None]) - np.maximum(west, lon0[:, None]), 0, None)
    return cols % round(360 / grid.step), overlap          # wrap to the grid's own longitudes


def weights(cells: pd.DataFrame, grid: Grid, d: float = 10.0) -> Weights:
    """Weights of each source pixel in each cell of `cells` (columns grid_row_u, grid_col_r), in table order.

    Cells of one grid row share their latitude band and width, so the work is done row by row.
    """
    parts = []
    for row, group in cells.groupby("grid_row_u", sort=False):
        cell = Cell(int(row), 0, d)
        lat_rows, lat_w = _lat_parts(cell.lat, cell.lat + cell.dlat, grid)
        lon0 = group["grid_col_r"].to_numpy() * cell.dlon
        lon_cols, lon_w = _lon_parts(lon0, cell.dlon, grid)
        area = cell.dlon * (math.sin(math.radians(cell.lat + cell.dlat)) - math.sin(math.radians(cell.lat)))
        w = lon_w[:, None, :] * lat_w[None, :, None] / area          # cells x lat rows x lon columns
        pixel = lat_rows[None, :, None] * grid.width + lon_cols[:, None, :]
        index = np.broadcast_to(group.index.to_numpy()[:, None, None], w.shape)
        keep = (w > 0) & (lon_cols[:, None, :] < grid.width)        # regional grids: drop columns outside
        parts.append((index[keep], pixel[keep], w[keep]))
    index, pixel, w = (np.concatenate(p) for p in zip(*parts))
    position = pd.Index(cells.index).get_indexer(index)               # table order, whatever the index is
    return Weights(position, pixel, w, len(cells))


def cell_means(w: Weights, field: np.ndarray, pixel_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(value, valid) per cell from one time step (height x width, NaN where missing).

    Missing pixels drop out and the rest are re-weighted; `valid` is the share of the cell with data.
    `pixel_weight` (height x width, 0..1) scales each pixel, e.g. a land fraction for land-only variables:
    the value is then the mean over the land part, and `valid` the land share with data.
    """
    values = field.ravel()[w.pixel]
    weight = w.weight if pixel_weight is None else w.weight * pixel_weight.ravel()[w.pixel]
    ok = np.isfinite(values)
    valid = np.bincount(w.cell, weight * ok, minlength=w.n_cells)
    total = np.bincount(w.cell, weight * np.where(ok, values, 0.0), minlength=w.n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid > 0, total / valid, np.nan), valid


class ByCell:
    """Weights sorted by cell, so the weights of any subset of cells are a cheap gather (`subset`)."""

    def __init__(self, w: Weights):
        order = np.argsort(w.cell, kind="stable")
        self.pixel, self.weight = w.pixel[order], w.weight[order]
        self.start = np.searchsorted(w.cell[order], np.arange(w.n_cells + 1))

    def subset(self, cells: np.ndarray) -> Weights:
        """Weights of `cells` (indices into the original table); result cell i is cells[i]."""
        lengths = self.start[cells + 1] - self.start[cells]
        first = np.repeat(self.start[cells] - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
        index = first + np.arange(lengths.sum())
        return Weights(np.repeat(np.arange(len(cells)), lengths), self.pixel[index], self.weight[index], len(cells))
