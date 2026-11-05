> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (added 2026-11-05).

# Major TOM grid

Short form of [docs/SPEC.md](https://github.com/ESA-PhiLab/Major-TOM/blob/refactor/v1/docs/SPEC.md) sections 2 and 3.
Code: `majortom/cells.py` (cells), `majortom/spec.py` (windows).

## Rows, columns, names

At spacing `d` km (default 10):

- Rows run from pole to pole: `n_rows(d) = ceil(π · 6378.137 / d)` (2,004 at 10 km), each `180 / n_rows` degrees
  high. Row 0 starts at the equator; north is positive.
- Each row has its own number of columns, from the circumference at the row's southern edge:
  `n_cols = ceil(2π · 6378.137 · cos(lat) / d)`. Columns run from `-(n_cols // 2)` to `n_cols - 1 - n_cols // 2`;
  column 0 starts at 0° longitude.
- A cell is named after its south-west corner: `624U_29L` is row 624 north (`U`), column 29 west (`L`).
  South is `D`, east is `R`. In tables: `grid_cell` (name), `grid_row_u` and `grid_col_r` (signed integers).
- At 10 km the formulas give 5,113,745 cells.

| Function | Gives |
|---|---|
| `Cell(row, col, d=10.0)` | `.name`, `.lat`, `.lon` (south-west corner), `.dlat`, `.dlon`, `.n_cols`, `.centroid` |
| `cell_at(lon, lat, d)` | the cell containing a point |
| `cell_from_name(name, d)` | the cell called `name` |
| `cells_in((west, south, east, north), d)` | every cell intersecting a lon/lat box (not across 180°) |
| `cells_from_points(df, d)` | one row per cell holding points (`lon`, `lat` columns); `n_points`, `point_lon`, `point_lat` |
| `n_rows(d)` | rows from pole to pole |

## Irregularities

- Cells are not exactly `d × d`: north of the equator slightly smaller, south slightly larger (0.16% at 45°,
  0.9% at 80°).
- At 10 km the southernmost row (1002D) is one disc cell around the south pole; the northernmost row's cells
  are wedges meeting at the north pole.
- Near 180° a row with an odd number of columns has an easternmost cell that wraps across the antimeridian.
- The older `majortom.Grid` class generates rows between −85° and 85° only.
- Published cell lists differ from the formulas: Major-TOM/Spatial-Reference-Grid has 5,091,140 cells (no rows
  beyond ±85°), the Major-TOM/index has 5,055,204 (80°S to 84°N). Use `majortom.cells` as the reference.

## From cell to window

A sample covers a square window around the cell, in a projected CRS:

| | v2 (this spec, default in `majortom.build`) | Core v1 (the Hugging Face Core datasets) |
|---|---|---|
| CRS | UTM zone chosen by rule (UPS beyond 84°N / 80°S) | the UTM zone of the product used |
| Anchor | cell centroid, projected and snapped to a 60 m lattice | the cell's south-west corner minus 340 m |
| Size at 10 m | 1056 px (10,560 m), centred | 1068 px (10,680 m) |

`majortom.spec.window(cell, lattice, ...)` gives a v2 window; `legacy_window(cell, crs)` the Core v1 one;
`cell_coverage(cell, window)` the share of the cell inside a window.
