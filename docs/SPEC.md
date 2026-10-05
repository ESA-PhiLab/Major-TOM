> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (sections 1–3 drafted; 4–9 to follow)

# Major TOM sample specification

How a Major TOM grid cell becomes a sample: a window of pixels from a data source. Applies to any grid spacing and any source; Sentinel-2 at 10 km is one instance of it.

## 1. Scope and terms

**In scope:** the grid, the window around each cell, its coordinate reference system (CRS) and pixel grid, and the metadata a sample must carry.

**Not in scope:** which acquisition to pick for a cell (dates, clouds), how pixels are corrected, and how samples are stored (TACO, rumi, parquet). Those are choices of each dataset or tool.

| Term | Meaning |
|---|---|
| grid spacing `d` | Target distance between neighbouring grid points, in km. Major TOM Core uses `d` = 10 km; any `d` is allowed. |
| grid point | One point of the grid, named by its row and column, e.g. `451U_946L`. |
| row | All grid points at one latitude. Named `kU` (k rows north of the equator) or `kD` (south). |
| column | Position of a point within its row. Named `cR` (east of 0° longitude) or `cL` (west). |
| cell | The area between a grid point and the next row and column: the point is the cell's south-west corner. A cell shares its point's name. |
| cell centroid | The midpoint of the cell in latitude and longitude. |
| source | Where pixels come from: a sensor product (e.g. Sentinel-2 L2A) or a derived product (e.g. a DEM). |
| source profile | Everything the window rule needs to know about a source, or about a set of sources used together: its CRS family, its pixel sizes and its lattice (section 4). |
| lattice `L` | The spacing, in metres, of a grid of positions on which the edges of every pixel in the profile fall. Example: Sentinel-2 pixels of 10, 20 and 60 m all have edges every 60 m, so `L` = 60 m. |
| anchor | The point a window is placed around. In this spec: the cell centroid, moved to the nearest lattice position. |
| window | A square of side `S` metres in the profile's CRS, centred on the anchor, with edges on the lattice. |
| sample | The pixels of one acquisition inside one window, plus metadata (section 8). |

## 2. The grid

The grid is an index: a set of named points covering the Earth. It is defined by the paper (arXiv 2402.12095, §3.1) and implemented in `majortom/grid.py`. This section restates that implementation; it does not change it.

**The grid is frozen.** Cell names are the key that joins all Major TOM datasets. Nothing in this spec or the library changes which points exist or how they are named (decision D10 in `docs/STRATEGY.md`). Irregularities listed below are handled by the window rule, not by changing the grid.

### 2.1 Rows

With `R` = 6378.137 km (the equatorial radius used by `grid.py`):

    N_rows = ceil(π · R / d)          number of rows from pole to pole
    Δφ     = 180° / N_rows           spacing between rows, in latitude

Row latitudes are whole multiples of `Δφ`, so row 0 lies on the equator: row `kU` is at `+k·Δφ`, row `kD` at `−k·Δφ`. For `d` = 10 km: `N_rows` = 2,004 and `Δφ` ≈ 0.0898°, about 10 km.

### 2.2 Columns

A row at latitude `φ` is a circle of circumference `2π·R·cos φ`. It is split into

    n(φ)  = ceil(2π · R · cos φ / d)       columns
    Δλ(φ) = 360° / n(φ)                    spacing in longitude

Column longitudes are whole multiples of `Δλ(φ)`, so column 0 lies on 0° longitude in every row: column `cR` is at `+c·Δλ`, `cL` at `−c·Δλ`. Rounding up makes neighbouring points at most `d` apart along a row.

### 2.3 Cells

The cell of point (`φ`, `λ`) spans latitudes `φ` to `φ + Δφ` and longitudes `λ` to `λ + Δλ(φ)`. Its centroid is at (`φ + Δφ/2`, `λ + Δλ(φ)/2`).

### 2.4 Known irregularities

These follow from the formulas above. They are documented, not fixed.

- **Default latitude range.** `Grid(d)` generates rows between −85° and +85° only (`latitude_range=(-85, 85)`). Rows nearer the poles exist only when asked for.
- **North and south cells differ slightly in size.** The number of columns comes from the row's own latitude `φ`, which is the cell's southern edge. In the northern hemisphere that is the cell's wider edge, so cells are slightly smaller than `d × d`; in the southern hemisphere it is the narrower edge, so cells are slightly larger. The difference is about `Δφ · tan φ` (in radians):

  | Latitude | Size difference, `d` = 10 km |
  |---|---|
  | 45° | 0.16% |
  | 60° | 0.3% |
  | 80° | 0.9% |
  | innermost polar ring | north row 1001U: 7 cells of ~45 km²; south row 1002D: 1 disc cell of ~314 km² |

- **The poles depend on whether `N_rows` is even or odd.**
  - Even (e.g. `d` = 10 km, 2,004 rows): the lowest row lies exactly on the south pole, and its single cell is a disc around it; the highest row's cells end exactly at the north pole, as wedges meeting there.
  - Odd (e.g. `d` = 500 km, 41 rows): the lowest row is at −87.8°, so the area south of it belongs to no cell; the highest row's cells reach past the north pole (to 92.2°).
- **Column 0 is on 0° longitude in every row**, so cell edges line up along that meridian in every row; elsewhere they do not.

## 3. From cell to window

The same rule applies to every grid spacing and every source profile. Its inputs are a cell (section 2) and a source profile (section 4), which provides:

- a **CRS rule**: how to choose a projected CRS for a cell (for most sensors, a UTM zone; near the poles, a polar stereographic CRS);
- a **lattice** `L` and its origin (`x₀`, `y₀`) in that CRS;
- the **pixel sizes** `r₁ … rₘ` of its bands, each dividing `L`;
- a **window side** `S` in metres (section 3.5).

### 3.1 CRS

The profile's CRS rule picks one projected CRS for the cell. For UTM-based sources the rule is: among the zones in which the source has native data covering the whole window, take the zone whose central meridian is nearest the cell centroid. Beyond 84°N and 80°S, where UTM is not defined, use the polar stereographic CRS (UPS: EPSG:32661 north, EPSG:32761 south). Profiles for sources on other grids (e.g. geostationary or swath sensors) define their own rule.

### 3.2 Anchor

Project the cell centroid into the chosen CRS. Call the result (`x_c`, `y_c`). Pole cells are an exception (section 6).

### 3.3 Snapping to the lattice

Move the anchor to the nearest lattice position:

    x_a = x₀ + L · round((x_c − x₀) / L)
    y_a = y₀ + L · round((y_c − y₀) / L)

This moves the anchor by at most `L/2` in each direction. It is what keeps every band's pixels whole: a window whose edges are on the lattice cuts no pixel of any band, so no band has to be resampled.

### 3.4 The window

    window = [x_a − S/2,  x_a + S/2]  ×  [y_a − S/2,  y_a + S/2]

`S` must be a whole multiple of `2L`. Then `S/2` is a multiple of `L`, so both edges of a window centred on a lattice position fall on the lattice. Each band `i` then has exactly `S / rᵢ` pixels per side, and its geotransform is

    (x_a − S/2,  rᵢ,  0,  y_a + S/2,  0,  −rᵢ)

which contains only whole numbers when the lattice and pixel sizes are whole metres.

Example: Sentinel-2 at `d` = 10 km uses `L` = 60 m and `S` = 10,560 m, giving 1056 × 1056 pixels at 10 m, 528 × 528 at 20 m and 176 × 176 at 60 m (worked through in section 5).

### 3.5 Choosing the window side `S`

`S` balances two effects, and every dataset states its choice:

- **Omission:** if `S` is too small, parts of some cells fall outside their window.
- **Duplication:** if `S` is too large, neighbouring windows overlap. The extra area per sample is about `(S / d)² − 1`.

A window contains its whole cell when

    S ≥ s · (cos γ + sin γ) + L

where `s` is the cell's largest extent (about `d`), `γ` is the angle between the cell's edges and the CRS axes, and `L` covers the snapping shift. In UTM, `γ` grows with distance from the zone's central meridian and with latitude: up to about 3° in regular 6° zones, and up to about 6° in the wide Norway/Svalbard zones. Choosing `S` for the largest `γ` of the whole Earth guarantees full coverage but duplicates more everywhere; choosing it for typical cells leaves slivers in a few, which are flagged (section 6).

Further constraints a profile may add: `S / rᵢ` a multiple of 16 for every band, so the window splits into unpadded GeoTIFF tiles.

### 3.6 Profiles with several sources

When sources are used together (e.g. Sentinel-2, Landsat, Sentinel-1 and a DEM), the profile has one lattice if all their native pixel grids share it. If they do not, the profile names a **reference source** whose lattice defines the window; the other sources are resampled onto the reference pixel grid, and each sample records which bands were resampled and how. Section 4 gives the combined optical/radar/elevation profile.

## 4. Source profiles

*To follow:* Sentinel-2 L1C/L2A; the combined Sentinel-2 + Landsat + Sentinel-1 + Copernicus DEM profile; notes on sensors with very different grids (e.g. Sentinel-5P, geostationary).

## 5. Worked examples

*To follow:* Sentinel-2 at `d` = 10 km for cell `451U_946L` (Snowbird); one other spacing.

## 6. Special cases

*To follow:* exception zones, pole caps, `cell_coverage` flags.

## 7. Legacy Core v1

*To follow:* bottom-left anchor, 1068 px, fractional-geotransform correction, 1068 → 1056 crop.

## 8. Required metadata per sample

*To follow.*

## 9. Open points

*To follow.*
