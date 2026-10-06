> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (sections 1–6 drafted; 7–9 to follow)

# Major TOM sample specification

How a Major TOM grid cell becomes a sample: a window of pixels from a data source. Applies to any grid spacing and any source; Sentinel-2 at 10 km is one instance of it.

## 1. Scope and terms

**In scope:** the grid, the window around each cell, its coordinate reference system (CRS) and pixel grid, and the metadata a sample must carry.

**Not in scope:** which acquisition to pick for a cell (dates, clouds), how pixels are corrected, and how samples are stored (TACO, rumi, parquet). Those are choices of each dataset or tool.

| Term | Meaning |
|---|---|
| grid spacing `d` | Target distance between neighbouring grid points, in km. Major TOM Core uses `d` = 10 km; any `d` is allowed. |
| grid point | One point of the grid, named by its row and column, e.g. `451U_946L`. |
| row | All grid points at one latitude. Named `kU` (k rows north of the equator) or `kD` (south). Row `0U` lies on the equator. |
| column | Position of a point within its row. Named `cR` (east of 0° longitude) or `cL` (west). Column `0R` is at 0 degree longitude. |
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

The equator row is `0U`; there is no `0D`. Northwards the rows are `0U`, `1U`, `2U`, …; southwards `1D`, `2D`, …. Because a cell is named after its south-west corner, the `0U` cells lie just north of the equator and the `1D` cells just south of it.

### 2.2 Columns

A row at latitude `φ` is a circle of circumference `2π·R·cos φ`. It is split into

    n(φ)  = ceil(2π · R · cos φ / d)       columns
    Δλ(φ) = 360° / n(φ)                    spacing in longitude

Column longitudes are whole multiples of `Δλ(φ)`, so column 0 lies on 0° longitude in every row: column `cR` is at `+c·Δλ`, `cL` at `−c·Δλ`. As with rows, the 0° column is `0R` and there is no `0L`: eastwards `0R`, `1R`, …; westwards `1L`, `2L`, …. Rounding up makes neighbouring points at most `d` apart along a row.

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

A profile fixes everything section 3 needs. Pixel grids below were read from product headers over Snowbird, Utah (UTM zone 12N), on 2026-10-06; see section 4.5.

### 4.1 What a profile records

| Field | Meaning |
|---|---|
| name and version | e.g. `s2` v1; recorded in every sample (section 8) |
| sources | the products the profile covers, with their bands and pixel sizes |
| reference source | the source whose pixel grid defines the lattice (section 3.6) |
| CRS rule | how a cell's CRS is chosen (section 3.1) |
| lattice `L` and origin | where pixel edges of the reference source fall |
| resampling | for every non-reference source: aligned (no resampling) or resampled, with the method and target pixel size |
| default `S` at `d` = 10 km | the window side, and the pixel counts it gives |

### 4.2 Sentinel-2 L1C / L2A (`s2`)

- **Bands:** 10 m (B02, B03, B04, B08), 20 m (B05, B06, B07, B8A, B11, B12, scene classes), 60 m (B01, B09, B10).
- **CRS rule:** UTM, through the Sentinel-2 tiling grid: among tiles that contain the whole window, the one whose zone's central meridian is nearest the cell centroid. UPS beyond 84°N / 80°S.
- **Lattice:** `L` = 60 m. Every band's tile origin is a multiple of 60 m (measured: x = 399,960 m, y = 4,500,000 m and 4,600,020 m), so pixel edges of all three resolutions coincide every 60 m. Origin: x ≡ 0, y ≡ 0 (mod 60) in the northern hemisphere; y ≡ 40 (mod 60) in the southern hemisphere, where UTM northings start at 10,000 km.
- **Default `S` at `d` = 10 km:** 10,560 m, giving 1056 / 528 / 176 pixels at 10 / 20 / 60 m. Divisible by 16 at every resolution, so it tiles into unpadded GeoTIFF blocks.

### 4.3 Optical, radar and elevation together (`s2-ls-s1-dem`)

Sentinel-2, Landsat 8/9, Sentinel-1 and the Copernicus DEM are often used together. They do not share one pixel grid, so Sentinel-2 is the reference source and its lattice defines the window:

| Source | Native pixel grid | In this profile |
|---|---|---|
| Sentinel-2 L1C/L2A | UTM, 10/20/60 m, edges on the 60 m lattice | reference: as in 4.2 |
| Sentinel-1 RTC (Planetary Computer) | UTM, 10 m, edges on whole multiples of 10 m | aligned with Sentinel-2's 10 m pixels when the UTM zone is the same: no resampling. Different zone: resampled (bilinear) |
| Landsat 8/9 Collection 2, Level 2 | UTM, 30 m, edges offset by 15 m from multiples of 30 m | always resampled onto a 30 m grid on the lattice (bilinear; nearest for quality bands) |
| Copernicus DEM GLO-30 | latitude/longitude (EPSG:4326), 1 arc-second (≈ 30 m), 1° tiles | always resampled onto a 30 m grid on the lattice (bilinear) |

- **Target pixel sizes:** Landsat and the DEM keep about their native resolution, 30 m, rather than being enlarged to 10 m: 30 m divides the 60 m lattice, so `S` = 10,560 m gives 352 × 352 pixels, and no detail is invented. A profile variant may resample them to 10 m for models that need one resolution.
- **Recorded per sample:** for every resampled band, the method and the source pixel grid (section 8).

### 4.4 Sensors on very different grids

Some sensors do not fit UTM and metre-scale lattices. A profile for them defines its own CRS rule and lattice, and usually a larger grid spacing `d`. Two examples, not specified here:

- **Swath sensors such as Sentinel-5P** (pixels of a few km, delivered along the orbit): the lattice would be that of a gridded product (e.g. a regular latitude/longitude or equal-area grid), with `d` of tens to hundreds of km.
- **Geostationary sensors** (a fixed view of one Earth disk): the CRS is the satellite's own fixed-grid projection, the lattice is its pixel grid, and cells outside the disk have no samples.

### 4.5 How the pixel grids were checked

Headers of three recent scenes per collection over Snowbird, from Planetary Computer's STAC catalogue (2026-10-06): the CRS, pixel size and origin of each band, and the origin's remainder against 60 m and 30 m. Script: `docs/evidence/check_grids.py`.

## 5. Worked examples

Both examples use the `s2` profile (section 4.2) at Snowbird, Utah (40.5829°N, 111.6556°W). All numbers were computed with the formulas of sections 2–3.

### 5.1 Sentinel-2 at `d` = 10 km: cell `451U_946L`

| Step | Result |
|---|---|
| Rows (2.1) | `N_rows` = 2,004; `Δφ` = 0.089820°. Snowbird is in row 451U, at 40.508982°N |
| Columns (2.2) | this row has 3,047 columns; `Δλ` = 0.118149°. Snowbird is in column 946L, at 111.768953°W |
| Cell (2.3) | `451U_946L`; 99.74 km²; centroid 40.553892°N, 111.709879°W |
| CRS (3.1) | UTM zone 12N (EPSG:32612); Sentinel-2 tile 12TVK contains the whole window |
| Anchor (3.2) | centroid projected: x = 439,897.17 m, y = 4,489,478.87 m |
| Snap (3.3) | anchor: x = 439,920 m, y = 4,489,500 m (moved by +22.8 m, +21.1 m) |
| Window (3.4) | `S` = 10,560 m: x 434,640–445,200 m, y 4,484,220–4,494,780 m |
| Pixels | 1056 × 1056 at 10 m, 528 × 528 at 20 m, 176 × 176 at 60 m |
| Geotransform at 10 m | (434640, 10, 0, 4494780, 0, −10) |
| Coverage (3.5) | the cell is rotated 0.46° against the UTM axes; `cell_coverage` = 1.0; window area is 11.8% larger than the cell |

### 5.2 Sentinel-2 at `d` = 1 km: cell `4517U_9441L`

A smaller spacing keeps the rule unchanged but changes the trade-off, because the 60 m lattice is now coarse compared with the cell.

| Step | Result |
|---|---|
| Rows and columns | `N_rows` = 20,038; Snowbird is in row 4517U (40.575906°N), which has 30,439 columns; column 9441L (111.658070°W) |
| Cell | `4517U_9441L`; 0.998 km²; about 1,008 m × 1,005 m; rotated 0.42° |
| Anchor | x = 444,780 m, y = 4,492,380 m (moved by −26.1 m, −3.2 m) |

`S` must be a multiple of `2L` = 120 m, so the choices step in 12% increments of `d`:

| `S` | Pixels at 10 / 20 / 60 m | `cell_coverage` | Window area beyond the cell |
|---|---|---|---|
| 960 m | 96 / 48 / 16 | 0.918 | −7.7% |
| 1,080 m | 108 / 54 / 18 | 1.0 | +16.9% |
| 1,200 m | 120 / 60 / 20 | 1.0 | +44.3% |

At this spacing a profile may prefer a finer lattice: without the 60 m bands, `L` = 20 m, `S` steps by 40 m, and about 1,040 m (104 / 52 px) covers this cell with roughly 8% extra area.

## 6. Special cases

### 6.1 Exception zones (Norway, Svalbard)

UTM widens some zones: zone 32V around southern Norway, and zones 31X, 33X, 35X and 37X over Svalbard (each 9–12° wide). Cells far from such a zone's central meridian are rotated by up to about 6°.

- **Norway:** Sentinel-2 also has tiles in the neighbouring regular zone 31V, so the CRS rule of section 3.1 picks that zone and the rotation stays below 3°.
- **Svalbard:** Sentinel-2 has no tiles in zones 32X, 34X and 36X, so no better zone exists. With `S` = 10,560 m, 3,508 Core cells there are not fully covered: at most 258 m of a cell, 0.9% of its area, lies outside its window. These samples carry `cell_coverage` < 1 (section 6.3).

### 6.2 Pole caps

- **CRS:** beyond 84°N and 80°S there is no UTM; the CRS rule uses UPS (EPSG:32661 north, EPSG:32761 south).
- **Even `N_rows`** (e.g. `d` = 10 km): the southernmost row lies on the south pole and has one cell, a disc around the pole. Its window is centred **on the pole**, not on the cell's latitude/longitude midpoint, which lies one half-row (about 5 km) away. The northernmost row's cells are wedges meeting at the north pole; the general rule applies to them.
- **Odd `N_rows`** (e.g. `d` = 500 km): no cell covers the area south of the southernmost row, and the northernmost cells extend past the north pole. Datasets at such spacings state how they treat the caps.
- **Coverage:** even with these rules, square windows around wedge-shaped cells leave gaps and overlaps. With `S` = 10,560 m and midpoint anchors, 1.7% of the area within 32 km of the north pole and 17.5% around the south pole was not covered (figure `docs/animations/out/poles_1056.png`); the south figure improves with the pole-centred window above (to be measured).
- **Data:** Sentinel-2 images systematically up to 82.8°N, and Antarctica only on request, so for `s2` these rules matter only for other sources.

### 6.3 `cell_coverage`

The share of the cell's area that lies inside its window, between 0 and 1:

    cell_coverage = area(cell ∩ window) / area(cell)

computed in the window's CRS, with the cell's edges densified so that curved edges are followed. Every sample records it (section 8). For `s2` at `d` = 10 km with `S` = 10,560 m, 0.32% of Core cells have `cell_coverage` < 1, mostly in Svalbard; the worst value is about 0.99.

### 6.4 Windows that no single tile contains

The CRS rule prefers a native tile that contains the whole window. Where none does, the sample is built from adjacent tiles of the **same acquisition and the same CRS**, which share one pixel grid, so no resampling is needed; the sample records every source tile. Tiles in different CRSs are not mixed: the cell then uses the next CRS that works, or is skipped.

## 7. Legacy Core v1

*To follow:* bottom-left anchor, 1068 px, fractional-geotransform correction, 1068 → 1056 crop.

## 8. Required metadata per sample

*To follow.*

## 9. Open points

*To follow.*
