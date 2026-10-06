> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed. Built autonomously on Miko's request for the 2026-10-07 workshop; needs review.

# 2026-10-06 — majortom.build and the workshop notebook

`majortom.build` turns points or an area into Major TOM samples from 13 Planetary Computer collections, at any grid spacing, with optional per-sample dates and margins. It writes the `.npz` + `.json` layout of the taco workshop notebook, so the notebook's rumi, TACO and publishing sections run unchanged. 694 lines including tests. Pushed to `refactor/v1`.

## Flow

    points / bbox ──► cells.py: cells_from_points / cells_in ──► table, one row per cell (and query)
                                                                      │
    build.download(cells, collection, datetime) ── per row, in threads:
        spec.py: window(cell)  ──►  search.py: scenes ─► tiles_for (CRS + tiles)
        ──►  read.py: layout (pixel grid) ─► read_asset (mask first, ranks up to 5 scenes)
        ──►  chips/<collection>/<index>.npz + .json

## majortom/cells.py

Cells straight from the grid formulas (SPEC section 2), so any spacing is fast. `cell_at` wraps the column at 180°: in rows with an odd column count the easternmost cell crosses the antimeridian (`cells.py:68`).

    row = math.floor(lat / (180 / n_rows(d)))
    n = Cell(row, 0, d).n_cols
    col = (math.floor(lon / (360 / n)) + n // 2) % n - n // 2

`cells_from_points` groups points by cell and by the optional query columns `datetime`, `days`, `margin`; other columns (e.g. `class`) are kept from the first point (`cells.py:100`).

## majortom/spec.py

The window rule of SPEC section 3 (`spec.py:65`). The lattice origin is the product's own pixel grid when the product is in the window's CRS, else 0 (north), 10,000 km mod L (south), or through the pole (UPS).

    x = x0 + lattice * round((x - x0) / lattice)
    y = y0 + lattice * round((y - y0) / lattice)
    side = 2 * lattice * math.ceil((1 + margin) * cell.d * 1000 / (2 * lattice) - 1e-9)

## majortom/build/search.py

One STAC search per sample over the window, sorted by cloud (or newest for static products); results grouped into scenes by acquisition time (`search.py:61`). `tiles_for` picks the window's CRS per scene, as SPEC 3.1 says: the cell's UTM zone if its tiles' footprints cover the window, else another projected zone that does; latitude/longitude products keep the cell's zone. One CRS per sample, never mixed (`search.py:80`).

    for crs in projected:                     # cell's zone first
        need = shapely.box(*transform_bounds(crs, "EPSG:4326", *window(...).bounds, densify_pts=21))
        ...
        if shapely.union_all(list(footprints.values())).contains(need):
            return crs, sorted(by_crs[crs], key=lambda it: not footprints[it.id].contains(need))

## majortom/build/read.py

Every read goes through a WarpedVRT onto the window grid (`read.py:57`). Same CRS and window on the product's grid: nearest neighbour, an exact copy. Otherwise the collection's method. The alpha band marks covered pixels, used for stitching and no-data.

    with WarpedVRT(src, crs=win.crs, transform=win.transform(res), width=n, height=n,
                   resampling=method, add_alpha=True) as vrt:
        tile = retry(vrt.read)                       # bands, then alpha (0 = not covered)
    values, alpha = tile[:-1], tile[-1] > 0

## majortom/build/__init__.py

`pick_scene` reads the cloud mask (or the first asset) of up to 5 scenes and takes the first with no-data ≤ `max_nodata` (1%) and cloud ≤ `max_cloud` (5%), else the best of them (`__init__.py:61`). Mask functions return clouds and fill, so fill pixels inside a file count as no-data. Each `.json` holds the notebook's fields plus `grid_cell`, `grid_km`, `window_m`, `cell_coverage`, `nodata`, `profile`, `resampled`, `query`.

## Verification

- `tests/test_spec.py`, 8 tests, offline: SPEC 5.1 window exactly; 1 km sides; `0U_0R` / `1D_1L`; 200 random points equal to `Grid.latlon2rowcol`; antimeridian; grouping by query; `cells_in`; south-pole window centred on the pole. All 13 tests pass.
- Real build, 6 ski resorts × 5 collections: 29 of 30 samples. The failure is correct: NASADEM has no data north of 60°N.
- Exactness: Sentinel-2 B04 from the builder equals a direct window read of the source file, pixel for pixel.
- Notebook sections 3–6 as a script in Python 3.12 with `taco-eo` 0.14.2 and `rumi-eo`: 8 golf-course cells, 0 failures, TACO ZIP 129.8 MB, `valid=True`.
- `pip install "majortom[build] @ git+https://github.com/ESA-PhiLab/Major-TOM@refactor/v1"` in a fresh env: installs, imports without torch.

## Bugs found and fixed on the way

- South-pole anchor 20 m off: UPS lattice now passes through the pole.
- My `cell_at` produced a non-existent column at 180° in odd rows; the official grid was right.
- Stitching filled 5 no-data pixels from a tile in another UTM zone (resampled): stitching now stays in one CRS.
- A Landsat scene 99% empty passed as 0% no-data: `qa_pixel` has no nodata value; the fill bit now counts.
- Official `Grid.latlon2rowcol` returns the northernmost row for points south of the lowest generated row (default ±85°). Not fixed (grid frozen); for WS2 docs.

## workshop notebook: docs/workshop/workshop.ipynb

Copy of `asterisk-labs/taco` `examples/workshop.ipynb` (2026-09-29). Changed cells only: 3 (install `majortom[build]`), 4 and 6 (text), 7 (`from majortom import build` replaces 13 kB of helpers), 8 (cells and download), 17 (`dist_km=GRID_KM`, description), 19 (`to_taco(..., cells, ...)`).

## Decisions

- One read path (WarpedVRT) over separate native and resampled paths: exact for native, simpler to review.
- Window CRS per scene by native coverage over always the cell's zone: keeps pixels native at zone edges (Björkliden: zone 33, not resampled).
- Latitude/longitude products resampled to the cell's UTM grid: square samples in metres, aligned across collections.
- Cell-based samples (1056 px for Sentinel-2) over the notebook's 264 px point chips: each sample is a whole cell.
- Default `N_SAMPLES` 100 → 30 and advice "50 or fewer": 1056 px samples are ~16 MB each.
- Own thin builder over aereo/odc-stac for now: WS5 decision gate still open.

## Known gaps

- Planetary Computer only; Earth Search and CDSE catalogues not yet in `build`.
- `cells_in` does not cross 180°.
- Collections with assets of different CRSs within one item are not supported (none of the 13).
- SSH to github.com on port 22 times out from cirrus since today; pushes went through port 443: `GIT_SSH_COMMAND="ssh -o Hostname=ssh.github.com -o Port=443 -o HostKeyAlias=github.com" git push`.

## Look at first

- majortom/build/search.py:80 (`tiles_for`, the CRS choice)
- majortom/build/read.py:57 (`read_asset`)
- docs/workshop/workshop.ipynb cell 8
