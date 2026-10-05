> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed. GIF 1 was walked through and approved step by step; GIFs 2–4 were made in one batch on request and need review.

# 2026-10-04 — Tutorial GIFs: grid, anchoring, multiples of 12, trade-off

Four looping GIFs for the Major TOM tutorial in `docs/animations/out/`, Asterisk Labs style (palette Ink/Paper/Indigo/Mint/Yellow, League Spartan). No logos. Not committed.

## Pipeline

    MajorTOM/grid.py ──► geometry.py ──┐   (grid, 10 km cells, two window algorithms)
    tradeoff_curve.csv ────────────────┤   (GIF 4 only)
                                       ▼
    gifN_*.py: draw(t), t in [0, 1)  ──► style.render_loop ──► shared palette ──► write_gif ──► verify_gif
                                                                                     └─► out/gifN_*.gif

Every script has the same shape: `draw(t)` returns one matplotlib figure; `render_loop` calls it for `t = i/n`.

## Results

| GIF | Loop | Frames | Size | Seam | Largest frame step | Decode check |
|---|---|---|---|---|---|---|
| 1 grid | 12 s | 240 | 1.6 MB | 0.000 | 0.45 | 0 px error |
| 2 anchoring | 14 s | 280 | 2.6 MB | 0.000 | 2.58 (windows shrinking) | 0 px error |
| 3 multiples of 12 | 16 s | 320 | 1.2 MB | 0.000 | 3.29 (2 px steps, scene B) | 0 px error |
| 4 trade-off | 18 s | 360 | 1.7 MB | 0.000 | 1.27 | 0 px error |

Seam = mean pixel change from the last frame back to the first. 0.000 because every loop starts and ends in the same rest state. Largest steps are motion, not jumps.

## docs/animations/style.py

Shared look and the loop writer. Timing is declarative: each element's visibility is a `phase` that rises and later falls back.

    def phase(t, start, end):                       # style.py:82
        return ease((t - start) / (end - start))    # smoothstep, 0 before start, 1 after end
    ...
    rows_on = phase(t, 0.08, 0.30) - phase(t, 0.82, 0.92)   # gif1_grid.py: grow, later retract

Frames are written as changes only (`style.py:155`). Pixels equal to the previous frame get palette slot 255 (transparent); the viewer keeps the previous frame underneath (`disposal=1`).

    for prev, cur in zip(frames, frames[1:]):
        a = np.asarray(cur)
        delta = Image.fromarray(np.where(a == np.asarray(prev), CLEAR, a).astype(np.uint8), mode="P")
    ...
    deltas[0].save(path, save_all=True, ..., transparency=CLEAR, disposal=1, optimize=False)

GIF 1 went from 8.4 MB to 1.6 MB with this. `verify_gif` (`style.py:171`) decodes the file and compares every shown frame with the rendered one; all four GIFs report 0 error.

The palette (`style.py:97`) is shared by all frames: brand colours exact, 247 slots fitted to sampled frames, slot 255 magenta (never used by a pixel).

## docs/animations/geometry.py

Grid code comes from `MajorTOM/grid.py`, loaded from its file (`import MajorTOM` pulls in torch). 10 km cells use the same rule per cell, because `Grid(10)` builds 5M points:

    lat = r * DLAT_10KM                                                       # geometry.py:83
    n_cols = math.ceil(2 * math.pi * R_KM * math.cos(math.radians(lat)) / 10)
    dlon = 360 / n_cols
    ...
    return Cell(name, c * dlon, lat, dlon, DLAT_10KM)

`python geometry.py` checks this against grid.py's own `get_rows` / `subdivide_circumference` on 10 cells: max difference 1.42e-14°.

The two anchors (`geometry.py:150`, `:161`), both with the 60 m snap optional:

    x, y = transformer("EPSG:4326", crs).transform(cell.lon, cell.lat)        # bottom-left: grid point
    x0, y0 = (x - margin, y - margin)                                          # margin = (side - 10 km) / 2
    ...
    cx, cy = transformer(...).transform(cell.lon + cell.dlon / 2, cell.lat + cell.dlat / 2)   # centroid
    return shapely.box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

## docs/animations/gif2_anchoring.py

A cell of row 902U (81°N) slides east inside zone 33X, from the central meridian (15°E) to the zone edge (21°E). Panels are drawn relative to the cell centre, so the cell rotates in place. Both windows 1056 px, unsnapped.

    def cell_at(dl):                                   # gif2_anchoring.py:36
        return BASE._replace(lon=CM + dl - BASE.dlon / 2)
    ...
    missed_pct = 100 * poly.difference(full).area / poly.area

Measured in the GIF: at 2.9° rotation bottom-left misses 0.87%, centre 0.00%; at 5.6° bottom-left 4.61%, centre 0.73%. Caption switches at 3°: rotation beyond that only occurs in the Norway/Svalbard zones.

## docs/animations/gif3_multiples_of_12.py

Three strips of the same 480 m: 10, 20, 60 m pixels. A pixel is mint if whole inside the window, yellow if an edge splits it (`gif3_multiples_of_12.py:56`).

    if x1 - x0 > 1e-6 and x0 <= p0 + 1e-6 and p1 <= x1 + 1e-6:
        states.append("in")
    elif x1 - x0 > 1e-6 and p0 < x1 - 1e-6 and p1 > x0 + 1e-6:
        states.append("cut")

Scene A: window from a corner at 120 m, 1 px steps to 12 → all strips clean at 6 and 12. Scene B: window around 300 m, 2 px steps to 24 → clean at 12 and 24. Scene C: sizes 1044–1152 with ÷12 and ÷16 badges; 1056, 1104, 1152 pass both.

## docs/animations/gif4_tradeoff.py

Two 3×3-cell patches (45°N, 13.5°E; 81°N at the 33X/35X zone edge) coloured by how many centroid windows cover each 30 m pixel. Windows from the neighbouring zone are reprojected into the patch CRS. Size schedule: 1056 → 1152 → 960 → 1056 in 12 px steps.

The curve comes from the research table once (`gif4_tradeoff.py:46`):

    no_tiles = (d.lat >= 72) & zone.isin([32, 34, 36])          # Svalbard: no S2 tiles in the plain zone
    side_m = np.where(no_tiles, d.side_index, d.side_best)      # smallest window, best CRS with S2 tiles
    ...
    rows = [(s, 100 * (side_m > s * 10).mean(), 100 * ((s * 0.01) ** 2 / area_km2).mean() - 100) for s in SIZES]

Values match the earlier research: 0.3236% of cells not covered at 1056, 0 from 1116; extra area +11.6% at 1056, +32.8% at 1152, −7.8% at 960.

## Decisions

- Static globes over a swaying view (GIF 1): 20.4 MB → 8.4 MB before delta writing, and easier to read.
- Toy spacing 1,000 km over 500 km (GIF 1): polar crowding on the plain grid becomes the visible contrast.
- Changed-pixels-only GIFs over full frames: 8.4 → 1.6 MB for GIF 1; checked by decoding.
- One palette, no dithering, over per-frame palettes: no flicker, also across the seam.
- GIF 2 sweep inside one zone over a jump to Svalbard: no visual cut; rotation runs 0 → 5.6°.
- GIF 2 without snapping: the 60 m snap made windows jitter 2–3 px; it moves windows ≤ 30 m.
- GIF 3 one-dimensional strips over overlaid 2-D lattices: one reading per band.
- GIF 3 ÷16 badge over ÷48: ÷16 is the GeoTIFF tile rule; ÷48 is ÷12 and ÷16 together.
- GIF 4 sweep down to 960 px over 1008: holes are under 3 px wide above ~1000 px at patch scale.
- GIF 4 negative "extra area" shown, not clipped: below ~1000 px the window is smaller than its cell.
- League Spartan committed with `OFL.txt` over a user-folder install: GIFs rebuild anywhere.

## Known gaps

- GIF 1 land outlines come from geopandas 0.13's bundled Natural Earth data, removed in geopandas 1.0.
- GIF 4 patches pick CRS with grid.py's zone rule; the curve uses "best CRS with S2 tiles". They differ only in the Norway exception zone (not shown).
- "Extra area" is window area over cell area minus 1, not measured overlap.
- `tradeoff_curve.csv` depends on `v2_misses_cells.parquet` (scratchpad, not committed), produced by `docs/evidence/v2_misses.py`, itself unreviewed.
- GIF 1 highlights the cell above and right of its point, following the code's bottom-left convention.

## Look at first

- docs/animations/style.py:155 (changed-pixels GIF writing)
- docs/animations/geometry.py:150 (the two anchoring algorithms)
- docs/animations/gif4_tradeoff.py:46 (curve export from the research table)
