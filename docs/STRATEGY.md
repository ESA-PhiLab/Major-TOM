> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Major TOM refactor strategy

Branch: `refactor/v1`. Evidence: `docs/reports/2026-09-30-refactor-research.md`.

## 1. Goals

Main aim: clean, documented code for the next generation of the Major TOM ecosystem (TACO, rumi, new datasets). Legacy Core data is documented and read correctly; its known issues are fixed by the next generation, not by re-engineering the old files.

- Make Major TOM an installable PyPI package (`majortom`) with a light core and optional extras.
- Define one sample-window specification (v2) and correct the georeferencing of existing Core data.
- Provide quality reports for any Major TOM dataset.
- Provide `majortom.build` for building new Major TOM datasets from public archives.
- Publish Core v1.1 on HF and Major TOM v2 on source.coop.
- Supply a Major TOM workshop notebook built on the package.

Non-goals:

- Depending on or following `majortom-eg`. The authority is this repo and arXiv 2402.12095.
- TACO- or rumi-specific code in this repo. It belongs in `asterisk-labs/taco`.
- Deleting rows from published Core datasets.

## 2. Principles

- The Major TOM **grid** is an indexing system. It covers the whole Earth.
- A Major TOM **dataset** does not have to cover every point. Each dataset balances duplication, consistency and omission, and states which balance it chose.
- The **point grid is frozen.** Cell names are the shared key across all Major TOM datasets; nothing in the refactor changes which points exist or how they are named. Irregularities are documented and handled in the sample-window spec, not in the grid.
- 1056 px is the default for most applications. Other datasets may choose differently, and must document the choice using the trade-off in `docs/figures/fig3_tradeoff.png`.

## 3. Decisions

| # | Decision | Date |
|---|---|---|
| D1 | Import name `majortom`; `MajorTOM` remains a working alias. No breaking changes. | 2026-09-29 |
| D2 | No dependency on `majortom-eg`. Grid follows the original implementation and paper. | 2026-09-29 |
| D3 | New datasets use 1056 px; 1068 is legacy Core. Provide a 1068→1056 helper. | 2026-09-29 |
| D4 | Optional extras, not separate packages. | 2026-09-29 |
| D5 | Main notebooks in root, minor examples in `MajorTOM/extras/`. | 2026-09-29 |
| D6 | Every AI-generated document carries a label and review status. | 2026-09-29 |
| D7 | Core v1.1 on HF (corrected headers if cheap, quality sidecar), v2 on source.coop. | 2026-09-30, proposed |
| D8 | v2 window: centroid anchor, 1056 px, S2 60 m lattice snap, native S2 CRS chosen by rule. Cells not fully contained (≈0.3%, mostly Svalbard) are flagged with `cell_coverage`, not fixed. | 2026-09-30 |
| D9 | v2 payload: COG primary; rumi optional. | 2026-09-30, proposed |
| D10 | The point grid is frozen: no change to points, rows, columns or names. | 2026-10-05 |
| D11 | Legacy Core issues that need new files (e.g. slow cell lookup) are listed as known issues and solved in the next generation, not by re-publishing Core. | 2026-10-05 |

## 4. Workstreams

### WS0 — Scaffolding

`CONTEXT.md`, `CHECKLIST.md`, `LOG.md`, `docs/`, `.gitignore` for large local files.

### WS1 — Packaging and PyPI (issues #20, #18, #14)

- `pyproject.toml`. Package directory becomes `majortom/`; `MajorTOM` is a shim that re-exports it.
- Extras:

| Extra | Adds | Enables |
|---|---|---|
| core | numpy, pandas, geopandas, shapely, pyproj, pyarrow, fsspec, rasterio | grid, metadata, sample I/O, transform correction |
| `[torch]` | torch, torchvision | `Dataset`, fragmenting |
| `[embed]` | `[torch]` + timm, torchgeo, open-clip-torch, transformers | embedders |
| `[viz]` | matplotlib, cartopy | plotting, thumbnails, coverage maps |
| `[build]` | pystac-client, odc-stac, odc-geo, planetary-computer | `majortom.build` |
| `[all]` | all of the above | |

- Heavy submodules import lazily; a missing extra raises an `ImportError` naming the extra.
- CI: tests on 3.10–3.12, build wheel, publish on tag.

### WS2 — Grid (issues #19, #20)

- Document grid construction against paper §3.1, including the `linspace + mod` equator shift (#19).
- Add `cells_in(polygon)` and point→cell lookup without per-row pandas loops (#20).
- Document the column-count asymmetry, with examples (figure: `docs/animations/out/poles_1056.png`). Each row's column count comes from its own latitude, the cells' bottom edge: the wide edge in the north, the narrow edge in the south. Cells are therefore slightly smaller than 10 x 10 km in the north and slightly larger in the south:

| Latitude | North vs south size difference |
|---|---|
| 45° | 0.16% |
| 60° | 0.3% |
| 80° | 0.9% |
| innermost polar ring | north 1001U: 7 cells of ~45 km²; south 1002D: 1 disc cell of ~314 km² |

  Consequence for a fixed window: slightly more overlap in the far north, slightly more omission in the far south; large only at the poles. Not fixed (D10).
- Add `code_100km` / `code_1000km` if adopted by the spec; verify the sign convention (see §5).
- Tests against paper formulas and against the published Core `grid_cell` values. Not against `majortom-eg`.

### WS3 — Sample-window specification and Core correction

- `docs/SPEC.md`: cell → CRS → integer geotransform → window, for any size and resolution set.
- v2 rule (D8). CRS rule: among S2 tiles containing the snapped window, choose the zone whose central meridian is nearest the centroid; tie-break by tile id. Resolves the index's 2.9% CRS/tile mismatch.
- Exception zones: the CRS rule moves Norway cells to native zone-31 tiles. Svalbard cells (3,508) have no native tile in a nearer zone; they keep a `cell_coverage` < 1 flag. Figures: `docs/figures/`.
- Reference sensor: S2 keeps native pixels. Other sensors (Landsat, DEM, S1) are reprojected onto the cell grid and flagged, or keep their own window. To decide.
- Recommend spatial splits by `code_100km`, not by cell: windows overlap ~12%.
- Pole caps: the v2 rule centres a window on the lat/lon midpoint, which for the south-pole disc cell (1002D) is 5 km off the pole. The spec defines pole-cap cells explicitly (e.g. centre on the pole) and states that UPS replaces UTM beyond 84°N / 80°S.
- `majortom.spec` functions:
  - `sample_window(cell, size_px, snap_m)` → CRS, geotransform.
  - `correct_core_transform(transform, band_res)` → snapped transform for v1 samples.
  - `crop_1068_to_1056(sample)` → cropped arrays or `NotCroppable` (48% of Core cells are croppable).
- Readers (`read_row`, `Dataset`) apply `correct_core_transform` by default.
- Acceptance: pixel-match test on ~10 products, covering L1C, southern hemisphere, zone edges.

### WS4 — Quality reports (issues #17, #8)

- Per-sample checks: nodata fraction, all-zero bands, shape (S1RTC has 854², 1424², 1068×1069), value ranges (DEM negatives), duplicate `grid_cell`/`product_id`, metadata/image mismatch, geotransform alignment, v1 cell coverage, croppable-to-v2.
- Output: `quality.parquet` keyed by (`grid_cell`, `product_id`), plus a summary Markdown report.
- `filter_metadata(..., quality="clean")`.
- Full-dataset runs as Slurm CPU jobs (`~/slurm-templates/cpu.sbatch`), `ProcessPoolExecutor` sized by `sched_getaffinity`.

### WS5 — `majortom.build`

Pipeline, each stage swappable:

    cells   = grid.cells_in(aoi)
    spec    = SampleSpec(size_px=1056, snap_m=60)
    builder = Builder(catalog=STACCatalog("planetary-computer", "sentinel-2-l2a"),
                      selector=LowestCloud(max_cloud=20), spec=spec)
    plan    = builder.plan(cells, time=("2024-01", "2024-12"))
    builder.run(plan, out=..., executor="local" | "slurm")
    builder.report(out)   # WS4

- **Decision gate before implementation:** how products are read. Options: existing packages (aereo, odc-stac/odc-geo, phidown for CDSE/PhiSat-2) or custom readers ("binders") per archive. Choose with a short comparison against the measurements in `docs/reports/2026-10-05-acquisition-v2.md`; the rest of this section is finalised after that.
- Catalogs: Earth Search, Planetary Computer, CDSE STAC. phidown as optional CDSE/PhiSat-2 backend.
- Rate limits and retries: retry with exponential back-off on HTTP 429/5xx; batch searches by tile and date, not per cell; cache search results. Measured: CDSE STAC returned 429 in 2 of 4 searches of one small test.
- Source preference per product: COG archives for window reads (12 requests per 4-band window); CDSE JP2 only where needed (~225 requests and JPEG 2000 decoding per window), e.g. L1C or a specific reprocessing.
- Provenance per sample, easy to access: product ID, processing baseline, archive and collection, acquisition time; raster metadata (CRS, geotransform, band names, dtype, nodata, offset/scale actually applied); dataset metadata (spec version, window size, build date, code version).
- Reader: windowed read at integer offsets; no resampling when CRS matches, else reproject and flag.
- Offset normalisation: Earth Search and CDSE declare `raster:bands` offset; Planetary Computer requires `s2:processing_baseline >= 04.00` → subtract 1000.
- Writers: COG (352 px blocks) and Core-style parquet shards + `metadata.parquet`. Writer returns arrays + geotransform so TACO/rumi writers in the taco repo can consume them.
- v1→v2 path: for croppable cells, derive v2 from Core v1.1 instead of re-downloading.

### WS6 — Code cleanup and embedder

- Module names to snake_case with aliases for old names. Explicit imports; no `import *` inside the package.
- Bugs: `plot` scaling, `read_row` row-group prefetch, `tqdm.notebook` import, `filter_download` column list, README `src/grid.py` link.
- Embedder: lazy import (fixes the re-enable in `stash@{0}`), verify `fragment_fn` offsets and `pixel_bbox` order, support non-square tiles, one pyproj `Transformer` per sample.
- Acceptance: finish the 05-notebook reproducibility check as a test — regenerated SSL4EO embeddings match `Core-S2L1C-SSL4EO` (cosine ≥ 0.999).
- Notebooks: main ones stay in root; minor ones move to `MajorTOM/extras/`. Strip outputs.

### WS7 — Major TOM workshop notebook

Draft in this repo; Miko finalises. Replaces §4 of `asterisk-labs/taco/examples/workshop.ipynb` with `majortom.build`. Sections:

1. Grid and cells.
2. Why windows matter: one high-latitude zone-edge cell showing the cell outline, v1 and v2 windows, and the 60 m lattice.
3. World map of Core cells by status: v1 fine / v1 misses cell / v2 croppable / v2 needs re-read.
4. Hands-on: v1 sample before and after `correct_core_transform`, overlaid on the source COG.
5. Backends compared: Earth Search, CDSE, Planetary Computer (auth, format, offset, quota).
6. Build a small v2 dataset; hand off to TACO/rumi packaging (taco repo code).

### WS8 — Releases

- Core v1.1 on HF: first test whether Xet chunk dedup keeps a header-patched parquet re-upload small (one file). If yes, patch headers; if no, ship `quality.parquet` + corrections sidecar only. New revision; no rows removed.
- Major TOM v2 on source.coop: built with WS5, TACO container, COG payload, optional rumi.

## 5. Known issues of legacy Core (not fixed in the data)

- **Slow cell lookup.** Finding a cell means querying `metadata.parquet` (173 MB, 4,492 row groups). Filtering on the text `grid_cell` reads most of the file (~11 s): alphabetical min/max statistics cannot skip row groups. Filtering on the integer `grid_row_u`/`grid_col_r` takes ~2.5 s, mostly the footer. The library uses the integer filter (WS1); a proper index comes with the next generation (TACO metadata).
- **Fractional geotransforms** in every sample: corrected at read time (WS3).
- **Duplicate cells and nodata samples** (#17, #8): flagged by quality reports (WS4).

## 6. Sequence

    WS0 → WS1 → WS3 spec → WS2 + WS3 functions → WS4 → WS5 → WS7 → WS8
    WS6 runs alongside each step, in the files it touches.

## 7. Open questions

- With the focus on the next generation (D11), does D7 (Core v1.1 on HF with corrected headers and a quality sidecar) still stand, or do we limit Core to reader-side corrections and a known-issues list?
- Product reading (WS5 decision gate): existing packages or custom binders?
- Reference-sensor rule for non-S2 sensors (reproject onto S2 grid, or own window per sensor)?
- 5,120 Core cells above ~72° are missing from the ELLIOT index. Include them in v2?
- `code_1000km` sign in `MT_grid_10km_alpha.parquet` (e.g. `890D_345L` → `MT1000_9U_4R`). Bug or convention?
- Report ELLIOT index defects (2.9% CRS/tile mismatch) to its authors?
- HF storage status for the Major-TOM org (grant or paid) before a v1.1 re-upload.
- rumi licence (GPL-3.0) acceptable as an optional dependency?
- Does S1RTC / DEM v2 follow the same 60 m snap, or its own native lattice?

## 8. References

- Paper: https://arxiv.org/abs/2402.12095
- Core datasets: https://huggingface.co/Major-TOM
- ELLIOT index: https://source.coop/major-tom/index
- ELLIOT pretrain: https://source.coop/major-tom/elliot-pretrain
- Issues: #17, #19, #20 at https://github.com/ESA-PhiLab/Major-TOM/issues
- TACO: https://github.com/asterisk-labs/taco · rumi: https://github.com/asterisk-labs/rumi
- phidown: https://github.com/ESA-PhiLab/phidown · aereo: https://github.com/frandorr/aereo
