> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed. LLM-generated; verify numbers and claims before relying on them.

# 2026-09-30 — Refactor research: geotransforms, sample windows, storage, tooling

Findings that feed `docs/STRATEGY.md`. Scripts are in `docs/evidence/`. Large intermediate data (Core metadata, index copy, per-cell CSV) was not committed; the scripts regenerate it.

## 1. Core geotransforms are fractional in every sample

Checked 43 Core-S2L2A and 12 Core-S2L1C samples by HTTP range reads (`docs/evidence/check_geotransform.py`, `analyse.py`). 55/55 have origins that are not multiples of 10 m.

- Offset to the native grid: mean 2.5 m, max 5.0 m (10 m bands); max 9.9 m (20 m); max 29.9 m (60 m).
- All bands carry the same origin. 20 m and 60 m bands are therefore also misregistered relative to each other.
- Shape 1068×1068 at 10.0 m for all samples. CRS matches the product's MGRS zone in 55/55. No reprojection.
- The stored origin equals the unsnapped formula (grid point → UTM, −340 m) to 1e-9 m in 55/55.

Pixel ground truth on one sample (`pixel_match.py`, cell `48U_704L`, product `S2A_MSIL2A_20230826T143731_N0509_R096_T20NMK`) against the Earth Search COG:

    origin at native pixel (col 5447.636, row 1314.216)
    B04 == COG[1314:, 5448:] + 1000   # 100% of pixels
    B11 matches at (2724, 657), B01 at (908, 219)

Pixels are native and unresampled, read at the rounded offset. The header records the unrounded offset. True origin for this sample is +3.64 m x, −2.16 m y from the stored one.

Root cause, `MajorTOM/extras/extract-sample-from-raw-S2.ipynb` (JSON lines):

    left, bottom = transformer.transform(lon, lat)          # ~L102, float, never snapped
    left, bottom = left - box_offset, bottom - box_offset   # box_offset = 340
    src.read(window=from_bounds(*window.bounds, src.transform))  # ~L179, fractional window

Fix without touching pixels: per band of resolution r, `origin' = round(origin / r) * r` (native tile corners are multiples of 60 m; see caveat in §3).

Caveat: pixel-exact check covers one product. Repeat on ~10 products (L1C, southern hemisphere, zone edge) before announcing.

## 2. Three competing window definitions

| Source | Anchor | Size | Snap | CRS |
|---|---|---|---|---|
| Core (HF) | bottom-left grid point −340 m | 1068 | none (float) | product's MGRS zone |
| `MT_OFFICIAL_GRID_v2.ipynb` (untracked) | bottom-left grid point −340 m | 1068 | `np.rint`, 1 m | centroid's UTM zone |
| ELLIOT index (source.coop) | cell centroid −5280 m | 1056 | nearest S2 60 m lattice | assigned MGRS tile |

The 1 m rounding in the v2 notebook does not fix the problem; pixel lattice is 10/20/60 m.

Paper (arXiv 2402.12095 §3.1) does not state bottom-left or centre. It calls grid points "anchors" and says extents are source-dependent and "overlap slightly". Bottom-left comes only from code comments in `MajorTOM/grid.py`.

## 3. ELLIOT index construction (verified by recomputation, `compare.py`)

- Centroid = lat/lon midpoint of the cell (error < 1e-13°).
- Raw origin = centroid projected to `majortom:crs`, then (x − 5280, y + 5280). Error 0.0 m.
- Snap: nearest point with x ≡ 0 mod 60 and y ≡ 0 mod 60 (north) or ≡ 40 mod 60 (south, 10,000 km false northing). Max shift ±30 m.

Window size is 1056, not 1052. 1056 = 2⁵·3·11 = 3 × 352. Divisible by 6 (10/20/60 m align), by 3 (30 m Landsat/DEM align), and splits into 3×3 blocks of 352 px with no padding. 1068 = 2²·3·89 pads any 16-multiple block size. 1052 = 2²·263 breaks 60 m alignment.

Index defects found:

- 2.9% of cells: `majortom:mgrs_tile` is in the western neighbour zone of `majortom:crs` (e.g. `11WPR` with `EPSG:32612`).
- `MT_grid_10km_alpha.parquet`, cell `MT10_890D_345L` (~80°S) has `code_1000km = MT1000_9U_4R`. Probable sign loss. Not verified against the generator.
- 5,120 Core cells (|lat| > ~72°) are absent from the index.

## 4. v1 (Core) vs v2 (centroid) on real Core cells

`core_cells.py`, all 2,245,886 Core-S2L2A samples (2,233,569 cells present in the index), in each sample's product CRS.

| Metric | Value |
|---|---|
| v1 window misses part of its own cell | 8.32% of cells (186,303 samples) |
| v1 missed area, among misses | median 0.08%, p95 0.88%, max 13.2% of cell (1,716 m) |
| v2 window misses part of its cell | 0.37% |
| v2 croppable from v1 (same CRS, inside) | 47.92% of cells (1,073,813 samples) |
| CRS differs v1/v2 | 3.61% |
| Origin offset v2−v1 | median (+60, +30) m; p95 \|dx\| 230 m, \|dy\| 210 m |
| Overlap v1∩v2 / v2 | median 99.91%, p5 97.46% |

By latitude, v1 misses rise from 0% below 30° to 20% at 50–60° and 38% at 70–80°. By distance from the central meridian: 0% within 1°, 21% at 2–3°, ~100% beyond 4°. Cells beyond 3° (3.7%) are products in a neighbouring or exception zone; they hold the worst misses.

Cause: meridian convergence rotates the lat/lon cell against UTM axes. Bottom-left anchoring puts the full 10 km lever arm on one side (~175 m drift per degree); centroid anchoring splits it (~87 m per degree).

## 4b. Why 1056 px windows still miss (`v2_misses.py`, figures in `docs/figures/`)

8,239 Core cells (0.37%) are not fully contained by their 1056 px window.

| Cause | Cells | \|γ\| | Max miss | Fixable natively |
|---|---|---|---|---|
| Exception zones | 4,518 | 2.9–5.9° | 258 m, 0.88% of cell | Norway (1,010) via zone-31 tiles; Svalbard (3,508) no: no S2 tiles in 32X/34X/36X |
| Index tile/CRS mismatch | 362 | 2.6–3.0° | 26 m | no, already nearest zone |
| Regular zone, > 62° | 3,359 | 2.3–3.0° | 50 m, 0.02% of cell | no |

Required side follows from `s·(cos γ + sin γ) + 60 m`. γ ≤ 3° in regular zones gives ~1061 px (measured max 1066); γ ≤ 6° in exception zones gives ~1109 px (measured 1107.5). 100% containment with native S2 needs 1152 px (+19% pixels). Windows duplicate ~12% of area at 1056 px, including at the extreme cells (fig4). Neighbours cover 90% of regular-zone slivers and 60% of exception-zone slivers.

Chosen: 1056 with `cell_coverage` flags. Rejected: 1152 (duplication everywhere to fix 0.3% of cells).

## 5. Tooling and hosting

- PyPI: `majortom`, `major-tom` unregistered (404). Closes issue #20 once published.
- `majortom-eg` (Earth Genome): unofficial. Not a dependency (decision 2026-09-29).
- phidown (ESA-PhiLab, Apache-2.0): CDSE/PhiSat-2 search and whole-product download. No windowed reads.
- aereo (Apache-2.0): Major TOM-aligned extraction, depends on `majortom-eg`.
- TACO (`taco-eo`, MIT, spec v3): container + parquet metadata. Core-DEM already published as TACO v3 + rumi on source.coop (1,735,044 samples, 24 GB), rebuilt from source COGs.
- rumi (`rumi-eo` 0.26.1): GPL-3.0, spec 0.1.0 draft, no GDAL driver, wheels Linux x86-64 and macOS arm64 only.
- Workshop draft (`asterisk-labs/taco/examples/workshop.ipynb`) §4 is a point-centred prototype of `majortom.build` on Planetary Computer only; 264 px chips.

Backends for S2 L2A:

| Backend | Format | Auth | Windowed read | +1000 offset |
|---|---|---|---|---|
| Earth Search `sentinel-2-c1-l2a` | COG | none | yes | declared in `raster:bands` |
| CDSE | JP2 in SAFE | account + S3 keys | yes, slow decode | declared |
| Planetary Computer | COG | free SAS | yes | not declared; derive from `s2:processing_baseline` |

CDSE quota: 4 connections × 20 MB/s, 12 TB / 30 days. Planetary Computer is the only one with S1 RTC.

Hosting 60 TB: 1 Gbit/s ≈ 5.6 days line rate; 10 Gbit/s ≈ 13 h. HF public storage beyond PRO/Team quotas is ~$10–12/TB/month or grant. Source Cooperative is free during beta; Major TOM org exists.

## 6. Other repo findings

- `MajorTOM/metadata_helpers.py` `read_row`: `open_parquet_file` without `row_groups=`, prefetches every row group.
- `MajorTOM/sample_helpers.py` `plot`: ignores `scaling`.
- `MajorTOM/embedder/grid_cell_fragment.py` `fragment_fn`: square only; row/col offsets and `pixel_bbox` order look swapped. Not yet verified.
- Stash `stash@{0}` on `main`: 05 notebook embedding reproducibility check (local SSL4EO vs `Core-S2L1C-SSL4EO`, cosine similarity) and embedder import re-enabled. Unfinished (`df_head`, `df_local` undefined).

## Decisions

- Centroid anchor over bottom-left: at the same 1068 px, 0.16% vs 8.32% of Core cells not contained (`fig3_tradeoff.png`, index CRS; snap and CRS still differ). Paper does not mandate bottom-left.
- 60 m lattice snap over 1 m rounding: only the former makes 10/20/60 m bands align.
- Core v1.1 on HF (header fix + quality sidecar, no row removal) plus v2 on source.coop, over in-place rumi conversion: keeps `parquet_row` indexes and GDAL compatibility.

## Look at first

- docs/figures/fig1_cells_1056.png, fig3_tradeoff.png
- docs/evidence/core_cells.py (v1 vs v2 metrics)
- docs/evidence/pixel_match.py (pixel ground truth)
- MajorTOM/extras/extract-sample-from-raw-S2.ipynb (root cause)
