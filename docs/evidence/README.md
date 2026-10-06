> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Evidence scripts

Scripts behind `docs/reports/2026-09-30-refactor-research.md`. They write outputs next to themselves; large outputs are not committed.

- `check_geotransform.py` — range-reads B04/B11/B01 from random Core rows, writes `results_*.jsonl`.
- `analyse.py` — summarises origin offsets from `results_*.jsonl`.
- `pixel_match.py` — compares one Core sample to the source COG pixel by pixel.
- `compare.py` — rebuilds ELLIOT index origins; v1 vs v2 cell coverage on an index sample.
- `core_join.py` — joins Core-S2L2A metadata to the index.
- `core_cells.py` — v1 vs v2 metrics on all Core-S2L2A cells; writes `core_cells_v1_v2.csv`. Run via Slurm.
- `v2_misses.py` — causes of 1056 px misses, CRS rule test, minimal window size, neighbour coverage; writes `v2_misses_cells.parquet` (input to `docs/figures/make_figures_1056.py`).
- `check_grids.py` — reads CRS, pixel size and origin of Sentinel-2, Landsat, Sentinel-1 RTC and Copernicus DEM products over Snowbird (spec section 4).
