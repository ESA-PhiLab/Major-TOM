> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed. Changed 2026-11-05, not yet reviewed: harmonised with the code, LOG and reports
> (items done but not ticked, stale build lines), Miko's decisions of 2026-11-05, final steps added.

# Checklist

Workstreams are defined in [docs/STRATEGY.md](docs/STRATEGY.md).

## Final steps (order of work, 2026-11-05)

Decisions (Miko, 2026-11-05): exception-zone rule as in SPEC 6.1; Core v1.1 later, once it is decided where the
data is processed; full quality runs only with v1.1, otherwise metadata checks and checks in `majortom.build`;
explicit imports now, module renames after the merge; CSLC issue posted.

1. WS1: keep `majortom/extras/` out of the wheel
2. WS6: bug fixes and explicit imports; embedder lazy import and `fragment_fn` checks
3. WS2: grid tests vs paper formulas and Core `grid_cell`; vectorised point→cell; polygon `cells_in`; grid note (#19)
4. WS6: embedding reproducibility test (one short GPU job); notebook moves and output stripping
5. WS4: metadata-only quality checks for Core; per-sample checks recorded by `majortom.build`
6. WS9: ERA5-Land south fill ends; re-upload changed Core-Weather day files; README south note; delete build folders
7. Reviews (Miko): reports, GIFs, Core-Weather README figures, agent skill
8. Merge `refactor/v1`; publish to PyPI; module renames with aliases
9. Later: Core v1.1 (Xet test first), Major TOM v2 and the new Sentinel-2 Core on source.coop, Hugging Face copies
   of the climate datasets, climate data in the public viewing app

## WS0 — Scaffolding
- [x] Branch `refactor/v1`; stash `main` changes
- [x] [CONTEXT.md](CONTEXT.md), [CHECKLIST.md](CHECKLIST.md), [LOG.md](LOG.md), [docs/STRATEGY.md](docs/STRATEGY.md), research report
- [ ] Review and mark docs reviewed (Miko): SPEC and STRATEGY done; reports, GIFs pending

## WS1 — Packaging and PyPI
- [x] `pyproject.toml` with extras; uv env (`.venv`, Python 3.10), `uv.lock`; `setup.py` removed
- [x] `majortom/` package with `MajorTOM` alias (old module names kept)
- [x] Lazy imports with extra-naming `ImportError` (`tests/test_imports.py`)
- [ ] Keep `majortom/extras/` notebooks and images out of the wheel
- [x] Agent skill for using Major TOM: `skills/major-tom/` (SKILL.md and reference/; snippets run 2026-11-05; review pending)
- [x] CI (tests on 3.10–3.13, wheel build): `.github/workflows/tests.yml`
- [ ] Publish to PyPI (#20)

## WS2 — Grid
- [ ] Document construction vs paper (#19)
- [x] `cells_in(bbox)`, `cells_from_points`, `cell_at`, `cell_from_name` (`majortom/cells.py`)
- [ ] `cells_in` for a polygon; vectorised point→cell (`cells_from_points` loops over points) (#20)
- [ ] Tests vs paper formulas and Core `grid_cell` values

## WS3 — Spec and Core correction
- [x] Pixel match on 12 samples per Core dataset: native samples are exact copies at the stored origin rounded to the band grid ([report](docs/reports/2026-10-22-core-pixel-match.md))
- [x] [docs/SPEC.md](docs/SPEC.md) drafted (any spacing, source profiles) and reviewed (Miko, 2026-10-15)
- [x] `correct_core_transform` (`majortom/spec.py`): rounded origin for Core-S2L2A, S2L1C, S1RTC (all samples, no flag needed); Core-DEM unchanged
- [x] Windows: `window`, `legacy_window` (`majortom/spec.py`); v1→v2 crop as `crop_core_to_v2` (SPEC 7.3)
- [x] Readers apply correction by default: `read_row(georef=True)`, `filter_download`, `read_tif`; `raw=True` keeps the stored transform (SPEC 7.5)
- [x] Exception-zone rule: SPEC 6.1 (Norway to zone 31V; Svalbard keeps `cell_coverage` < 1) (Miko, 2026-11-05)

## WS4 — Quality reports
- [ ] Core: checks from metadata only (duplicate cell/product pairs, shapes, alignment)
- [ ] v2: per-sample checks recorded by `majortom.build` in each sidecar
- [ ] Only with Core v1.1: full per-sample `quality.parquet`, `filter_metadata(quality=...)`, Slurm runs over Core

## WS5 — majortom.build
- [x] First version for the workshop: `majortom.build` on Planetary Computer, 13 collections (WS5 decision gate still open)
- [x] Scene choice (`prefer`, `cloud` range, `strict`), yearly/static products, pass grouping
- [x] `fallback` for scene choice: `"complete"` (default) or `"clear"`
- [x] Record per sample: offset, declared scaling, processing baseline, platform, archive, timing, cloud source
- [x] Backends behind one interface (`search`, `header`, `read_tile`, `session`): `stac` and `earthengine`
- [x] Archives: Planetary Computer, Earth Search, CDSE, NASA Earthdata (LP DAAC, PO.DAAC, ASF), Earth Engine
- [x] Credentials: env or owner-only files, setup command, secret scan in CI (`tests/test_no_secrets.py`)
- [x] 33 products from 55 product-archive pairs build at cell 624U_29L (`tests/test_products.py`); PC HLS L30 partial there (archive gap)
- [x] Same data identical across archives where processing matches: S2 L2A/L1C, Landsat 5/9, HLS L30, Copernicus DEM
- [x] Read paths: native/warped COG and JPEG 2000, ground control points (raw S1 GRD), netCDF swaths (Sentinel-3)
- [x] Product gallery (`docs/evidence/gallery.py`) and reading-speed report (`docs/evidence/build_speed.py`)
- [x] Docs: [docs/ARCHIVES.md](docs/ARCHIVES.md) (generated products table), [CONTRIBUTING.md](CONTRIBUTING.md)
- [x] Sentinel-3 swath reads: 83 s -> 24 s per sample (download cache; the first coordinate read per scene stays ~20 s)
- [x] MODIS: Terra and Aqua told apart (`platform`, filter `platform="terra"`); no longer stitched together
- [x] Producer page per product, linked in the products table
- [x] Global geocoded complex Sentinel-1 (CSLC): GitHub issue posted (Miko, 2026-11-05); work later
- [x] COG writer: `majortom/release/cog.py`; tables (parquet) in `majortom/release/tables`
- [x] v1→v2 crop path: `crop_core_to_v2`, and `download(window="v1")` for samples that line up with Core

## WS6 — Cleanup and embedder
- [ ] Explicit imports (no `import *` in `majortom/__init__.py`, `metadata.py`, `embedder/`)
- [ ] After the merge: module renames to snake_case with aliases (Miko, 2026-11-05)
- [ ] Bug fixes listed in STRATEGY WS6 (still open 2026-11-05: `tqdm.notebook`, `plot` ignores `scaling`, README `src/grid.py` link; check `read_row` prefetch, `filter_download` columns)
- [ ] Embedder lazy import, `fragment_fn` checks
- [ ] Embedding reproducibility test
- [ ] Notebook moves and output stripping

## WS7 — Workshop notebook
- [x] Tutorial GIFs: grid, anchoring, multiples of 12, trade-off (`docs/animations/`, review pending)
- [x] Workshop notebook using `majortom.build`: `docs/workshop/workshop.ipynb`
- [ ] Draft notebook with figures and backend comparison (overlaps `docs/workshop/workshop.ipynb`: drop or reword, Miko)

## WS8 — Releases
- [x] `majortom.release` (extra `[release]`): COG, rumi, TACO via taco-eo, parquet tables, upload to source.coop and Hugging Face; `major-tom/release-test` read back identical (2026-10-27)
- [ ] Later (where to process undecided, Miko 2026-11-05): Xet dedup test on one Core parquet
- [ ] Later: Core v1.1 on HF
- [ ] Major TOM v2 on source.coop
- [ ] Merge `refactor/v1`, then a new Sentinel-2 Core version on source.coop

## WS9 — Climate (`majortom.climate`, docs/CLIMATE.md)
- [x] Design reviewed; readers for ERA5, ERA5-Land, CAMS, NOAA CPC, IMERG, Sentinel-5P; hourly and daily paths; monthly series and climatology; merge
- [x] Core-Weather ERA5 (12 variables): 3,704,013 acquisitions, complete
- [x] IMERG-, CAMS-, ERA5- (12 variables), CPC-Monthly built; IMERG climatology
- [x] ERA5 variable set decided: principle A, 207 + 16 static (docs/ERA5_VARIABLES.md, majortom/climate/era5_catalogue.py)
- [x] ERA5-Monthly v2 (202 variables, 1940-2025, 2.5 TB); ERA5-Static (16 fields)
- [x] ERA5Land-Monthly (11 variables, 1950-2025); climatologies for ERA5-Land, CPC, IMERG (1991-2020), CAMS (2003-2025)
- [x] Core-Weather Sentinel-5P: all 2,976 acquisition days
- [x] Core-Weather ERA5 207, CAMS, NOAA CPC daily, IMERG: all 2,976 days, validated against all 19 sources (2026-11-02)
- [x] ERA5-Land hourly, all 49 variables, from Earth Data Hub zarr (replaces the Earth Engine plan: measured ~30× the estimate, docs/CLIMATE.md 7): group 1 done, group 2 at 69% (resumes 1 November), groups 3-6 ~900k requests; key via `python -m majortom.build.credentials edh`
- [x] Earth Data Hub keys per person (`credentials edh --name`, `EDH_KEY_NAME`): two extra keys in use from 2026-10-28; groups 2-4 and 5-6 run as two chains, all 49 variables expected by ~30 October
- [x] ERA5Land-Monthly with all 49 variables, climatology 1991-2020 (2026-11-02); ERA5-Monthly-v2 removed, ptype kept (2026-11-02)
- [x] Core-Weather published to source.coop major-tom/core-weather (2026-11-02, without the ERA5-Land south fill)
- [ ] Fill ERA5-Land south of 57.1°S from the Climate Data Store (docs/CLIMATE.md 6.1): 2016 and 2018 done; 2019 and 2020 running in parallel (2026-11-05, one cache folder per year); 2021-2024 queued one after another (fill job 2611 last); then re-upload the changed day files, update the README south note, delete the build folders
- [x] Validate all sources and clear the download cache (2026-11-02; south-fill downloads kept)
- [x] Publish Core-Weather and the monthly datasets to source.coop (2026-11-02)
- [ ] Hugging Face copies of Core-Weather and the monthly datasets (Miko, 2026-10-24: source.coop first, Hugging Face second)
- [x] Climate Globe (apps/climate-globe, private Space)
- [ ] Climate datasets in the public Major TOM viewing app (final task)
- [x] Intro animation for Major TOM Climate and Core-Weather: the logo builds first, then examples of the data animated (time series drawn month by month, global maps filling in); requested 2026-11-03, v2 done 2026-11-04 (docs/animations/intro_climate_weather.py, square and wide, 66 s)

## WS10 — Apps (`apps/`, apps/README.md)

- [x] Asterisk Labs logo (map corner, links to asterisk.coop) and Earth Compress credit in the Raster Viewer and Climate Globe (2026-11-03, deployed)
- [x] Core Viewer: layout clean-up: header, controls, fonts, text prompt, placeholder, acquisition time (b1a22d6, 2026-11-03, pushed)
- [x] Raster Viewer: Core-Weather panel for the shown sample (hourly ±24 h, daily ±30 d; ERA5, CAMS, Sentinel-5P); image always on the map, opaque (2026-11-03, d0733a5, pushed to origin/refactor/v1)
- [x] Core-Weather README figures and README uploaded (2026-11-03); Earth Compress footer added (2026-11-05)
- [ ] Review the Core-Weather README figures (Miko)
- [x] Raster Viewer: VIIRS Nighttime Light and AlphaEarth embeddings, with downloads (2026-11-03, deployed)
- [ ] Raster Viewer: climate panel for the selected cell
- [ ] Future: text search in the browser (Raster Viewer): SigLIP text encoder in transformers.js; embeddings in an inverted-file index (cluster centres loaded once, Parquet sorted by cluster, nearest clusters read by range request, optional product quantisation); check the exact SigLIP model of Core-S2RGB-SigLIP, its ONNX export and download size
