> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed. Changed 2026-11-04, not yet reviewed: Core Viewer layout ticked, weather panel marked
> pushed, ERA5-Land south fill timing.

# Checklist

Workstreams are defined in [docs/STRATEGY.md](docs/STRATEGY.md).

## WS0 — Scaffolding
- [x] Branch `refactor/v1`; stash `main` changes
- [x] [CONTEXT.md](CONTEXT.md), [CHECKLIST.md](CHECKLIST.md), [LOG.md](LOG.md), [docs/STRATEGY.md](docs/STRATEGY.md), research report
- [ ] Review and mark docs reviewed (Miko): SPEC and STRATEGY done; reports, GIFs pending

## WS1 — Packaging and PyPI
- [x] `pyproject.toml` with extras; uv env (`.venv`, Python 3.10), `uv.lock`; `setup.py` removed
- [x] `majortom/` package with `MajorTOM` alias (old module names kept)
- [x] Lazy imports with extra-naming `ImportError` (`tests/test_imports.py`)
- [ ] Keep `majortom/extras/` notebooks and images out of the wheel
- [x] CI (tests on 3.10–3.13, wheel build): `.github/workflows/tests.yml`
- [ ] Publish to PyPI (#20)

## WS2 — Grid
- [ ] Document construction vs paper (#19)
- [ ] `cells_in(polygon)`, vectorised point→cell (#20)
- [ ] Tests vs paper formulas and Core `grid_cell` values

## WS3 — Spec and Core correction
- [x] Pixel match on 12 samples per Core dataset: native samples are exact copies at the stored origin rounded to the band grid ([report](docs/reports/2026-10-22-core-pixel-match.md))
- [x] [docs/SPEC.md](docs/SPEC.md) drafted (any spacing, source profiles) and reviewed (Miko, 2026-10-15)
- [x] `correct_core_transform` (`majortom/spec.py`): rounded origin for Core-S2L2A, S2L1C, S1RTC (all samples, no flag needed); Core-DEM unchanged
- [ ] `sample_window`, `crop_1068_to_1056`
- [ ] Readers apply correction by default
- [ ] Decide exception-zone rule

## WS4 — Quality reports
- [ ] Per-sample checks and `quality.parquet`
- [ ] `filter_metadata(quality=...)`
- [ ] Slurm runs over Core datasets

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
- [ ] Global geocoded complex Sentinel-1 (CSLC): GitHub issue, for later
- [ ] COG and parquet writers
- [ ] v1→v2 crop path

## WS6 — Cleanup and embedder
- [ ] Module renames with aliases, explicit imports
- [ ] Bug fixes listed in STRATEGY WS6
- [ ] Embedder lazy import, `fragment_fn` checks
- [ ] Embedding reproducibility test
- [ ] Notebook moves and output stripping

## WS7 — Workshop notebook
- [x] Tutorial GIFs: grid, anchoring, multiples of 12, trade-off (`docs/animations/`, review pending)
- [x] Workshop notebook using `majortom.build`: `docs/workshop/workshop.ipynb`
- [ ] Draft notebook with figures and backend comparison

## WS8 — Releases
- [ ] `majortom.release` (extra `[release]`, proposed): writers for COG, zarr and rumi; TACO containers via the taco package (its own repo); upload to Hugging Face and source.coop
- [ ] Xet dedup test on one Core parquet
- [ ] Core v1.1 on HF
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
- [ ] ERA5 207 in Core-Weather: groups 1-2 done, 3 at 60% (restarted 2026-10-28 after a hang), 4 running, 5-8 queued
- [ ] Core-Weather CAMS: 1,956 / 2,976 days (running)
- [ ] Core-Weather NOAA CPC daily (job 1901, started 2026-10-28; was never built before)
- [ ] ERA5-Land hourly, all 49 variables, from Earth Data Hub zarr (replaces the Earth Engine plan: measured ~30× the estimate, docs/CLIMATE.md 7): group 1 done, group 2 at 69% (resumes 1 November), groups 3-6 ~900k requests; key via `python -m majortom.build.credentials edh`
- [x] Earth Data Hub keys per person (`credentials edh --name`, `EDH_KEY_NAME`): two extra keys in use from 2026-10-28; groups 2-4 and 5-6 run as two chains, all 49 variables expected by ~30 October
- [ ] 1 November: resume IMERG in Core-Weather on Earth Engine (256 / 2,976 days; ~130 EECU-hours), after a one-hour calibration against the quota counter
- [ ] ERA5-Monthly v2 climatology (1991-2020)
- [ ] Optional: ERA5Land-Monthly with all 49 variables from Earth Data Hub's monthly store (check its chunking on 1 November)
- [x] Core-Weather published to source.coop major-tom/core-weather (2026-11-02, without the ERA5-Land south fill)
- [ ] Fill ERA5-Land south of 57.1°S from the Climate Data Store (docs/CLIMATE.md 6.1): 2016 done; 2018 running (jobs 2397-2402 since 2026-11-03, fill 2261); 2019-2024 queued in sequence (to job 2303): 2.5-4.5 days per year for 2019-2023, under 1 day for 2024; chain ends about 22 Nov 2026; then re-upload about 3,200 changed day files
- [ ] Validate ERA5 groups 3-8 and ERA5-Land groups 2-6; clear the 815 GB download cache after publishing
- [ ] Merge sources and publish Core-Weather and the monthly datasets: source.coop first, Hugging Face second (Miko, 2026-10-24)
- [ ] Climate datasets in the Major TOM viewing app (HF Space; final task)
- [x] Intro animation for Major TOM Climate and Core-Weather: the logo builds first, then examples of the data animated (time series drawn month by month, global maps filling in); requested 2026-11-03, v2 done 2026-11-04 (docs/animations/intro_climate_weather.py, square and wide, 66 s)

## WS10 — Apps (`apps/`, apps/README.md)

- [x] Asterisk Labs logo (map corner, links to asterisk.coop) and Earth Compress credit in the Raster Viewer and Climate Globe (2026-11-03, deployed)
- [x] Core Viewer: layout clean-up: header, controls, fonts, text prompt, placeholder, acquisition time (b1a22d6, 2026-11-03, pushed)
- [x] Raster Viewer: Core-Weather panel for the shown sample (hourly ±24 h, daily ±30 d; ERA5, CAMS, Sentinel-5P); image always on the map, opaque (2026-11-03, d0733a5, pushed to origin/refactor/v1)
- [ ] Core-Weather README figures: review, then upload figures/ and the README (2026-11-03)
- [x] Raster Viewer: VIIRS Nighttime Light and AlphaEarth embeddings, with downloads (2026-11-03, deployed)
- [ ] Raster Viewer: climate panel for the selected cell
- [ ] Future: text search in the browser (Raster Viewer): SigLIP text encoder in transformers.js; embeddings in an inverted-file index (cluster centres loaded once, Parquet sorted by cluster, nearest clusters read by range request, optional product quantisation); check the exact SigLIP model of Core-S2RGB-SigLIP, its ONNX export and download size
