> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Checklist

Workstreams are defined in `docs/STRATEGY.md`.

## WS0 — Scaffolding
- [x] Branch `refactor/v1`; stash `main` changes
- [x] `CONTEXT.md`, `CHECKLIST.md`, `LOG.md`, `docs/STRATEGY.md`, research report
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
- [x] `docs/SPEC.md` drafted (any spacing, source profiles) and reviewed (Miko, 2026-10-15)
- [ ] `sample_window`, `correct_core_transform`, `crop_1068_to_1056`
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
- [x] Docs: `docs/ARCHIVES.md` (generated products table), `CONTRIBUTING.md`
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
- [ ] Xet dedup test on one Core parquet
- [ ] Core v1.1 on HF
- [ ] Major TOM v2 on source.coop
