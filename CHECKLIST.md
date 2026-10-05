> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Checklist

Workstreams are defined in `docs/STRATEGY.md`.

## WS0 — Scaffolding
- [x] Branch `refactor/v1`; stash `main` changes
- [x] `CONTEXT.md`, `CHECKLIST.md`, `LOG.md`, `docs/STRATEGY.md`, research report
- [ ] Review and mark docs reviewed (Miko)

## WS1 — Packaging and PyPI
- [x] `pyproject.toml` with extras; uv env (`.venv`, Python 3.10), `uv.lock`; `setup.py` removed
- [x] `majortom/` package with `MajorTOM` alias (old module names kept)
- [x] Lazy imports with extra-naming `ImportError` (`tests/test_imports.py`)
- [ ] Keep `majortom/extras/` notebooks and images out of the wheel
- [ ] CI (tests, wheel build)
- [ ] Publish to PyPI (#20)

## WS2 — Grid
- [ ] Document construction vs paper (#19)
- [ ] `cells_in(polygon)`, vectorised point→cell (#20)
- [ ] Tests vs paper formulas and Core `grid_cell` values

## WS3 — Spec and Core correction
- [ ] Repeat pixel-match on ~10 products
- [ ] `docs/SPEC.md`
- [ ] `sample_window`, `correct_core_transform`, `crop_1068_to_1056`
- [ ] Readers apply correction by default
- [ ] Decide exception-zone rule

## WS4 — Quality reports
- [ ] Per-sample checks and `quality.parquet`
- [ ] `filter_metadata(quality=...)`
- [ ] Slurm runs over Core datasets

## WS5 — majortom.build
- [ ] Catalog / Selector / Reader / Writer interfaces
- [ ] Earth Search, Planetary Computer, CDSE catalogs
- [ ] Offset normalisation
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
- [ ] Draft notebook with figures and backend comparison

## WS8 — Releases
- [ ] Xet dedup test on one Core parquet
- [ ] Core v1.1 on HF
- [ ] Major TOM v2 on source.coop
