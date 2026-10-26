> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Major TOM — context

Major TOM (Terrestrial Observation Metaset) is a standard for large EO datasets: a global grid of ~10 km cells plus a shared metadata schema, so datasets from different sensors can be joined by `grid_cell`. Paper: arXiv 2402.12095. Datasets: https://huggingface.co/Major-TOM (Core-S2L2A, Core-S2L1C, Core-S1RTC, Core-DEM, embedding sets) and https://source.coop/major-tom.

## Repo map

- `majortom/` — the package (`import majortom`; `import MajorTOM` and the old module names still work through `MajorTOM.py` and small alias modules).
  - `grid.py`, `cells.py` — grid construction; cells at any spacing from the grid formulas.
  - `spec.py` — sample windows: cell → CRS → window on a pixel lattice (`docs/SPEC.md`).
  - `metadata.py`, `sample.py`, `dataset.py` — metadata reads and filters, sample decoding, torch `Dataset`.
  - `embedder/` — tile fragmenting and embedding models (SSL4EO, DINOv2, SigLIP).
  - `build/` — `majortom.build`: samples for any cell from public archives (`docs/ARCHIVES.md`). Products and band names per archive in `collections.py`; archives in `archives.py`; access methods in `backends/` (`stac`, `earthengine`); credentials in `credentials.py`.
- `tests/` — offline tests (CI); `-m network` tests read live archives.
- `docs/SPEC.md` — sample specification. `docs/STRATEGY.md` — refactor plan. `docs/ARCHIVES.md` — data sources and setup. `CONTRIBUTING.md` — adding products and archives.
- `docs/reports/` — dated reports. `docs/evidence/` — scripts and tables behind report findings (benchmark, gallery, cross-archive comparisons).
- `docs/workshop/workshop.ipynb` — workshop notebook (tag `cng-workshop-2026` holds the version taught).
- `0*-*.ipynb` — the original example notebooks.

## Current state (2026-10-22)

- Refactor in progress on `refactor/v1`. Plan: `docs/STRATEGY.md`; progress: `CHECKLIST.md`.
- `majortom.build`: five archives (Planetary Computer, Earth Search, CDSE, NASA Earthdata, Earth Engine), 33 products, all building at the test cell 624U_29L.
- Known defect: every Core sample has a fractional geotransform (sub-pixel error). Pixel match done ([report](docs/reports/2026-10-22-core-pixel-match.md)); `majortom.spec.correct_core_transform` gives the corrected grid. Readers do not apply it yet.
- `main` has a stash (`stash@{0}`) with the embedding reproducibility check and the embedder import re-enabled.

## Development environment

- Package development uses **uv** with Python 3.11 (3.10 until 2026-10-25). **Long Slurm builds run on `.venv311`** (`UV_PROJECT_ENVIRONMENT=.venv311 uv sync --extra build --extra gee --extra viz --extra climate`), by its full path, never on `.venv`: any `uv run` or `uv sync` in the repo recreates `.venv` from scratch without extras, and on 2026-10-25 12:53 UTC that killed every running climate build: `.venv/` in the repo root (not committed), built from `pyproject.toml`; `uv.lock` is committed once it exists. Activate with `source .venv/bin/activate`, or prefix commands with `uv run`. In Slurm jobs, activate inside the job.
- uv lives in `~/.local/bin/uv` (add `~/.local/bin` to `PATH`).
- Optional extras: `build` (archives), `gee` (Earth Engine), `viz` (gallery), `torch`, `embed`: `uv sync --extra build --extra gee --extra viz`.
- The tutorial scripts in `docs/animations/` still run in the conda env `miko-torch`.
