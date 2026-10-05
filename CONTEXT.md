> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Major TOM — context

Major TOM (Terrestrial Observation Metaset) is a standard for large EO datasets: a global grid of ~10 km cells plus a shared metadata schema, so datasets from different sensors can be joined by `grid_cell`. Paper: arXiv 2402.12095. Datasets: https://huggingface.co/Major-TOM (Core-S2L2A, Core-S2L1C, Core-S1RTC, Core-DEM, embedding sets) and https://source.coop/major-tom.

## Repo map

- `MajorTOM/grid.py` — grid construction, lat/lon ↔ row/col, cell footprints, UTM EPSG.
- `MajorTOM/metadata_helpers.py` — metadata loading, filtering, remote row reads, download.
- `MajorTOM/sample_helpers.py` — GeoTIFF/PNG byte decoding, plotting.
- `MajorTOM/MajorTOMDataset.py` — torch `Dataset` over downloaded samples.
- `MajorTOM/embedder/` — tile fragmenting and embedding models (SSL4EO, DINOv2, SigLIP).
- `MajorTOM/extras/` — thumbnails, coverage plots, minor notebooks.
- `0*-*.ipynb` — main example notebooks.
- `docs/STRATEGY.md` — refactor plan. `docs/reports/` — dated reports. `docs/evidence/` — scripts behind report findings.

## Current state (2026-09-30)

- Refactor in progress on `refactor/v1`. Plan: `docs/STRATEGY.md`.
- Known defect: every Core sample has a fractional geotransform (sub-pixel error). See `docs/reports/2026-09-30-refactor-research.md`.
- `main` has a stash (`stash@{0}`) with the embedding reproducibility check and the embedder import re-enabled.

## Development environment

- Package development uses **uv** with Python 3.10: `.venv/` in the repo root (not committed), built from `pyproject.toml`; `uv.lock` is committed once it exists. Activate with `source .venv/bin/activate`, or prefix commands with `uv run`. In Slurm jobs, activate inside the job.
- uv lives in `~/.local/bin/uv` (add `~/.local/bin` to `PATH`).
- The tutorial scripts in `docs/animations/` still run in the conda env `miko-torch`.
