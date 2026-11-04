> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** not yet reviewed. First version 2026-11-02. Changed 2026-11-04: Raster Viewer and Climate Globe
> descriptions, Spaces and their SDKs, deployment, tests per app.

# Apps

Web apps for browsing Major TOM data. Each folder here is the code of one Hugging Face Space in the
[Major-TOM](https://huggingface.co/Major-TOM) organisation; GitHub is where they are edited. The apps are not part of
the installable `majortom` package.

| Folder | Space | What it shows | How it runs |
|---|---|---|---|
| `viewer/` | [MajorTOM-Core-Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer) | one Core sample at the map centre or found by a text prompt (SigLIP); Sentinel-2, Sentinel-1, DEM, AlphaEarth, VIIRS | Python server (Solara, Docker Space) |
| `climate-globe/` | [MajorTOM-Climate-Globe](https://huggingface.co/spaces/Major-TOM/MajorTOM-Climate-Globe) (private) | Major TOM Climate (CPC, IMERG, CAMS, ERA5) on a 3D globe: monthly maps, normals, a cell's whole series from `by_cell/` (ERA5-Monthly included since 15d5a93, 2026-11-04) | static Space, reads source.coop |
| `raster-viewer/` | [MajorTOM-Raster-Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Raster-Viewer) (private) | where each Core dataset has samples, on a 3D globe; the sample in a clicked cell (thumbnail, bands, composites, downloads); VIIRS Nighttime Light tiles; AlphaEarth embeddings as a PCA; Core-Weather at the acquisition (hourly ±24 h, daily ±30 d) | static Space, reads Hugging Face and source.coop |

## Static apps

`climate-globe/` and `raster-viewer/` have no server: plain HTML and JavaScript modules, libraries from jsDelivr at
pinned versions (MapLibre GL for the globe, hyparquet for Parquet). They read Parquet files over HTTP range
requests, so a click costs a few requests for the parts of a file it needs, not a download of the file. Both draw
their map tiles in the browser: each pixel is coloured by the Major TOM cell under it (`tiles.js`, the same in both).

```
panel or click -> Parquet footer (cached per file) -> the row groups or pages holding the cells -> values
  -> field on the (row, column) array of cells -> tiles.js: MapLibre tiles -> globe
```

Each app's README (its Space card) lists its files, flow and measured reads.

## Deployment

`.github/workflows/apps-spaces.yml` runs on pushes to `main` and `refactor/v1` that touch `apps/**`, and by hand
("Run workflow", which uploads every app). Each app whose folder changed is uploaded to its Space (files added or
replaced, none deleted; `tests/`, `pytest.ini`, `__pycache__/` and `.gitignore` left out):

| Folder | Space | SDK |
|---|---|---|
| `apps/viewer` | `Major-TOM/MajorTOM-Core-Viewer` | Docker (Solara) |
| `apps/climate-globe` | `Major-TOM/MajorTOM-Climate-Globe` (private) | static |
| `apps/raster-viewer` | `Major-TOM/MajorTOM-Raster-Viewer` (private) | static |

The token is `HF_TOKEN` in the GitHub environment `hf-space`. Files that are not in git stay in the Space once uploaded:
`viewer/helpers/*.parquet` (local copies of the four Core metadata files) and `raster-viewer/footprints/` (built by `build_index.py`).

## Tests

| App | Test files | Command |
|---|---|---|
| `viewer/` | none | - |
| `climate-globe/` | `tests/test_app.py` (7 tests: headless Chromium, Playwright; reads source.coop) | `cd apps/climate-globe && pytest -m network` |
| `raster-viewer/` | `tests/test_app.py` (13 tests: headless Chromium; reads Hugging Face and source.coop; needs `footprints/` built) | `cd apps/raster-viewer && pytest -m network` |
| `raster-viewer/` | `tests/test_imports.py` (1 test: every local module is imported by `app.js`; no network) | `cd apps/raster-viewer && pytest tests/test_imports.py` |

`climate-globe/` and `raster-viewer/` each have their own `pytest.ini` (test path `tests`, marker `network`).
`.github/workflows/apps-tests.yml` runs the Climate Globe's tests on GitHub, on pushes and pull requests that touch
`apps/climate-globe/**`. The Raster Viewer's browser tests need `footprints/`, which is not in git, so they run where
it is built.
