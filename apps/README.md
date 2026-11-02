> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review, 2026-11-02.

# Apps

Web apps for browsing Major TOM data. Each folder here is the code of one Hugging Face Space in the
[Major-TOM](https://huggingface.co/Major-TOM) organisation; GitHub is where they are edited. The apps are not part of
the installable `majortom` package.

| Folder | Space | What it shows | How it runs |
|---|---|---|---|
| `viewer/` | [MajorTOM-Core-Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer) | one Core sample at the map centre or found by a text prompt (SigLIP); Sentinel-2, Sentinel-1, DEM, AlphaEarth, VIIRS | Python server (Solara, Docker) |
| `climate-globe/` | [MajorTOM-Climate-Globe](https://huggingface.co/spaces/Major-TOM/MajorTOM-Climate-Globe) | Major TOM Climate (CPC, IMERG, CAMS, ERA5) on a 3D globe: monthly maps, normals, a cell's whole series | static page, reads source.coop |
| `raster-viewer/` | [MajorTOM-Raster-Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Raster-Viewer) (private) | where each Core dataset has samples, on a 3D globe; the thumbnail of the sample in a clicked cell | static page, reads Hugging Face |

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

`.github/workflows/apps-spaces.yml`: a push that changes a folder uploads that folder to its Space (files added or
replaced, none deleted; tests and caches left out). Files that are not in git stay in the Space once uploaded:
`viewer/helpers/*.parquet` (search index) and `raster-viewer/footprints/` (built by `build_index.py`).

## Tests

```
cd apps/climate-globe && pytest -m network        # headless Chromium (Playwright), reads source.coop
cd apps/raster-viewer && pytest -m network        # reads Hugging Face; needs footprints/ built
```

`.github/workflows/apps-tests.yml` runs the Climate Globe's on GitHub. The Raster Viewer's need `footprints/`, which
is not in git, so they run where it is built.
