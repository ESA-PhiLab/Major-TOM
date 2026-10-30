> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review, 2026-10-30. First version, used for read tests; not yet a Space.

# Major TOM Climate Globe

A web page that shows the Major TOM Climate datasets on a 3D globe, read straight from source.coop in the browser.
No server: plain HTML and JavaScript, libraries from jsDelivr at pinned versions.

## Flow

```
panel: layer, source, month
  -> data.js:  preview image | monthly Parquet read live (hyparquet) | test layouts (Zarr, Parquet overviews)
  -> tiles.js: map tiles drawn in the browser, each pixel coloured by the Major TOM cell under it
  -> app.js:   MapLibre globe; neighbouring months loaded ahead; cursor shows the cell and its value
click a cell -> its monthly normals (one climatology file) and, on request, its whole series
```

## Files

| File | What it does |
|---|---|
| `index.html` | page layout and styles (Major TOM colours, IBM Plex Mono, Source Sans 3) |
| `app.js` | globe, panel, loading, cursor, selected cell |
| `data.js` | the layers and every read from source.coop, with bytes and time counted |
| `tiles.js` | cell under a point (as `majortom.cells`), tiles from a field |
| `scales.js` | colour scales of the README figures, as lookup tables |
| `charts.js` | small SVG line charts for the selected cell |
| `build_images.py` | climatology -> longitude/latitude preview images (CPC precipitation so far) |

## Run locally

```
cd apps/climate-globe && python3 -m http.server 8790        # then open http://localhost:8790
```

Preview images are not in git: build them with `build_images.py` (Slurm, `miko-torch` environment). The test
sources read `major-tom/climate/_experiments/`, which will be removed before the release goes public.

## Measurements

`docs/reports/2026-10-30-climate-globe-read-experiments.md`.
