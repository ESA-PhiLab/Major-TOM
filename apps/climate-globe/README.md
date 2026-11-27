---
title: Major TOM Climate Globe
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Major TOM Climate datasets on a 3D globe, from source.coop
---

> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** not yet reviewed. First version 2026-10-30. Changed 2026-11-04: deployment (Space synced by
> `.github/workflows/apps-spaces.yml`); ERA5 series read from `by_cell/` (commit 15d5a93). 2026-11-27 (not yet reviewed): Spaces public; app text and Code links updated for the release. 2026-11-27 (not yet reviewed): every variable of CPC, IMERG, CAMS, ERA5-Land and ERA5 (a dataset list with a variable dropdown; `scales/`).

# Major TOM Climate Globe

A web page that shows the Major TOM Climate datasets on a 3D globe, read straight from source.coop in the browser.
No server: plain HTML and JavaScript, libraries from jsDelivr at pinned versions.

## Flow

```
panel: dataset (CPC, IMERG, CAMS, ERA5-Land, ERA5), one of its variables, source, month
  -> data.js:  preview image | Parquet read live (hyparquet): overviews/ at globe scale, series/ zoomed in
  -> tiles.js: map tiles drawn in the browser, each pixel coloured by the Major TOM cell under it
  -> app.js:   MapLibre globe; neighbouring months loaded ahead; cursor shows the cell and its value;
              place names from zoom 5 (labels.js)
click a cell -> its monthly normals (one climatology file) and, on request, its whole series (by_cell/; every dataset:
  one variable, e.g. 1,032 months for ERA5, 1940-2025)
```

## Files

| File | What it does |
|---|---|
| `index.html` | page layout and styles (Major TOM colours, IBM Plex Mono, Source Sans 3) |
| `app.js` | globe, panel, loading, cursor, selected cell |
| `data.js` | the datasets, their curated variables (hand-set name, unit, scale) and every read from source.coop, with bytes and time counted |
| `tiles.js` | cell under a point (as `majortom.cells`), tiles from a field |
| `scales.js` | colour scales of the README figures, as lookup tables |
| `charts.js` | small SVG line charts for the selected cell |
| `labels.js` | place names from zoom 5: largest first, overlaps skipped |
| `places.json` | cities and landmarks (Natural Earth 10m, public domain), made by `build_places.py` |
| `scales/<dataset>.json` | each dataset's variables: name, unit, colour scale, and whether it has series overviews; made by `build_scales.py` |
| `build_images.py` | climatology -> preview images on the cell grid, one pixel per cell, for four curated variables |
| `build_scales.py` | a dataset's `climatology_overviews/` and README variables table -> `scales/<dataset>.json` (`*_valid` left out) |
| `build_places.py` | Natural Earth GeoJSON -> `places.json` |

## Run locally

```
cd apps/climate-globe && python3 -m http.server 8790        # then open http://localhost:8790
```

Preview images are not in git: build them with `build_images.py` (Slurm, `miko-torch` environment) and upload
them to the Space by hand (the GitHub sync uploads code only). Place names: `build_places.py` from Natural Earth.

## Tests

```
cd apps/climate-globe && pytest -m network        # headless Chromium (Playwright), reads source.coop; python-last env
                                                   # (miko-torch's Playwright 1.48 cannot start MapLibre 6's worker)
```

On GitHub they run when the app changes (`.github/workflows/apps-tests.yml`); the preview test is skipped there, as
the images are not in git.

## Measurements

[docs/reports/2026-10-30-climate-globe-read-experiments.md](../../docs/reports/2026-10-30-climate-globe-read-experiments.md).

## Credit

Developed by [Mikolaj Czerkawski](https://github.com/mikonvergence) and Sponsored by Earth Compress. A project funded by the Advanced Research + Invention Agency (ARIA) and co-led by Asterisk Labs and National Oceanography Centre UK. Learn more here: [www.earthcompress.org](https://www.earthcompress.org)
