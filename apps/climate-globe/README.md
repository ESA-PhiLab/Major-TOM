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

> **Review status:** draft for review, 2026-10-30. Space created (private), not yet deployed.

# Major TOM Climate Globe

A web page that shows the Major TOM Climate datasets on a 3D globe, read straight from source.coop in the browser.
No server: plain HTML and JavaScript, libraries from jsDelivr at pinned versions.

## Flow

```
panel: layer, source, month
  -> data.js:  preview image | Parquet read live (hyparquet): overviews/ at globe scale, series/ zoomed in
  -> tiles.js: map tiles drawn in the browser, each pixel coloured by the Major TOM cell under it
  -> app.js:   MapLibre globe; neighbouring months loaded ahead; cursor shows the cell and its value;
              place names from zoom 5 (labels.js)
click a cell -> its monthly normals (one climatology file) and, on request, its whole series (by_cell/)
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
| `labels.js` | place names from zoom 5: largest first, overlaps skipped |
| `places.json` | cities and landmarks (Natural Earth 10m, public domain), made by `build_places.py` |
| `build_images.py` | climatology -> preview images on the cell grid, one pixel per cell, for the four layers |
| `build_places.py` | Natural Earth GeoJSON -> `places.json` |

## Run locally

```
cd apps/climate-globe && python3 -m http.server 8790        # then open http://localhost:8790
```

Preview images are not in git: build them with `build_images.py` (Slurm, `miko-torch` environment) and upload
them to the Space by hand (the GitHub sync uploads code only). Place names: `build_places.py` from Natural Earth.

## Measurements

[docs/reports/2026-10-30-climate-globe-read-experiments.md](../../docs/reports/2026-10-30-climate-globe-read-experiments.md).
