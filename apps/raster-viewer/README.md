---
title: Major TOM Raster Viewer
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Major TOM Core samples on a 3D globe, read in the browser
---

> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review, 2026-11-02. Space created (private) and deployed 2026-11-02.

# Major TOM Raster Viewer

A web page that shows where the Major TOM Core datasets have samples, on a 3D globe, and loads the image of the
sample in a clicked cell. It runs in the browser and reads the Core Parquet files on Hugging Face directly: no
server. Plain HTML and JavaScript, libraries from jsDelivr at pinned versions.

The [Core Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer) (a Python server) stays as it is,
with its text search and the AlphaEarth and VIIRS layers.

## Flow

```
build_index.py (once): each Core dataset's metadata.parquet -> footprints/<name>.parquet, sorted by cell
panel: dataset
  -> footprints.js: the two position columns of its footprint file -> cells with a sample -> coverage tiles
  -> app.js:        MapLibre globe; cursor shows the cell and whether it has a sample
click a cell -> its footprint rows (page index) -> the image file on Hugging Face (one row group per sample)
  -> samples.js: that row group's thumbnail column -> panel, with acquisition time, cloud cover or orbit, product
in the background: the other datasets' footprints once the first is drawn; after a click, the same cell's samples
  and first thumbnail in the other datasets (kept, so switching dataset shows them at once)
```

## Files

| File | What it does |
|---|---|
| `index.html` | page layout and styles (as the Climate Globe) |
| `app.js` | globe, dataset list, loading flag, cursor, selected cell |
| `footprints.js` | a dataset's footprints: coverage mask, a cell's samples |
| `samples.js` | one sample's thumbnail from its image file |
| `parquet.js` | range requests for hyparquet, with bytes and time counted |
| `tiles.js`, `labels.js`, `places.json` | copied from the Climate Globe: cells and tiles, place names |
| `build_index.py` | Core `metadata.parquet` -> `footprints/<name>.parquet` and `footprints/datasets.json` |
| `footprints/` | built files, not in git: uploaded to the Space once (S2L2A and S2L1C 20.5 MB each, S1RTC 12.1 MB, DEM 4.5 MB) |

## Reads

| Read | Bytes | Time |
|---|---|---|
| S2L2A footprints (coverage of 2,245,886 samples) | 3.9 MB in 37 requests | 0.7 s |
| One S2L2A thumbnail, first from its image file (footer 1.5 MB) | 1.1 MB in 4 requests | 1.8 s |

Reading each dataset's `metadata.parquet` from Hugging Face instead was measured first: S2L2A's has one row group
per image file (4,492), so its positions took 462 s in 4,492 requests; S1RTC's and DEM's have two row groups, so
one sample's row cost 23-27 MB. Hence the footprint files.

## Run locally

```
cd apps/raster-viewer && python build_index.py /data/Major-TOM footprints     # once
pytest -m network                                                           # serves the folder with byte ranges
```

The page needs a server that answers HTTP range requests (the tests' server does; `python -m http.server` does not).
