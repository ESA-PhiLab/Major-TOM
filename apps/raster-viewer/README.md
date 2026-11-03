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
the shown image: Enlarge (full window, zoom and pan) and always on the map over its footprint, opaque: every image
  is 10,680 m square in its UTM zone, centred on the sample's centre_lat, centre_lon, so corners need no band read
View menu: the stored thumbnail (default), composites (S2: true colour, false colour, SWIR, one stretch for the three
  bands; S1: VV, VH, VV/VH in dB; DEM: elevation in colour) and every band (grey, 2-98% stretch, value under the
  cursor in the full window) -> bands.js: the band columns (LZW GeoTIFFs) of the sample's row group, geotiff.js
VIIRS Nighttime Light (viirs.js): no thumbnail column; footprints/VIIRS.parquet gives each tile's zip shard, offset and
  size -> one range request -> the GeoTIFF (zstd: decoder added to geotiff.js, fzstd) -> radiance in inferno, log
  0.1-100 nW/cm²/sr (the Core Viewer's scale) -> panel and map, corners from the GeoTIFF's bounds; no downloads
AlphaEarth Embeddings (aef.js): footprints/AEF.parquet maps each Core-S2L2A cell to the 81.9 km COG tile holding its
  centre, footprints/AEF-tiles.parquet each tile to its file per year (tge-labs/aef on source.coop; one button per year,
  the cell's Sentinel-2 year first) -> PCA of the 64 bands in two steps: the whole 160 m level in one request (~10 MB)
  -> preview (~4.5 s); the 8 bands that best predict the components, at 40 m -> the sharp image (~10 s). The COGs are
  stored south-up: origin and row direction from each file. View menu: single bands at 10 m. No downloads yet
Downloads (download.js): the shown band as stored (.tif); every band, the thumbnail and metadata.json (.zip); a
  Python snippet that reads the same row group from Hugging Face
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
| `bands.js` | a sample's band columns: read, decode (geotiff.js), stretch, composites |
| `download.js` | band GeoTIFFs as stored, all bands as a zip (fflate), the Python snippet |
| `parquet.js` | range requests for hyparquet, with bytes and time counted; each file's footer read once |
| `tiles.js`, `labels.js`, `places.json` | copied from the Climate Globe: cells and tiles, place names |
| `build_index.py` | Core `metadata.parquet` -> `footprints/<name>.parquet` and `footprints/datasets.json` |
| `footprints/` | built files, not in git: uploaded to the Space once (S2L2A and S2L1C 32.9 MB each, S1RTC 20.6 MB, DEM 14.5 MB, VIIRS 21.5 MB, AEF 14.3 MB, AEF-tiles 1.8 MB) |

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
