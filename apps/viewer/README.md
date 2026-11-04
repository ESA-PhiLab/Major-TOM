---
title: Major TOM Viewer
emoji: 🌍
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
short_description: Quick View of Samples in the MajorTOM-Core Dataset
---

> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** not yet reviewed, 2026-11-04. README written from the code in `apps/viewer/` (previously the
> Space front matter only).

# Major TOM Core Viewer

A Solara app that shows one Major TOM Core sample: the sample at the map centre, or the sample found by a text
prompt. It runs as a Python server in the Docker Space
[Major-TOM/MajorTOM-Core-Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer).

## Flow

```
start: the metadata.parquet of Core-S2L2A, Core-S2L1C, Core-S1RTC and Core-DEM (Hugging Face, or helpers/*.parquet
  if present) -> one table per source; the SigLIP FAISS index and its metadata; the AlphaEarth index (source.coop)
"Find sample at map centre": map centre -> Major TOM cell (helpers/grid.py, 10 km grid)
  -> the cell's row in the chosen source's metadata -> parquet_url, parquet_row
  -> the thumbnail column of that row group (fsspec, pyarrow) -> 1,068 x 1,068 image
"Search": text -> SigLIP text encoder -> FAISS IVF-PQ index, nearest embedding (k = 1) -> its grid_cell -> as above
AlphaEarth Embeddings: the COG holding the cell centre in the year of the cell's S2L2A sample (tge-labs/aef)
  -> the 10,680 m window -> PCA of the bands to RGB
VIIRS Nighttime Light: INDEX.parquet (Major-TOM/Core-VIIRS-Nighttime-Light) -> the tile's shard, offset and size
  -> GDAL range read of the GeoTIFF in the year of the cell's S2L2A sample -> log scale 0.1-100 nW/cm²/sr, inferno
panel: the image; grid cell, latitude and longitude, acquisition time, source
```

Data source menu: Sentinel-2 L2A, Sentinel-2 L1C, Sentinel-1 RTC, COP-DEM, AlphaEarth Embeddings, VIIRS Nighttime
Light (`app.py:49`). For AlphaEarth and VIIRS the panel shows the year instead of an acquisition time.

## Files

| File | What it does |
|---|---|
| `app.py` | Solara page: header, instructions, source menu, map (leafmap), find and search buttons, image and table |
| `helpers/functional.py` | loads the four metadata tables and the search index; cell -> metadata row -> thumbnail |
| `helpers/grid.py` | the Major TOM grid: latitude and longitude to row and column, and back |
| `helpers/TextSearch.py` | SigLIP text encoder (`timm/ViT-SO400M-14-SigLIP-384`, open_clip) and FAISS search; GPU if present, else CPU, `nprobe` 32 on both |
| `helpers/ReadAlphaEarth.py` | AlphaEarth index from source.coop, the cell's window of one COG, PCA to RGB |
| `helpers/ReadVIIRSNTL.py` | VIIRS tile index from Hugging Face, one tile read by byte range, colour scale |
| `Dockerfile` | conda env (Python 3.9, faiss-gpu 1.7.4, MKL 2021, libgomp), pip requirements, the two SigLIP index files, `solara run` on port 7860 |
| `requirements.txt` | pip packages (Solara, leafmap, torch 2.1.2, open_clip_torch 2.23.0, transformers 4.46.3, duckdb 1.0.0, ...) |
| `helpers/*.parquet` | optional local copies of the four metadata files, not in git (`.gitignore`); uploaded to the Space once |

The Dockerfile downloads `siglip_ivfpq.index` and `siglip_ivfpq_metadata.parquet` from
`Major-TOM/Core-S2RGB-SigLIP`, folder `extras/faiss/viewer-app/`, at build time.

## Run locally

With Docker, from this folder:

```
docker build -t majortom-core-viewer . && docker run -p 7860:7860 majortom-core-viewer    # http://localhost:7860
```

Without Docker: a conda env as in the `Dockerfile` (faiss from conda, then `pip install -r requirements.txt`), the two
SigLIP files downloaded into this folder (URLs in the `Dockerfile`), then `solara run app.py` from this folder. The
paths in `helpers/functional.py` are relative to it.

Start-up reads the four metadata files and the AlphaEarth index over the network unless local copies exist.

## Tests

None. The app has no test folder.
