---
name: major-tom
description: Work with Major TOM, the global 10 km grid and its Earth observation datasets (Core-S2L2A, Core-S2L1C, Core-S1RTC, Core-DEM, embeddings, Core-Weather, the monthly climate datasets). Use when the user wants to find grid cells for points or areas, read Core samples or metadata from Hugging Face, read weather or climate values for cells from source.coop, or build new Major TOM samples from public archives with the majortom package.
---

> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (added 2026-11-05).

# Major TOM

Major TOM divides the Earth into cells of about 10 km × 10 km. Every dataset is keyed by the cell name
(`grid_cell`, e.g. `624U_29L`: row 624 north, column 29 west), so datasets join on that column.

    point or area  ->  cells (majortom.cells)  ->  join on grid_cell  ->  Core samples (Hugging Face)
                                                                     ->  Core-Weather, climate (source.coop)
                                                                     ->  new samples (majortom.build)

## Install

The package is not on PyPI yet. Install from GitHub (branch `refactor/v1` until it is merged into `main`):

    pip install "majortom[build] @ git+https://github.com/ESA-PhiLab/Major-TOM@refactor/v1"

Extras: `build` (new samples), `gee` (Earth Engine archive), `climate`, `torch`, `embed`, `viz`, `release`.
Reading Core, Core-Weather and climate needs no extra: `pandas`, `pyarrow` and `fsspec` (with `aiohttp`).

## 1. Cells for points or an area

```python
import pandas as pd
from majortom.cells import cell_at, cell_from_name, cells_in, cells_from_points

cell = cell_at(-4.6, 56.1)                       # Cell(row=624, col=-29): Loch Lomond
cell.name, cell.centroid                         # '624U_29L', (lon, lat) of the midpoint
cells = cells_in((-5.0, 55.8, -4.0, 56.4))       # lon/lat box (west, south, east, north); table of cells
points = cells_from_points(pd.DataFrame({"lon": [-4.6], "lat": [56.1]}))
```

Tables have `grid_cell`, `row`, `col`, `grid_km`, `lon`, `lat`. Other spacings: pass `d` in km. Details and edge
cases (poles, 180°): [reference/grid.md](reference/grid.md).

## 2. Core samples from Hugging Face

Each Core dataset has a `metadata.parquet` (one row per sample) pointing into image parquets (`parquet_url`,
`parquet_row`). Filter the metadata on `grid_row_u` and `grid_col_r` (signed row and column), not on
`grid_cell`: the row and column filters skip row groups, a `grid_cell` filter reads the whole file. Even so,
the metadata read plus one sample took 111 s in a test (Core-S2L2A, 2026-11-05): read the metadata once and
reuse it for many cells.

```python
import fsspec, pandas as pd
from majortom import read_row

url = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"
meta = pd.read_parquet(url, filesystem=fsspec.filesystem("https"),
                       filters=[("grid_row_u", "=", 624), ("grid_col_r", "=", -29)])
row = meta.iloc[0]
bands = read_row(row, columns=["B04", "B03", "B02"])              # dict of numpy arrays
b04, transform, crs = read_row(row, columns=["B04"], georef=True)["B04"]
```

With `georef=True`, Core-S2L2A, Core-S2L1C and Core-S1RTC transforms are corrected to the grid the pixels were
read on (the stored origin is off by up to half a pixel). Keep that default; `raw=True` gives the stored one.
Datasets, columns and the v1 → v2 crop: [reference/datasets.md](reference/datasets.md).

## 3. Weather around a Core acquisition (Core-Weather)

Hourly values for ±24 h and daily values for ±30 days around every Core-S2L2A, S2L1C and S1RTC acquisition,
from ERA5, ERA5-Land, CAMS, CPC, IMERG and Sentinel-5P. Files are split by acquisition day.

```python
https = fsspec.filesystem("https")
url = "https://data.source.coop/major-tom/core-weather"
links = pd.read_parquet(f"{url}/acquisitions.parquet", filesystem=https, filters=[("grid_cell", "=", "624U_29L")])
pick = links[links.dataset == "Core-S2L2A"].iloc[0]
day = f"year={pick.acquired:%Y}/{pick.acquired:%Y-%m-%d}.parquet"
here = [("grid_cell", "=", pick.grid_cell), ("acquired", "=", pick.acquired)]
hourly = pd.read_parquet(f"{url}/hourly/{day}", filesystem=https, filters=here, columns=["step_h", "era5_t2m"])
```

## 4. Monthly climate per cell (major-tom/climate)

CPC-Monthly, IMERG-Monthly, CAMS-Monthly, ERA5-Monthly, ERA5Land-Monthly, ERA5-Static. Each has `series/`
(one file per month), `climatology/` and `by_cell/` (20 bands of grid rows), and `overviews/`.

```python
url = "https://data.source.coop/major-tom/climate"
band = (624 + 1002) * 20 // 2004                                 # grid row -> band file: 16
here = [("grid_row_u", "=", 624), ("grid_col_r", "=", -29)]
series = pd.read_parquet(f"{url}/CPC-Monthly/by_cell/band={band:02d}.parquet", filesystem=https, filters=here)
normals = pd.read_parquet(f"{url}/CPC-Monthly/climatology/band={band:02d}.parquet", filesystem=https, filters=here)
```

## 5. New samples (majortom.build)

One sample per cell, searched and read from a public archive (Planetary Computer by default, no account).

```python
from majortom import build
cells = build.cells_from_points(pd.DataFrame({"lon": [-4.6], "lat": [56.1]}), d=10)
errors = build.download(cells, "sentinel-2-l2a", "2024-06-01/2024-08-31")   # chips/<collection>/<index>.npz + .json
```

Products, archives, credentials and scene choice: [reference/build.md](reference/build.md).

## Rules

- Join datasets on `grid_cell`; filter large files on `grid_row_u` / `grid_col_r`.
- Read remote parquet with column and row filters; never download a whole Core dataset to answer a question
  about a few cells (Core-S2L2A alone is 22.95 TB).
- Core v1 windows are 1068 px at 10 m; v2 windows are 1056 px, centred on the cell. They are not a centre crop
  of each other: use `majortom.spec.crop_core_to_v2`.
- Never print or store credentials; `python -m majortom.build.credentials <archive>` stores them owner-only.
