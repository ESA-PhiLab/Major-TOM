> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (added 2026-11-05).

# Major TOM datasets

## Core (Hugging Face, `huggingface.co/datasets/Major-TOM/<name>`)

| Dataset | Content | Samples | Image parquet size |
|---|---|--:|--:|
| Core-S2L2A | Sentinel-2 L2A, 12 bands + cloud mask + thumbnail | 2,245,886 | 22.95 TB |
| Core-S2L1C | Sentinel-2 L1C, 13 bands + cloud mask + thumbnail | 2,245,886 | 23.55 TB |
| Core-S1RTC | Sentinel-1 RTC, `vv`, `vh` + thumbnail | 1,469,955 | 15.69 TB |
| Core-DEM | Copernicus DEM 30 | 1,837,843 | |

Embeddings of Core: Core-S2L1C-SSL4EO, Core-S1RTC-SSL4EO, Core-S2RGB-DinoV2, Core-S2RGB-SigLIP. Every dataset of
the organisation: `https://huggingface.co/api/datasets?author=Major-TOM`.

## Core-VIIRS-Nighttime-Light (Hugging Face)

VIIRS annual median night-time radiance (`median`, nW/cm²/sr), 2016-2024, 2,108,341 tiles of 1,056 px at 10 m (v2
window). One tile per cell, in one year. Tiles are GeoTIFFs inside zip shards (cozip); `INDEX.parquet` (146 MB, one
row per tile) has `id` (`MT10_<grid_cell>`), `year`, `shard`, `offset`, `size`. Read only those columns:

```python
import fsspec, pyarrow.parquet as pq, rasterio
url = "https://huggingface.co/datasets/Major-TOM/Core-VIIRS-Nighttime-Light/resolve/main/"
index = pq.read_table(f"{url}INDEX.parquet", filesystem=fsspec.filesystem("https"),
                      columns=["id", "year", "shard", "offset", "size"], filters=[("id", "=", "MT10_624U_29L")]).to_pandas()
t = index.iloc[0]                                                    # 2022 at Loch Lomond
with rasterio.open(f"/vsisubfile/{t.offset}_{t['size']},/vsicurl/{url}{t.shard}") as src:
    radiance = src.read(1)
```

### Files

- `metadata.parquet`, one row per sample. Core-S2L2A columns: `grid_cell`, `grid_row_u`, `grid_col_r`,
  `product_id`, `timestamp`, `cloud_cover`, `nodata`, `centre_lat`, `centre_lon`, `crs`, `parquet_url`,
  `parquet_row`. Core-S1RTC has `orbit_state` instead of `cloud_cover`.
- `images/*.parquet`, 500 samples per file, one row group per sample. Each band column holds one GeoTIFF
  (bytes); `thumbnail` holds a PNG.

### Reading

| Function (`from majortom import ...`) | Does |
|---|---|
| `filter_metadata(gdf, region, daterange, cloud_cover, nodata)` | filters a GeoDataFrame of metadata |
| `read_row(row, columns, georef=False, raw=False)` | one sample: thumbnail (PIL image) or dict of band arrays; with `georef`, `(array, transform, crs)` per band |
| `filter_download(df, local_dir, source_name, tif_columns=None)` | writes the samples of `df` as GeoTIFFs and PNGs |
| `metadata_from_url(access_url, local_url)` | downloads a metadata file to `local_url` as a GeoDataFrame |

`read_row` reads only the row group of that sample over https.

### Known issues of Core v1

- **Geotransform:** the stored origin is the unrounded window corner, off the pixel grid by up to 5 m (10 m
  bands) or 30 m (60 m bands). Pixels are native and correct. `read_row(georef=True)` and `filter_download`
  correct it by default (`majortom.spec.correct_core_transform`); Core-DEM is left as stored.
- **Window:** 1068 px at 10 m from the south-west corner minus 340 m; 8.3% of cells are not fully inside it.
- **To v2:** `majortom.spec.crop_core_to_v2(data, transform, crs, cell, dataset)` cuts the v2 window out of a
  Core band, or returns `None` (about half of the cells; those need a new read from the source product).
- Duplicate cells and samples without data exist (GitHub issues #17, #8).

## Core-Weather (source.coop, `https://data.source.coop/major-tom/core-weather`)

Weather and air quality around every Core-S2L2A, Core-S2L1C and Core-S1RTC acquisition (2016-2024).

| Path | Rows |
|---|---|
| `acquisitions.parquet` | one per acquisition: `grid_cell`, `dataset`, `acquired`, ... |
| `hourly/year=YYYY/YYYY-MM-DD.parquet` | 49 per acquisition, `step_h` −24..24; 285 columns |
| `daily/year=YYYY/YYYY-MM-DD.parquet` | 61 per acquisition, `step_d` −30..30; 309 columns |

Columns are `<source>_<variable>` (daily: `_mean`, `_min`, `_max` where they apply): `era5_`, `era5land_`,
`cams_`, `cpc_`, `imerg_`, `s5p_`. Values are cell means. Filter on `grid_cell` and `acquired` inside a day file.
ERA5-Land south of 57.1°S is being filled from the Climate Data Store; until that is published, those cells
have no ERA5-Land values.

## Monthly climate (source.coop, `https://data.source.coop/major-tom/climate`)

Datasets: CPC-Monthly, IMERG-Monthly, CAMS-Monthly, ERA5-Monthly, ERA5Land-Monthly, ERA5-Static. Same layout in each:

| Path | One row per |
|---|---|
| `cells.parquet` | cell with data |
| `series/year=YYYY/MM.parquet` | cell, for one month: `<source>_<variable>`, `<source>_valid` |
| `climatology/band=NN.parquet` | cell and calendar month (1991-2020): `<variable>_mean`, `_std`, `n_years` |
| `by_cell/band=NN.parquet` | cell and month, each cell's months together (fast for one cell's series) |
| `overviews/year=YYYY/MM.parquet` | block of 4 × 4 cells: a whole-globe map from one file |
| `climatology_overviews/MM.parquet` | block of 4 × 4 cells, normals |

Band of a grid row: `band = (grid_row_u + 1002) * 20 // 2004`, written with two digits (`band=09.parquet`).
Rows are in Hilbert order with row group statistics on `grid_row_u` and `grid_col_r`: always filter on those two.
Anomaly: join `series` to `climatology` on `grid_cell` and `month`.

## Licences

Core: CC-BY-SA-4.0 (check each dataset card). Core-Weather and climate: the source licences listed in each
README (Copernicus for ERA5, ERA5-Land, CAMS; NOAA for CPC; NASA for IMERG). Cite the Major TOM paper
(Francis and Czerkawski, IGARSS 2024, doi:10.1109/IGARSS53475.2024.10640760) and the sources.
