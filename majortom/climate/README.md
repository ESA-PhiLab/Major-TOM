> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review, 2026-10-30.

# majortom.climate

Produces weather and climate values for Major TOM grid cells, from global sources (ERA5, ERA5-Land, CAMS, NOAA CPC,
GPM IMERG, Sentinel-5P). It produces data only; publishing layouts and the upload are in `majortom.release`.
Needs `pip install 'majortom[climate]'`.

## Two products

```
global source fields (latitude/longitude grids)
  -> readers/: one module per access method, a field per variable and time
  -> weights.py: source grid -> one value per 10 km cell (area-weighted)
  |
  +-> monthly.py: [Source]-Monthly, every cell, every month of the record
  |     -> series/year=YYYY/MM.parquet + cells.parquet, climatology/band=NN.parquet (normals), static.parquet
  |     -> majortom.release.tables (layout, overviews, by-cell) -> major-tom/climate
  |
  +-> stream.py, days.py, series.py: Core-Weather, the hours and days around each Sentinel acquisition
        -> hourly/ and daily/ tables per source -> merge.py -> one hourly and one daily table
```

## Modules

| Module | What it does |
|---|---|
| `sources.py` | each source: where it lives, its grid, variables, units, how it is read |
| `era5_catalogue.py` | the ERA5 variables kept for Core-Weather, ERA5-Monthly and the static table |
| `readers/` | `zarr` (ARCO-ERA5), `edh` (Earth Data Hub, ERA5-Land hourly), `queue` (Climate / Atmosphere Data Store), `files` (CPC, IMERG), `earthengine` (Sentinel-5P, IMERG half-hours) |
| `weights.py` | cell values from a latitude/longitude source grid |
| `monthly.py` | monthly datasets, the static table and the climatology |
| `stream.py` | Core-Weather for any number of acquisitions: each hour read once, memory bounded |
| `days.py` | daily sources around each acquisition (ERA5-Land, CPC, Sentinel-5P) |
| `series.py` | the hourly and daily tables of Core-Weather |
| `merge.py` | per-source Core-Weather folders -> one hourly and one daily table |

## Command line

```
python -m majortom.climate monthly     --source era5-monthly --years 1940 2025 --out <dir>
python -m majortom.climate climatology --out <dir> --period 1991 2020
python -m majortom.climate static      --source era5-all --out <dir>
python -m majortom.climate stream      --samples <metadata.parquet> [...] --source era5 --out <dir>
python -m majortom.climate merge       --root <Core-Weather> --sources era5 cams cpc s5p --out <dir>
python -m majortom.climate fetch       --source cams --months 2003-01 2003-12
```

Runs resume where they stopped. Credentials for each source: `python -m majortom.build.credentials`.

## More

- `docs/CLIMATE.md`: sources, the cell-value method, both products, decisions
- `docs/ERA5_VARIABLES.md`: the ERA5 variable catalogue
- `docs/datasets/climate/`: the published dataset READMEs
