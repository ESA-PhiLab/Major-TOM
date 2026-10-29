> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed draft (2026-10-28) of the top-level README for source.coop `major-tom/climate`.
> Remove these two notes when publishing. Placeholders: `[ERA5…]` rows are added when those datasets are published.

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate-banner.svg" alt="Major TOM Climate" width="100%">

# Major TOM Climate

Monthly climate records and climate normals for every cell of the [Major TOM](https://github.com/ESA-PhiLab/Major-TOM) grid, from reanalysis, satellite and rain-gauge sources, in Parquet tables keyed by `grid_cell`.

Each dataset is a separate folder with its own README.

<img src="https://data.source.coop/major-tom/climate/figures/overview.png" alt="Annual precipitation (CPC, IMERG), July daily maximum temperature (CPC), surface height (ERA5), aerosol optical depth and surface PM2.5 (CAMS)" width="100%">

| Dataset | What | Source | Record | Grid | Cells | Variables | Size |
|---|---|---|---|---|---|---|---|
| [CPC-Monthly](https://source.coop/major-tom/climate/CPC-Monthly) | rain-gauge precipitation, daily max and min temperature | NOAA CPC Global Unified | 1979–2025 | 0.5° | 1,657,095 (land) | 3 | 12 GB |
| [IMERG-Monthly](https://source.coop/major-tom/climate/IMERG-Monthly) | satellite precipitation | NASA GPM IMERG V07 | 1998–2024 | 0.1° | 5,109,586 (globe) | 1 | 17.5 GB |
| [CAMS-Monthly](https://source.coop/major-tom/climate/CAMS-Monthly) | air quality: trace gases, aerosols, particulate matter | Copernicus CAMS EAC4 | 2003–2025 | 0.75° | 5,113,745 (globe) | 19 | 58 GB |
| [ERA5-Static](https://source.coop/major-tom/climate/ERA5-Static) | fixed surface fields: terrain, land and lake cover, soil and vegetation types | ECMWF ERA5 | — | 0.25° | 5,113,745 (globe) | 16 | 0.1 GB |

Files are at `https://data.source.coop/major-tom/climate/<dataset>/...`.

## Motivation

The sources were chosen so that the main quantities come from more than one type of data:

- **Data types.** Reanalyses (ERA5, ERA5-Land, CAMS) are models fitted to observations and are complete in space and time. Satellite products (IMERG) are observations at finer resolution, with retrieval errors. Rain-gauge analyses (CPC) are ground measurements and are sparse where gauges are sparse.
- **Precipitation from three types of source.** Precipitation affects imagery most (wet soil, flooding, cloud) and is reproduced least well by reanalyses, especially in mountains and the tropics. CPC (gauges), IMERG (satellites) and ERA5 / ERA5-Land (models) can be compared or chosen per region.
- **Weather and land surface.** ERA5 covers the atmosphere and ocean surface at 0.25°. ERA5-Land reruns the land surface at 0.1°, with temperatures adjusted for elevation.
- **Air quality.** CAMS is a global air-quality reanalysis without gaps. Sentinel-5P (in Major TOM Core-Weather) observes at a few kilometres, with gaps under cloud.
- **Access.** Global coverage, long records, open licences and free bulk access, measured before building.

> Routes we tested and did not use: IMERG from NASA GES DISC (0.09 MB/s from our server, against 12.9 MB/s from Earth Engine); Sentinel-5P level-2 swaths (whole orbits must be downloaded, ~18 TB for NO₂ alone); ERA5-Land daily statistics from the Climate Data Store (no accumulated variables such as precipitation).

Monthly normals at the three cells of Major TOM's release test. Gauges and satellites agree on the timing of the Amazon's wet season and differ in amounts in Scotland and the Alps.

<img src="https://data.source.coop/major-tom/climate/figures/places.svg" alt="Monthly precipitation (CPC, IMERG) and temperature at Loch Lomond, Corvara and Manaus" width="100%">

<img src="https://data.source.coop/major-tom/climate/figures/air-quality-seasons.svg" alt="Monthly aerosol optical depth at Loch Lomond, Corvara and Manaus" width="100%">

## Layout

Every monthly dataset has the same three parts:

| Path | One row per | Columns |
|---|---|---|
| `cells.parquet` | cell | `grid_cell`, `grid_row_u`, `grid_col_r`: the cells with data, in the order every file uses |
| `series/year=YYYY/MM.parquet` | cell and month | `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, `<source>_<variable>`, `<source>_valid` (share of the cell with data) |
| `climatology/band=NN.parquet` | cell and calendar month | `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `<variable>_mean`, `<variable>_std` (spread across years), `n_years` |

Rows are ordered along a Hilbert curve over the globe (the order used in Major TOM's TACO releases), so neighbouring cells are stored together. Files use row groups of 65,536 rows with statistics on `grid_row_u` and `grid_col_r`. A filter on these two columns reads only the row groups that cover the area. Reading one cell's 47-year CPC series takes 0.35 GB instead of 6.9 GB.

### Series

One row per cell and month, for every month of the record. Each column holds the dataset's monthly value for that cell, as defined in the dataset's README: a monthly mean for most variables (CAMS concentrations, IMERG's mean precipitation rate in mm/h), the month's total for CPC precipitation, the mean of the month's daily maxima and minima for CPC temperature. `<source>_valid` is the share of the cell's area covered by source pixels with data (1 = fully covered).

### Climatology

One row per cell and calendar month (12 rows per cell), computed from the series over a reference period: 1991–2020 where the record covers it (CPC); the whole record otherwise (IMERG 1998–2024, CAMS 2003–2025).

- `<variable>_mean`: the mean of that calendar month over the reference years, e.g. the average July value 1991–2020.
- `<variable>_std`: the sample standard deviation (n − 1) of that calendar month across the reference years, i.e. the usual year-to-year variation.
- `n_years`: the number of reference years with a value, counted on the dataset's first variable. Another variable can have fewer years in the same cell.

The anomaly of a month is its series value minus the climatology `_mean` of the same cell and calendar month; dividing by `_std` gives a standardised anomaly.

Climatology files are split into 20 bands of grid rows, so a cell's 12 months are in one file: band = (row + 1002) × 20 // 2004, where row is `grid_row_u` (`624U_29L`: row 624, band 16; `35D_667L`: row −35, band 9).

## Reading

```python
import pandas as pd

url = "https://data.source.coop/major-tom/climate"
july = pd.read_parquet(f"{url}/CPC-Monthly/series/year=2020/07.parquet")          # one month, every cell
here = [("grid_row_u", "=", 624), ("grid_col_r", "=", -29)]                        # Loch Lomond, 624U_29L
normals = pd.read_parquet(f"{url}/CPC-Monthly/climatology/band=16.parquet", filters=here)   # 12 months
series = pd.concat(pd.read_parquet(f"{url}/CPC-Monthly/series/year={y}/{m:02d}.parquet", filters=here)
                   for y in range(2015, 2025) for m in range(1, 13))                 # 10 years, one cell
```

```sql
-- DuckDB: a region, using the row and column ranges of the cells it covers
SELECT grid_cell, cpc_precip FROM 'https://data.source.coop/major-tom/climate/CPC-Monthly/series/year=2020/07.parquet'
WHERE grid_row_u BETWEEN 610 AND 640 AND grid_col_r BETWEEN -60 AND 0;
```

```python
anomaly = series.merge(normals, on=["grid_cell", "grid_row_u", "grid_col_r", "month"])
anomaly["precip_anomaly"] = anomaly.cpc_precip - anomaly.cpc_precip_mean    # mm, against 1991-2020
```

Join to any Major TOM dataset on `grid_cell`; join a series to its climatology on `grid_cell` and `month` to get anomalies. For xarray: `series.set_index(["grid_cell", "year", "month"]).to_xarray()`.

## Values

- **Cell means.** Each value is the area-weighted mean of the source pixels overlapping a 10 km cell. Nothing is interpolated: where a source pixel is larger than a cell (0.25° to 0.75°), neighbouring cells share its value.
- **Units as published** by each source, not converted; each dataset's README lists them.
- **Reanalyses.** ERA5 and CAMS values are model states fitted to observations.
- **Source errors.** Values are not edited. Known problems are listed in each dataset's README.

## Licences

Each dataset keeps its source's terms; see its README for the exact acknowledgement and citations. In short: CPC has no usage restrictions (NOAA PSL asks to be acknowledged); IMERG is NASA open data (cite the dataset); ERA5 and CAMS are under CC BY 4.0 (since 2 July 2025), credited as "Contains modified Copernicus … Service information [year]", as ECMWF recommends.

## Author and citation

Mikolaj Czerkawski, Asterisk Labs.

To cite this dataset, cite the Major TOM paper below and the sources listed in each dataset's README. A Major TOM v2 paper will replace this reference when it is published.

```bibtex
@inproceedings{Francis2024MajorTOM,
  author={Francis, Alistair and Czerkawski, Mikolaj},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium},
  title={Major TOM: Expandable Datasets for Earth Observation},
  year={2024},
  pages={2935-2940},
  doi={10.1109/IGARSS53475.2024.10640760}
}
```

---

<img src="https://data.source.coop/major-tom/climate/brand/earth-compress.svg" alt="Earth Compress" width="200">

Developed by [Asterisk Labs](https://asterisk.coop) as part of [Earth Compress](https://asterisk.coop/project-pages/earth-compress.html), an ARIA-funded project jointly led with the National Oceanography Centre.
