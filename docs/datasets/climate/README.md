> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed draft (2026-10-28) of the top-level README for source.coop `major-tom/climate`.
> Remove these two notes when publishing. Placeholders: `[ERA5…]` rows are added when those datasets are published.

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate.svg" alt="Major TOM Climate" width="420">

# Major TOM Climate

Monthly climate records and climate normals for every cell of the [Major TOM](https://github.com/ESA-PhiLab/Major-TOM) grid: decades of global reanalysis, satellite and rain-gauge data, condensed into compact tables keyed by `grid_cell`. Climate context for any place on Earth becomes a join, not a download.

Each dataset is a folder of its own, with its own README, and can be used without the others.

| Dataset | What | Source | Record | Grid | Cells | Variables | Size |
|---|---|---|---|---|---|---|---|
| [CPC-Monthly](CPC-Monthly/) | rain-gauge precipitation, daily max and min temperature | NOAA CPC Global Unified | 1979–2025 | 0.5° | 1,657,095 (land) | 3 | 12 GB |
| [IMERG-Monthly](IMERG-Monthly/) | satellite precipitation | NASA GPM IMERG V07 | 1998–2024 | 0.1° | 5,109,586 (globe) | 1 | 17.5 GB |
| [CAMS-Monthly](CAMS-Monthly/) | air quality: trace gases, aerosols, particulate matter | Copernicus CAMS EAC4 | 2003–2025 | 0.75° | 5,113,745 (globe) | 19 | 58 GB |
| [ERA5-Static](ERA5-Static/) | fixed surface fields: terrain, land and lake cover, soil and vegetation types | ECMWF ERA5 | — | 0.25° | 5,113,745 (globe) | 16 | 0.1 GB |

Every folder is also reachable directly: `https://data.source.coop/major-tom/climate/<dataset>/...`

## Motivation

The sources were chosen to complement each other, so that every quantity that matters for Earth observation comes from more than one kind of evidence:

- **Three kinds of data.** Reanalyses (ERA5, ERA5-Land, CAMS) are models fitted to observations: complete in space and time, but modelled. Satellite products (IMERG) observe directly, at finer resolution, with their own retrieval errors. Rain-gauge analyses (CPC) are ground measurements, independent of both, but sparse where gauges are sparse.
- **Precipitation three ways.** Precipitation is the variable that matters most for imagery (wet soil, flooding, cloud) and the one reanalyses reproduce least well, especially in mountains and the tropics. CPC (gauges), IMERG (satellite) and ERA5 / ERA5-Land (models) let users compare, or pick the one they trust for their region.
- **Weather and land surface.** ERA5 covers the whole atmosphere and ocean surface at 0.25°; ERA5-Land reruns the land surface at 0.1° with temperatures adjusted for elevation.
- **Air quality.** CAMS is the only global, gap-free air quality reanalysis; Sentinel-5P (in Major TOM Core-Weather) observes at a few kilometres but with gaps under cloud.
- **Practical.** Global, long records, open licences, and free bulk access that we measured before building.

> Routes we tested and did not use: IMERG from NASA GES DISC (0.09 MB/s from our server, against 12.9 MB/s from Earth Engine); Sentinel-5P level-2 swaths (whole orbits must be downloaded, ~18 TB for NO₂ alone); ERA5-Land daily statistics from the Climate Data Store (no accumulated variables such as precipitation).

## Layout

Every monthly dataset has the same three parts:

| Path | One row per | Columns |
|---|---|---|
| `cells.parquet` | cell | `grid_cell`, `grid_row_u`, `grid_col_r`: the cells with data, in the order every file uses |
| `series/year=YYYY/MM.parquet` | cell and month | `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, `<source>_<variable>`, `<source>_valid` (share of the cell with data) |
| `climatology/band=NN.parquet` | cell and calendar month | `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `<variable>_mean`, `<variable>_std` (spread across years), `n_years` |

**Rows follow a Hilbert curve** over the globe (the same order as Major TOM's TACO releases), so nearby cells sit together, and files are written in row groups of 65,536 rows with statistics on `grid_row_u` and `grid_col_r`. A filter on those two integers reads only the row groups covering the area: one cell's 47-year CPC series reads 0.35 GB instead of 6.9 GB.

Climatology files are split into 20 bands of grid rows, so a cell's 12 months sit in one file: band = (row + 1002) × 20 // 2004, where row is `grid_row_u` (`624U_29L`: row 624, band 16; `35D_667L`: row −35, band 9).

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

Join to any Major TOM dataset on `grid_cell`; join a series to its climatology on `grid_cell` and `month` to get anomalies. For xarray: `series.set_index(["grid_cell", "year", "month"]).to_xarray()`.

## What the values are

- **Cell means.** Each value is the area-weighted mean of the source pixels overlapping a 10 km cell. Nothing is interpolated: where a source pixel is larger than a cell (0.25° to 0.75°), neighbouring cells share its value.
- **Units as published** by each source, not converted; each dataset's README lists them.
- **Reanalyses are models.** ERA5 and CAMS values are model states fitted to observations, not measurements at a place.
- **Source errors are kept.** Values are not edited; known problems in a source are listed in its README.

## Licences

Each dataset keeps its source's terms; see its README for the exact acknowledgement and citations. In short: CPC has no usage restrictions (NOAA PSL asks to be acknowledged); IMERG is NASA open data (cite the dataset); ERA5 and CAMS are under CC BY 4.0 (since 2 July 2025), credited as "Contains modified Copernicus … Service information [year]", as ECMWF recommends.

---

<img src="https://data.source.coop/major-tom/climate/brand/earth-compress.svg" alt="Earth Compress" width="200">

Developed by [Asterisk Labs](https://asterisk.coop) as part of [Earth Compress](https://asterisk.coop/project-pages/earth-compress.html), an ARIA-funded project jointly led with the National Oceanography Centre.
