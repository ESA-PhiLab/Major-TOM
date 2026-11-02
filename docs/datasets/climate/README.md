> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** reviewed (Miko) and published 2026-11-02 (File layout, ERA5Land-Monthly row). Earlier: by_cell/ and overviews/ sections and working code examples added 2026-10-30 (examples run as written). Earlier status: unreviewed draft (2026-10-28) of the top-level README for source.coop `major-tom/climate`.
> Remove these two notes when publishing. How by_cell/ files are written: added 2026-10-31, not yet reviewed or published; read table re-measured on the new by_cell/ files 2026-10-31 (docs/evidence/climate_bycell_bench.py, the app). By cell and overviews section rewritten 2026-10-31 (measurements from docs/reports/2026-10-30-climate-globe-read-experiments.md). ERA5-Monthly row and figures added 2026-10-30; ERA5Land-Monthly row added 2026-11-02 (draft README). Sizes include by_cell/ and overviews/. File layout section (diagram, settings, sizes, reads) added 2026-11-02, not yet reviewed or published; climatology/ now has a page index.

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate-banner.svg" alt="Major TOM Climate" width="100%">

# Major TOM Climate

Monthly climate records and climate normals for every cell of the [Major TOM](https://github.com/ESA-PhiLab/Major-TOM) grid, from reanalysis, satellite and rain-gauge sources, in Parquet tables keyed by `grid_cell`.

Each dataset is a separate folder with its own README.

<img src="https://data.source.coop/major-tom/climate/figures/overview.png" alt="Annual precipitation (CPC, IMERG, ERA5), July temperature (ERA5, CPC), surface height (ERA5), aerosol optical depth and surface PM2.5 (CAMS)" width="100%">

| Dataset | What | Source | Record | Grid | Cells | Variables | Size |
|---|---|---|---|---|---|---|---|
| [CPC-Monthly](https://source.coop/major-tom/climate/CPC-Monthly) | rain-gauge precipitation, daily max and min temperature, gauges nearby | NOAA CPC Global Unified | 1979–2025 | 0.5° | 1,657,095 (land) | 4 | 21.2 GB |
| [IMERG-Monthly](https://source.coop/major-tom/climate/IMERG-Monthly) | satellite precipitation and its quality fields | NASA GPM IMERG V07 | 1998–2024 | 0.1° | 5,113,745 (globe) | 5 | 54.4 GB |
| [CAMS-Monthly](https://source.coop/major-tom/climate/CAMS-Monthly) | air quality: trace gases, aerosols, particulate matter | Copernicus CAMS EAC4 | 2003–2025 | 0.75° | 5,113,745 (globe) | 19 | 110.9 GB |
| [ERA5-Static](https://source.coop/major-tom/climate/ERA5-Static) | fixed surface fields: terrain, land and lake cover, soil and vegetation types | ECMWF ERA5 | — | 0.25° | 5,113,745 (globe) | 16 | 0.12 GB |
| [ERA5Land-Monthly](https://source.coop/major-tom/climate/ERA5Land-Monthly) | land surface at 0.1°: temperature, soil water and temperature, snow, runoff, evaporation, fluxes, lakes | ECMWF ERA5-Land | 1950–2025 | 0.1° | 1,524,009 (land) | 49 | 469.8 GB |
| [ERA5-Monthly](https://source.coop/major-tom/climate/ERA5-Monthly) | weather and land surface: temperature, humidity, wind, pressure, precipitation, radiation, clouds, soil, snow, waves | ECMWF ERA5 | 1940–2025 | 0.25° | 5,113,745 (globe) | 201 | 3.03 TB (`by_cell/` being built) |

Files are at `https://data.source.coop/major-tom/climate/<dataset>/...`.

## Motivation

The sources were chosen so that the main quantities come from more than one type of data:

- **Data types.** Reanalyses (ERA5, ERA5-Land, CAMS) are models fitted to observations and are complete in space and time. Satellite products (IMERG) are observations at finer resolution, with retrieval errors. Rain-gauge analyses (CPC) are ground measurements and are sparse where gauges are sparse.
- **Precipitation from three types of source.** Precipitation affects imagery most (wet soil, flooding, cloud) and is reproduced least well by reanalyses, especially in mountains and the tropics. CPC (gauges), IMERG (satellites) and ERA5 / ERA5-Land (models) can be compared or chosen per region.
- **Weather and land surface.** ERA5 covers the atmosphere and ocean surface at 0.25°. ERA5-Land reruns the land surface at 0.1°, with temperatures adjusted for elevation.
- **Air quality.** CAMS is a global air-quality reanalysis without gaps. Sentinel-5P (in Major TOM Core-Weather) observes in pixels of 5.5 × 3.5 km at nadir, with gaps under cloud.
- **Access.** Global coverage, long records, open licences and free bulk access, measured before building.

> Routes we tested and did not use: IMERG from NASA GES DISC (0.09 MB/s from our server, against 12.9 MB/s from Earth Engine); Sentinel-5P level-2 swaths (whole orbits must be downloaded: an estimated 18 TB for NO₂ alone); ERA5-Land daily statistics from the Climate Data Store (no accumulated variables such as precipitation).

Monthly normals at the three cells of Major TOM's release test. Gauges and satellites agree on the timing of the Amazon's wet season and differ in amounts in Scotland and the Alps.

<img src="https://data.source.coop/major-tom/climate/figures/places.svg" alt="Monthly precipitation (CPC, IMERG, ERA5) and temperature at Loch Lomond, Corvara and Manaus" width="100%">

<img src="https://data.source.coop/major-tom/climate/figures/air-quality-seasons.svg" alt="Monthly aerosol optical depth at Loch Lomond, Corvara and Manaus" width="100%">

## Layout

Every monthly dataset has the same parts:

| Path | One row per | Columns |
|---|---|---|
| `cells.parquet` | cell | `grid_cell`, `grid_row_u`, `grid_col_r`: the cells with data, in the order every file uses |
| `series/year=YYYY/MM.parquet` | cell and month | `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, `<source>_<variable>`, `<source>_valid` (share of the cell with data) |
| `climatology/band=NN.parquet` | cell and calendar month | `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `<variable>_mean`, `<variable>_std` (spread across years), `n_years` |
| `by_cell/band=NN.parquet` | cell and month | as `series/`, but each cell's months together: rows by cell, then `year`, `month` (CPC, IMERG, CAMS) |
| `overviews/year=YYYY/MM.parquet` | block of 4 × 4 cells and month | `block_row`, `block_col`, `n_cells`, each variable's mean: a whole-globe map from one file, reading one column |
| `climatology_overviews/MM.parquet` | block of 4 × 4 cells and calendar month | `block_row`, `block_col`, `n_cells`, each variable's normal (block mean of `<variable>_mean`): a whole-globe map of a month's normal from one small file |

Rows are ordered along a Hilbert curve over the globe (the order used in Major TOM's TACO releases), so neighbouring cells are stored together. Every file has statistics on `grid_row_u` and `grid_col_r` for each row group, so a filter on these two columns reads only the row groups that cover the area. Reading one cell's 47-year CPC series from `series/` takes 0.35 GB instead of 6.9 GB.

### File layout

A Parquet file is split into row groups (blocks of consecutive rows), each row group into one column chunk per
column, and each column chunk into pages. The footer at the end of the file lists the row groups with the minimum
and maximum of each column. Readers fetch the footer first, then only the parts they need. The files here are
written so that one cell, one place or one month needs few and small parts:

```
cells.parquet                    every cell with data, along the Hilbert curve: rank 0, 1, 2, ...
  |                              (every file below keeps this order)
  +- series/year=2020/07.parquet       one month, one row per cell
  |    row group 0: cells 0 - 65,535          statistics on grid_row_u, grid_col_r: a place is one row group
  |    row group 1: cells 65,536 - 131,071
  |
  +- climatology/band=16.parquet       one band of grid rows, 12 rows per cell (calendar months 1-12)
  |    row group 0: rows 0 - 65,535 (5,461 cells and 4 rows)
  |      cpc_precip_mean: pages of 4 KB      page index: where each page starts and its first row
  |
  +- by_cell/band=16.parquet           one band, every month of each cell together
  |    row group 0: cells 0 - 1,023 x 564 months
  |      cpc_precip: pages of 4 KB, written one cell's 564 months at a time
  |    footer: row groups with statistics; page index
  |
  +- overviews/year=2020/07.parquet    one month, one row per block of 4 x 4 cells (whole globe; CPC: 2 MB)
```

A cell's rows in `climatology/` and `by_cell/` follow from its rank: the band holds the cells of `cells.parquet`
whose row is in the band, in the same order. With `rank` the cell's place among them, its rows are
`rank × 12` to `rank × 12 + 11` in `climatology/` and `rank × months` onwards in `by_cell/`. A reader that knows
the rows and uses the page index reads the footer, the page index of each column it needs, and the pages that hold
those rows. A reader without the page index (pandas, pyarrow) uses the statistics and reads the whole row group.

Settings per folder:

| Folder | Rows | Row group | Page index | Value dictionaries |
|---|---|---|---|---|
| `series/` | one per cell | 65,536 rows (65,536 cells) | no | yes (pyarrow default) |
| `climatology/` | 12 per cell | 65,536 rows (5,461 cells and 4 rows: a cell's 12 rows can span two row groups) | yes, pages of 4 KB | no |
| `by_cell/` | months per cell | 1,024 cells | yes, pages of 4 KB, written one cell at a time | no |
| `overviews/`, `climatology_overviews/` | one per 4 × 4 block | the whole file | no | yes |

Why these settings:

- **Row groups of 1,024 cells in `by_cell/`.** The footer holds statistics for every row group and column. Few row
  groups keep it small: ERA5's band 00 (31,712 cells) has a 0.73 MB footer with row groups of 1,024 cells (31),
  and had 10.4 MB with row groups of 65,536 rows (63 ERA5 cells, 500 row groups). Band 04 (259,858 cells, 254 row
  groups) has 5.84 MB.
- **`climatology/` keeps row groups of 65,536 rows.** ERA5's climatology footer is already 3.3 MB (74 row groups,
  407 columns); row groups of 1,024 cells would give band 10 394 row groups instead of 74, and a footer 5.3 times
  as large.
- **Page index.** Without it, a reader decodes a whole column chunk to reach one cell's rows. With it, one page.
- **No value dictionaries.** A column chunk's dictionary page is read whole before any of its pages: one ERA5
  variable of one cell costs 1.8 MB with dictionaries and 0.8 MB without (table below). Values are stored with
  byte-stream-split encoding and zstd instead.
  `grid_cell`, the position columns, `year`, `month` and `n_years` keep dictionaries.

Sizes on source.coop, and the largest footer of a file in each folder:

| Folder | CPC | IMERG | CAMS | ERA5-Land | ERA5 |
|---|---|---|---|---|---|
| `series/` | 12.4 GB, 0.03 MB | 33.7 GB, 0.10 MB | 55.4 GB, 0.22 MB | 286.6 GB, 0.16 MB | 2,769.9 GB, 1.74 MB |
| `climatology/` | 0.19 GB, 0.05 MB | 1.41 GB, 0.13 MB | 2.20 GB, 0.39 MB | 4.47 GB, 0.37 MB | 38.0 GB, 3.31 MB |
| `by_cell/` | 7.46 GB, 0.19 MB | 17.4 GB, 0.53 MB | 44.2 GB, 1.15 MB | 158.7 GB, 1.00 MB | |
| `overviews/` | 1.16 GB | 1.89 GB | 8.70 GB | 19.8 GB | 221.3 GB |
| `climatology_overviews/` | 0.03 GB | 0.07 GB | 0.38 GB | 0.27 GB | 2.61 GB |

One cell read with hyparquet (a JavaScript Parquet reader, with `useOffsetIndex`) over HTTP range requests:

| Read | Bytes | Requests | Time |
|---|---|---|---|
| CPC `by_cell/`, one cell's 564 months of one variable (source.coop) | 0.18 MB | 10 | 0.8 s |
| ERA5 `by_cell/` band 00, one cell's 1,032 months, one variable / all 201 (local server) | 0.8 MB / 3.3 MB | | |
| ERA5 `climatology/` band 10, one cell's 12 normals, one variable / all 201 (local server) | 3.3 MB / 3.6 MB | 4 / 404 | 0.2 s / 0.6 s |

Most of the ERA5 climatology read is its footer (3.3 MB). Before the page index and without dictionaries, the same
reads cost 3.6 MB and 51 MB. Measured on ERA5's `by_cell/` band 00 (31,712 cells, 201 variables, 1,032 months):

| Layout | File | Footer | One variable | All 201 variables |
|---|---|---|---|---|
| Row groups of 65,536 rows (63 cells) | 18.5 GB | 10.4 MB | 11.4 MB | 119 MB |
| Row groups of 1,024 cells | 14.2 GB | 0.7 MB | 16.4 MB | 1.8 GB |
| Row groups of 1,024 cells, page index | 14.7 GB | 0.8 MB | 1.8 MB | 109 MB |
| Row groups of 1,024 cells, page index, no value dictionaries (used) | 12.8 GB | 0.7 MB | 0.8 MB | 3.3 MB |

```js
// hyparquet; rank and count from cells.parquet (the band's cells, in order)
const months = Number(metadata.num_rows) / count, first = rank * months
await parquetRead({ file, metadata, compressors, useOffsetIndex: true, rowStart: first, rowEnd: first + months,
                    columns: ['year', 'month', 'cpc_precip'], onComplete: rows => console.log(rows) })
```

### Series

One row per cell and month, for every month of the record. Each column holds the dataset's monthly value for that cell, as defined in the dataset's README: a monthly mean for most variables (CAMS concentrations, IMERG's mean precipitation rate in mm/h), the month's total for CPC precipitation, the mean of the month's daily maxima and minima for CPC temperature. `<source>_valid` is the share of the cell's area covered by source pixels with data (1 = fully covered).

### By cell and overviews

`series/` and `by_cell/` hold the same values in two orders, each suited to one question. A month over many cells
is one `series/` file. One place over time is the question `series/` answers worst: every monthly file has to be
opened. `by_cell/` stores each cell's whole record together, in one row group of one file. Measured for Loch
Lomond's CPC record (564 months), read from source.coop:

| Read | From `series/` | From `by_cell/` |
|---|---|---|
| One cell, every month (in a browser, through the page index) | 1,140 requests, 153 MB, 10.2 s | 10 requests, 0.18 MB, 0.8 s |
| One cell, every month (pandas, the example below: the cell's row group) | one request per file, 564 files | 1.5 s |
| 1,000 cells chosen at random, every month (pyarrow, 16 reads at once, whole row groups) | 261 s, 2.3 GB | 60 s, 1.1 GB |
| 1,000 neighbouring cells, every month | | 1.1 s, 2.3 MB |

pandas and pyarrow, with a filter on `grid_row_u` and `grid_col_r` as in the dataset READMEs, read the whole row
group holding the cell (1,024 cells); readers that use the page index read only the cell's pages
([File layout](#file-layout)).

For training on series, read `by_cell/` a row group at a time (each holds whole series of neighbouring cells) and
shuffle row groups, or download a dataset's `by_cell/` folder first.

`overviews/` holds means over blocks of 4 × 4 cells, one small file per month, for maps of the whole globe. The
blocks follow the grid's numbering: rows 4i to 4i + 3 (`grid_row_u` + 1002) and, in each of those rows, columns 4j
to 4j + 3 (`grid_col_r` + 2004). Major TOM numbers each row's cells from the prime meridian; rows are 9.99 km high (180° / 2,004)
and each row is split into whole cells of at most 10 km, so a block is at most 40 × 40 km in distance north–south and east–west of the prime meridian:
the sinusoidal projection the figures above use, in which the grid is regular. In longitude and latitude, blocks far
from the prime meridian are sheared, as the cells are. `n_cells` counts a block's cells with a value (coasts and
the edges of the grid have fewer than 16). The Major TOM Climate Globe app reads these files when the whole globe
is in view and the monthly files when zoomed in.

ERA5-Monthly's `by_cell/` is being built (2026-11-02). Until it is published, read a place over time from
`series/`, selecting columns: each of the 1,032 monthly files has a 1.74 MB footer (207 columns), so one cell's record
of one variable costs 2.96 GB, 1.80 GB of it footers (Loch Lomond, `era5_t2m`, from the files' metadata).

### Climatology

One row per cell and calendar month (12 rows per cell), computed from the series over a reference period: 1991–2020 where the record covers it (CPC); the whole record otherwise (IMERG 1998–2024, CAMS 2003–2025).

- `<variable>_mean`: the mean of that calendar month over the reference years, e.g. the average July value 1991–2020.
- `<variable>_std`: the sample standard deviation (n − 1) of that calendar month across the reference years, i.e. the usual year-to-year variation.
- `n_years`: the number of reference years with a value, counted on the dataset's first variable. Another variable can have fewer years in the same cell.

The anomaly of a month is its series value minus the climatology `_mean` of the same cell and calendar month; dividing by `_std` gives a standardised anomaly.

Climatology files are split into 20 bands of grid rows, so a cell's 12 months are in one file: band = (row + 1002) × 20 // 2004, where row is `grid_row_u` (`624U_29L`: row 624, band 16; `35D_667L`: row −35, band 9, file `band=09.parquet`). `by_cell/` uses the same bands.

## Reading

```python
import fsspec
import pandas as pd

https = fsspec.filesystem("https")                                                   # reads only the parts needed
url = "https://data.source.coop/major-tom/climate"
july = pd.read_parquet(f"{url}/CPC-Monthly/series/year=2020/07.parquet", filesystem=https)       # one month, every cell
here = [("grid_row_u", "=", 624), ("grid_col_r", "=", -29)]                                       # Loch Lomond, 624U_29L
normals = pd.read_parquet(f"{url}/CPC-Monthly/climatology/band=16.parquet", filesystem=https, filters=here)  # 12 months
series = pd.read_parquet(f"{url}/CPC-Monthly/by_cell/band=16.parquet", filesystem=https, filters=here)      # 564 months
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

Built by [Mikolaj Czerkawski](https://github.com/mikonvergence), Asterisk Labs.

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
