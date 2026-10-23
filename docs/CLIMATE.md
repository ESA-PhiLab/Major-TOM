> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** reviewed (Miko, 2026-10-23), sources included. Design only; no code exists yet.

# Weather and climate per grid cell — design

Two kinds of dataset, built by one module, `majortom.climate`:

| Dataset | What | One row per | Sources |
|---|---|---|---|
| **Core-Weather** | weather around each Core acquisition: hourly ±24 h, daily ±30 days | acquisition × time step | all sources in one resource |
| **[Source]-Monthly**, e.g. `ERA5Land-Monthly` | monthly values for every land cell, the whole record, plus a climatology | cell × month | one dataset per source |

Both join to every Major TOM dataset on `grid_cell`. Core-Weather also joins on the acquisition time. The same tool builds Core-Weather-style tables for any expansion dataset (section 6).

## 1. Why two datasets

- **Monthly, global:** general purpose. It covers any cell, including cells no dataset has sampled yet, and gives each month's normal through the climatology.
- **Core-Weather, dense:** what happened at the time of the image. Rain in the days before (wet soil, which matters for Sentinel-1), snow, heat, and wind during a Sentinel-1 pass. With the climatology, "this July was 3 °C above normal" becomes one join.
- **Why not dense series for all land:** hourly values for every land cell over the whole record are far larger than the series around the ~2 million Core acquisitions. Sampling close to the acquisitions keeps the dense part small.

## 2. Sources

| Source | Variables | Grid | Time steps, period | Free access | Licence |
|---|---|---|---|---|---|
| [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | full atmosphere and surface | 0.25° (~31 km) | hourly, 1940– | Climate Data Store (free key); [ARCO-ERA5](https://github.com/google-research/arco-era5) Zarr on Google Cloud (anonymous, hourly, 0.25°); Earth Engine | CC-BY 4.0 (Copernicus) |
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) | land surface: temperature, soil water, snow, runoff, fluxes | 0.1° (~9 km) | hourly and monthly means, 1950– (2–3 months' delay) | Climate Data Store; Earth Engine | CC-BY (Copernicus) |
| [CAMS global reanalysis (EAC4)](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4) | air quality: NO₂, CO, O₃, aerosol optical depth, PM2.5 | 0.75° (~80 km) | 3-hourly, 2003–2025, no gaps | Atmosphere Data Store (free key, as the Climate Data Store) | CC-BY (Copernicus) |
| Sentinel-5P TROPOMI L3 | NO₂, CO, O₃, CH₄, SO₂, HCHO, aerosol index | each orbit gridded by Earth Engine from L2 pixels of 5.5 × 3.5 km at nadir (7 × 7 km for CO, CH₄) | one overpass a day (~13:30 local), mid-2018–; gaps under clouds and between swaths | [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2) only | Copernicus Sentinel data terms |
| [NOAA CPC Global Unified](https://developers.google.com/earth-engine/datasets/catalog/NOAA_CPC_Temperature) | gauge-based precipitation, daily min and max temperature | 0.5° | daily, 1979– | NOAA PSL downloads (anonymous); Earth Engine | no restriction; attribution requested |
| [GPM IMERG Final V07](https://data.nasa.gov/dataset/gpm-imerg-final-precipitation-l3-half-hourly-0-1-degree-x-0-1-degree-v07-gpm-3imerghh-at-g-fb698) | satellite precipitation | 0.1° | half-hourly, June 2000– | Earthdata (token, as for OPERA) | NASA open data |

To verify before the first build:
- Sentinel-5P: the exact redistribution wording in the Copernicus Sentinel data terms, and the L3 pixel size on Earth Engine.
- ERA5: the current end date in the ARCO-ERA5 store's metadata.
- NOAA CPC precipitation: the PSL product page (the temperature product is confirmed).
- ERA5-Land: whether the Climate Data Store's post-processed daily statistics for ERA5-Land cover the variables we need. It would avoid reading hourly ERA5-Land for the daily table.

Two air-quality sources, deliberately: CAMS has no gaps and covers all of Core (2017–), but at ~80 km and from a model; Sentinel-5P observes at a few km, but only from mid-2018 and with gaps. Sentinel-5P L2 swaths from CDSE were rejected: whole files must be downloaded (about 0.5 GB per orbit for NO₂, ~14 orbits a day), on the order of 18 TB for NO₂ alone over Core's days, for values averaged over 10 km cells anyway.

**Which table each source feeds:**

| Source | Hourly table | Daily table |
|---|---|---|
| ERA5 | yes | yes |
| ERA5-Land | no | yes |
| IMERG | yes (half-hours summed to hours) | yes |
| CAMS | yes, every third hour (others empty) | yes |
| NOAA CPC | no | yes |
| Sentinel-5P | no | yes |

ERA5-Land stays out of the hourly table: its atmosphere comes from ERA5 (it is a land-surface rerun at 9 km, with temperature adjusted for elevation), so hourly values add spatial detail but no new weather dynamics, and its slow variables (soil water, snow) barely change within 48 hours. Hourly ERA5-Land would add only ~3 GB, but ~13 TB of reads through the Climate Data Store queue.

## 3. From a source grid to a cell value

```
source field (one time step, lat/lon grid)
  -> cell outline in lat/lon (majortom.spec.cell_polygon; Major TOM cells are lat/lon boxes)
  -> overlap of each source pixel with the outline, as weights (computed once per cell and source)
  -> weighted mean over valid pixels                         -> value
  -> share of the cell's area with valid data                -> valid
```

- **Weights are precomputed** once per source grid as a sparse matrix (cells × source pixels). Each time step is then one matrix-vector product over the whole globe.
- **Cells smaller than a source pixel** (ERA5 at 31 km against 10 km cells) get the one or two overlapping pixels. Values are not interpolated: a cell value never claims more detail than the source has.
- **Valid share:** ERA5-Land has no values over the sea, Sentinel-5P has gaps under clouds and between swaths. A coastal cell's value comes from its land part, and `valid` says how much that is.

## 4. [Source]-Monthly

Two tables per source. The series is the expensive part; the climatology is computed from it.

**Series:** one row per land cell and month, the whole record.

```
grid_cell  year  month  t2m    tp     swvl1  ...  valid
624U_29L   2021  7      15.2   0.09   0.31        1.0
```

Files split by year (`series/year=2021/part-*.parquet`): a new year adds files and never changes old ones.

**Climatology:** one row per cell and calendar month, over a fixed reference period (1991–2020, the current WMO standard normal).

```
grid_cell  month  t2m_mean  t2m_std  tp_mean  tp_std  ...  n_years
624U_29L   7      14.6      0.9      0.11     0.05         30
```

- `std` is the spread across the 30 years: how unusual a given year is.
- Files split by grid row band, so one cell's 12 months sit in one file.
- **Coverage:** every cell the source has values for. Sources defined over land only (ERA5-Land, NOAA CPC) cover land plus one cell beyond the coast; sources with values over the sea (ERA5, CAMS, IMERG) cover the whole globe, so ocean variables (sea surface temperature, waves) are included.
- **Size, ERA5-Land, 10 variables:** about 1.5 million land cells (estimate, to be counted on the grid) × 912 months (1950–2025) ≈ 14 billion values, about 55 GB as float32 before compression.
- **Size, ERA5, 10 variables, whole globe:** about 5.1 million cells × 1,032 months (1940–2025) ≈ 53 billion values, about 210 GB as float32 before compression (land only would be ~66 GB). Reads are the same either way, ~41 GB of monthly fields: the globe costs storage, not processing.
- Each climatology is about 1 GB (ERA5 globe: about 3 GB).

## 5. Core-Weather

**Acquisitions, not samples:** Core-S2L2A and Core-S2L1C mostly share acquisitions. Rows are keyed by (`grid_cell`, `acquired`), the acquisition time in UTC. A small link table maps Core samples to acquisitions:

```
acquisitions:  grid_cell  acquired              dataset      product_id
               624U_29L   2021-07-14T11:21:19Z  Core-S2L2A   S2B_MSIL2A_...
               624U_29L   2021-07-14T11:21:19Z  Core-S2L1C   S2B_MSIL1C_...
```

Core-DEM has no acquisition time; it joins the monthly datasets only.

**Hourly table:** steps −24 to +24 h. Step 0 is the full hour nearest the acquisition (ERA5 states are instantaneous at full hours, so within 30 minutes of the image); step −1 is the hour before, +1 the hour after.

```
grid_cell  acquired              step_h  era5_t2m  era5_u10  era5_v10  imerg_precip  ...
624U_29L   2021-07-14T11:21:19Z  -1      288.4     2.1       -0.7      0.0
```

**Daily table:** steps −30 to +30 days. Step 0 is the 24 hours centred on the acquisition (−12 h to +12 h), step −1 the 24 hours before that (−36 h to −12 h), step +1 the 24 hours after (+12 h to +36 h). Steps are relative to the acquisition, not calendar days: a calendar day would begin at a different local time in each time zone.

Step 0 is the closest value in time in both tables. It is not a measurement at the image's pixels: ERA5, ERA5-Land and CAMS are reanalyses (models fitted to observations), and every value is a mean over a 10 km cell from sources of 0.1° to 0.75°.

```
grid_cell  acquired              step_d  era5land_t2m_mean  era5land_t2m_min  era5land_tp_sum  cpc_precip  s5p_no2  ...
```

- **Columns carry a source prefix** (`era5_`, `era5land_`, `cams_`, `s5p_`, `cpc_`, `imerg_`), so mixed sources and resolutions stay explicit. Each source adds a `<source>_valid` column.
- **Daily statistics per variable:** mean for states (temperature, soil water), sum for amounts (precipitation), plus min and max for temperature.
- **Sources with their own days:** NOAA CPC's daily values are fixed periods. A step takes the CPC day that holds most of its 24 hours. A Sentinel-5P overpass goes to the step that contains it.
- Files split by acquisition year.

**Size:** about 2 million acquisitions (estimate) × 61 daily steps × ~25 columns ≈ 3 billion values. The hourly table: 49 steps × ~15 columns ≈ 1.5 billion. Together roughly 18 GB as float32 before compression.

## 6. The tool: for Core and for any expansion

Anyone with a metadata table of `grid_cell` and a timestamp per sample can build the same series:

```python
from majortom.climate import series, monthly, climatology

series(samples,                          # DataFrame or parquet path: grid_cell, timestamp
       sources=["era5", "era5-land", "imerg", "cpc", "cams", "s5p"],
       hourly=24, daily=30,               # steps either side of each acquisition
       out="Weather/")                    # hourly/ and daily/ tables, as in section 5

monthly(source="era5-land", years=(1950, 2025), out="ERA5Land-Monthly/series/")
climatology("ERA5Land-Monthly/series/", period=(1991, 2020), out="ERA5Land-Monthly/climatology/")
```

```
python -m majortom.climate series --samples metadata.parquet --sources era5,era5-land --hourly 24 --daily 30 --out Weather/
```

**Loop over time, not samples.** Many samples share a day. Core-S2L2A's 2,245,886 samples fall on 2,883 distinct days (2016-01-01 to 2024-02-06), a median of 919 a day (counted by `docs/evidence/s5p_cost.py`). For each time step, the build reads the global field once, then applies the weights from section 3 to the cells that need that step:

```
needed steps  = union over acquisitions of their windows      (deduplicated)
for each step: read global field once  ->  weights @ field  ->  rows for every acquisition whose window holds it
```

Planned layout, following `majortom.build`:

```
majortom/climate/
  __init__.py      series(), monthly(), climatology()
  sources.py       per source: access, variables, units, time steps (like build/collections.py)
  weights.py       cell outline x source grid -> sparse weights
  __main__.py      command line
```

## 7. Build and cost

- **Reads dominate.** One global ERA5 field (1440 × 721) is about 4 MB per variable and hour. The hourly and daily windows of Core cover about 2,600 days ≈ 62,000 hours: about 2.5 TB for 10 variables from ARCO-ERA5, streamed, not stored.
- **ERA5-Land is daily only.** Hourly fields are about 6× larger than ERA5's (3600 × 1801); the Climate Data Store's daily statistics would avoid reading them, if they cover our variables (section 2).
- **CAMS** is small: 0.75°, 3-hourly, from the Atmosphere Data Store.
- **Sentinel-5P:** Earth Engine builds each day's mosaic and serves it on a 0.05° grid (32 tiles of 900 × 900 for the globe); cell means are computed on `cirrus` with the section 3 weights, and the grids are dropped. Measured for NO₂ on the busiest day (`docs/evidence/s5p_cost.py`, 2026-10-23): ~1.5 EECU-s per tile, ~50 per day, so ~200 EECU-hours for 2,110 days × 7 products (good to about ×2: one tile, one day, one product). Cell means on Earth Engine (`reduceRegions`) cost ~12,300 EECU-s per day for that day's 75,843 cells, ~50,000 EECU-hours in all, and were rejected. Runs in the dedicated project `major-tom-climate`, not `wiki-eo`.
- **Runs on `cirrus` as Slurm CPU jobs,** one job per source and year. Weight matrices are built once per source grid and saved.
- Otherwise Earth Engine stays for small tests only (shared `wiki-eo` quota).

## 8. Decisions

- Names: `Core-Weather` and `<Source>-Monthly` (e.g. `ERA5Land-Monthly`, `ERA5-Monthly`, `CPC-Monthly`). Confirmed (Miko, 2026-10-23).
- First variables: 2 m temperature, dewpoint, total precipitation, snowfall and snow cover, soil water layer 1, 10 m wind (u, v), solar radiation, surface pressure. Confirmed (Miko, 2026-10-23).
- Air quality: CAMS and Sentinel-5P L3 (Earth Engine) as two sources; Sentinel-5P L2 from CDSE rejected (cost, section 2).
- Monthly coverage: whole globe where the source has sea values (ERA5 ~210 GB); land plus coast otherwise.
- ERA5-Land: daily table only.
- Step 0 is centred on the acquisition, in both tables.

## 9. Open questions

1. Earth Engine tier for `major-tom-climate`: Community (150 EECU-hours a month, ~1.5 months for Sentinel-5P) or Contributor (1,000 a month, needs a billing account; noncommercial use is not charged).
2. The four checks in section 2, before the first build.
