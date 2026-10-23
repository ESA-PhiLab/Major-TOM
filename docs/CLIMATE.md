> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** reviewed (Miko, 2026-10-23), sources included. The section 2 checks and section 9 question 2 were added after review (unreviewed).

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

Checked before the first build (2026-10-23):
- **Sentinel-5P terms:** the [Copernicus Sentinel data legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) allows reproduction, distribution, communication to the public, and adaptation and combination with other data. Cell means are adapted data, so the dataset card must carry "Contains modified Copernicus Sentinel data [Year]". CAMS, ERA5 and ERA5-Land (Copernicus services, CC-BY) need "Contains modified Copernicus Service information [Year]" (to confirm against each CC-BY page).
- **Sentinel-5P L3 on Earth Engine:** 1113.2 m pixels, 2018-06-28 to present; pixels below a quality value of 75% are removed for tropospheric NO₂ ([catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2); its example command uses 50%, so the page is not fully consistent).
- **NOAA CPC precipitation:** [CPC Global Unified Gauge-Based Analysis of Daily Precipitation](https://developers.google.com/earth-engine/datasets/catalog/NOAA_CPC_Precipitation), 0.5°, daily, 1979 to present (one day's delay), 0.1 mm units, with a gauge count per pixel; quality weakest over tropical Africa and Antarctica; real-time values can be revised. Earth Engine's copy starts in 2006; the full record is at NOAA PSL (page not read directly). Use unrestricted, attribution requested.
- **ERA5-Land daily statistics:** the Climate Data Store's [post-processed daily statistics](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics) (0.1°, 1950 to present, CC-BY) give daily mean, min and max of states (2 m temperature and dewpoint, 10 m wind, surface pressure, snow cover, soil water), but **omit accumulated variables** (precipitation, and by the same rule snowfall and radiation), and compute **calendar days** at a fixed UTC offset, not windows centred on an acquisition. They do not fit the daily table as designed: see section 9.
- **ARCO-ERA5:** Zarr format 2; final data 1940-01-01 to 2026-06-30, preliminary ERA5T to 2026-10-17 (store metadata, read 2026-10-23). The time axis runs 1900-2050 with empty slots outside that range.

Two air-quality sources, deliberately: CAMS has no gaps and covers all of Core (2017–), but at ~80 km and from a model; Sentinel-5P observes at a few km, but only from mid-2018 and with gaps. Sentinel-5P L2 swaths from CDSE were rejected: whole files must be downloaded (about 0.5 GB per orbit for NO₂, ~14 orbits a day), on the order of 18 TB for NO₂ alone over Core's days, for values averaged over 10 km cells anyway.

**Which table each source feeds:**

| Source | Hourly table | Daily table |
|---|---|---|
| ERA5 | yes | yes |
| ERA5-Land | no | yes: states only, days centred by UTC offset (section 8) |
| IMERG | yes (half-hours summed to hours) | yes |
| CAMS | yes, every third hour (others empty) | yes |
| NOAA CPC | no | yes |
| Sentinel-5P | no | yes |

ERA5-Land stays out of the hourly table: its atmosphere comes from ERA5 (it is a land-surface rerun at 9 km, with temperature adjusted for elevation), so hourly values add spatial detail but no new weather dynamics, and its slow variables (soil water, snow) barely change within 48 hours. Hourly ERA5-Land would add only ~3 GB, but ~13 TB of reads through the Climate Data Store queue.

## 3. From a source grid to a cell value

```
source field (one time step, lat/lon grid)
  -> overlap of each source pixel with each cell: in longitude x in sin(latitude), as weights
     (Major TOM cells and source pixels are both lat/lon boxes, so the exact area overlap separates;
      computed once per source grid: majortom/climate/weights.py)
  -> weighted mean over valid pixels                         -> value
  -> share of the cell's area with valid data                -> valid
```

- **Weights are precomputed** once per source grid as a sparse matrix (cells × source pixels), kept as (cell, pixel, weight) arrays. Each time step is then one weighted sum (`numpy.bincount`) over the whole globe; no scipy needed.
- **Cells smaller than a source pixel** (ERA5 at 31 km against 10 km cells) get the one or two overlapping pixels. Values are not interpolated: a cell value never claims more detail than the source has.
- **Valid share:** ERA5-Land has no values over the sea, Sentinel-5P has gaps under clouds and between swaths. A coastal cell's value comes from its land part, and `valid` says how much that is. ERA5 stores land-only variables as ~0 over the sea instead of leaving gaps, so for those a land mask decides which pixels count (section 8).

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
- **Climate Data Store key** (ERA5-Land; CAMS): a personal access token, saved once with `python -m majortom.build.credentials cds` (owner-only file, `CDS_TOKEN` takes precedence); each dataset's licence is accepted once on its page.
- **Sentinel-5P:** Earth Engine builds each day's mosaic and serves it on a 0.05° grid (32 tiles of 900 × 900 for the globe); cell means are computed on `cirrus` with the section 3 weights, and the grids are dropped. Measured for NO₂ on the busiest day (`docs/evidence/s5p_cost.py`, 2026-10-23): ~1.5 EECU-s per tile, ~50 per day, so ~200 EECU-hours for 2,110 days × 7 products (good to about ×2: one tile, one day, one product). Cell means on Earth Engine (`reduceRegions`) cost ~12,300 EECU-s per day for that day's 75,843 cells, ~50,000 EECU-hours in all, and were rejected. Runs in the dedicated project `major-tom-climate`, not `wiki-eo`.
- **Runs on `cirrus` as Slurm CPU jobs,** one job per source and year. Weight matrices are built once per source grid and saved.
- Otherwise Earth Engine stays for small tests only (shared `wiki-eo` quota).

## 8. Decisions

- Names: `Core-Weather` and `<Source>-Monthly` (e.g. `ERA5Land-Monthly`, `ERA5-Monthly`, `CPC-Monthly`). Confirmed (Miko, 2026-10-23).
- First variables: 2 m temperature, dewpoint, total precipitation, snowfall and snow cover, soil water layer 1, 10 m wind (u, v), solar radiation, surface pressure. Confirmed (Miko, 2026-10-23). Sea surface temperature and sea ice cover added for ERA5 (2026-10-23): Core-S2L2A includes open-sea cells (34% of July 2021 acquisitions have no land pixel); ERA5 stores it as NaN over land, so a cell's value covers its sea part.
- Air quality: CAMS and Sentinel-5P L3 (Earth Engine) as two sources; Sentinel-5P L2 from CDSE rejected (cost, section 2).
- Monthly coverage: whole globe where the source has sea values (ERA5 ~210 GB); land plus coast otherwise.
- ERA5-Land: daily table only, states only (2 m temperature and dewpoint, 10 m wind, surface pressure, snow cover, soil water layer 1), from the Climate Data Store's daily statistics. Its days start at a chosen whole-hour UTC offset, so each acquisition gets offset = round(12 − its UTC hour): step 0 is then the 24 h centred on the acquisition within ±30 minutes, like the other sources. Acquisitions are grouped by offset and each group is requested only for its own area (for Sentinel-2, about 10:30 local solar time, a band of ~15° longitude), so the requests together cover the globe about once per sensor timing pattern: ~24 groups instead of one, about the same volume. Blending neighbouring UTC days was rejected: a blended daily mean is biased by the day–night cycle, and minima and maxima cannot be recovered. Precipitation, snowfall and radiation come from ERA5 (and IMERG, CPC). Hourly ERA5-Land (~13 TB of reads) rejected for cost. Confirmed (Miko, 2026-10-23). Verified 2026-10-23: the product takes 27 whole-hour offsets (`utc-12:00` to `utc+14:00`) and an area box, with one statistic and one month per request; an offset day starts at local midnight (`utc+12:00` day d = 12:00 UTC on d−1 to 11:00 UTC on d). Over April 2021 at 624U_29L, ERA5-Land minus the ERA5 window mean varies by 0.59–0.63 K under that reading for offsets 0, +12 and −6, against 0.76–1.46 K for the opposite sign or no shift (`docs/evidence/era5land_offset.py`). Month-long requests took 3–8 minutes in the queue.
- ERA5-Land request count (2026-10-23): this product runs one request at a time per user (10 tiny requests: 1 running at once, ~43 s each in sequence), so the ~7,000 requests of the full design would take weeks. Two cuts, confirmed (Miko, 2026-10-23): daily means only (minimum and maximum of 2 m temperature come from ERA5's hourly table), and UTC offsets in 3-hour steps (8 groups; days centred within ±1.5 h). About 800 requests, 2–4 days in sequence.
- Land-only variables (ERA5 soil water and snow depth; ERA5 stores ~0 over the sea): averaged over land pixels only, where a pixel is land if its land fraction is at least 0.5. ARCO-ERA5's regridding blends coastal pixels with sea (a pixel 40% land holds soil water diluted towards 0), so fractional weighting would bias coastal cells low. A cell with no land pixel gets no value; the column `era5_land` gives each cell's share in land pixels. Confirmed (Miko, 2026-10-23).
- Values are kept as published, including ERA5's tiny negative accumulations (about −2 × 10⁻⁹ m). Confirmed (Miko, 2026-10-23).
- Step 0 is centred on the acquisition, in both tables.

## 9. Open questions

1. Earth Engine tier for `major-tom-climate`: Community for now (Miko, 2026-10-23): 150 EECU-hours a month, so the full Sentinel-5P build (~200 EECU-hours) spans about two months, or runs on in restricted mode. Contributor (1,000 a month, needs a billing account) remains the option to finish within a month.
