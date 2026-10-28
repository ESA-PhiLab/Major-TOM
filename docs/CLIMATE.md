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
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) | land surface: temperature, soil water, snow, runoff, fluxes | 0.1° (~9 km) | hourly and monthly means, 1950– (2–3 months' delay) | Climate Data Store; [Earth Data Hub](https://earthdatahub.destine.eu/collections/era5/datasets/era5-land) zarr (free key, hourly, 57.1°S–90°N); Earth Engine | CC-BY (Copernicus) |
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

Layout as built (2026-10-23), following `majortom.build`:

```
majortom/climate/
  __main__.py      python -m majortom.climate stream --samples <metadata.parquet ...> --source <name> --out <dir>
  sources.py       per source: grid, variables, units, time step, reader, period ("hour" or "day")
  weights.py       cell x source pixel weights; cell means
  series.py        hourly and daily tables in memory (small sets)
  stream.py        hourly sources (ERA5, CAMS, IMERG): rolling buffer, each hour read once
  days.py          daily sources (ERA5-Land, NOAA CPC, Sentinel-5P): per source day, cell means cached once
  monthly.py       [Source]-Monthly: monthly series for every cell, and the climatology from it
  readers/         one module per access method
    zarr.py          ERA5 (ARCO-ERA5)
    queue.py         ERA5-Land daily statistics, CAMS EAC4 (Climate / Atmosphere Data Store, a month per request)
    files.py         NOAA CPC (yearly netCDF), IMERG (half-hourly HDF5)
    earthengine.py   Sentinel-5P (UTC-day mosaics, 0.05° tiles), IMERG (stored half-hours)
    edh.py           ERA5-Land hourly from Earth Data Hub (zarr v3, 60-day chunks; section 6.1)
```

Each source is written to its own folder (`<out>/<source>/hourly/`, `<out>/<source>/daily/`), keyed by
`grid_cell`, `acquired` and the step, so sources join on those three columns. Daily sources add
`<source>_start`, the UTC start of the source day used for each step. Downloads and cached means live under
`--cache` (Core build: `/data/Major-TOM/Core-Weather/cache/`).

### 6.1 ERA5-Land hourly from Earth Data Hub

Source `era5-land-edh`: all 49 time-varying ERA5-Land variables, hourly, from Destination Earth's Earth Data Hub (`https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr`, CC-BY 4.0). Column names are the same as the Earth Engine build's (`era5land_t2m`, ...), so the two join or replace each other.

**Access, once per user:**

1. Create a free account at <https://earthdatahub.destine.eu> (a Destination Earth platform login).
2. Copy the API key from the account settings page.
3. Run `python -m majortom.build.credentials edh` and paste the key at the hidden prompt. It is saved to `~/.config/majortom/edh-key` (owner-only, 600) and checked with one metadata read. In jobs where that file is not readable, set `EDH_KEY` instead; the environment takes precedence.
4. Other people's keys (each has its own quota): `python -m majortom.build.credentials edh --name <person>` saves `edh-key-<person>`; a job uses it with `sbatch --export=ALL,EDH_KEY_NAME=<person> ...`. Earth Data Hub confirmed (2026-10-27) that several users in one organisation may each register a key; no quota increase until Destination Earth's "Tenancy" feature.

The free quota is **500,000 requests a month** (resets on the 1st; shown on the account page). One request is one chunk: 60 days × 5° × 10° of one variable (~1–3 MB compressed). Opening the store costs ~9 requests.

**How the reader works** (`majortom/climate/readers/edh.py`):

```
stream -> use_cells(cells): the chunks the cells overlap, less those known to be open sea (<cache>/empty.json)
       -> field(variable, hour): first hour of a 60-day block -> fetch its chunks (32 requests in flight over all variables, each retried up to 6 times)
            -> decode once into <cache>/blocks/<variable>/<block>.f32 (hour-major, sparse; ~16 GB per variable)
          every hour: one global field from that file; the 2 newest blocks per variable are kept;
          a finished block is marked (<block>.filled.npy), so a rerun reuses it without requests
```

- Memory (2026-10-28): each hour is read straight from its block file (one 26 MB plane) and its pages are dropped from the page cache; a finished block's mapping is closed and dropped too. Before, the blocks stayed mapped and filled the job's memory limit (64 GB), and at 48 GB pushed 6 GB into the server's shared swap. Checked: new and old reads identical on 384 hours of 16 real blocks (`docs/evidence/edh_hour_check.py`, no requests); memory in a full run to be measured on group 3.

- Accumulated variables (19: precipitation, snowfall, evaporation and its parts, runoff, snowmelt, radiation and heat fluxes) are stored as totals since 00 UTC, restarting at 01 UTC; `field()` returns each hour's amount (this hour minus the previous), as the Earth Engine source did.
- Coverage: 90°N to 57.1°S, land only (NaN over the sea). **The gap south of 57.1°S, and how to fill it:**
  - What: Core cells south of 57.1°S (Antarctic coast and peninsula, South Orkney, South Sandwich; 728 cells in 2016-2017 alone) have no ERA5-Land values in Core-Weather. Their other sources (ERA5, CAMS, Sentinel-5P, IMERG) are unaffected; ERA5 (0.25°) carries the main variables there.
  - Why: a choice of Earth Data Hub's copy, not of ERA5-Land. ECMWF publishes ERA5-Land for the whole globe (Earth Engine's copy has values there); the GRIB Earth Data Hub converted already stopped at 57.1°S (1,472 of 1,801 latitude rows; store metadata, 2026-10-27). Their reason is not stated; it drops ~18% of the grid, almost all Antarctica, and keeps all other land (Cape Horn ~56°S).
  - Decision (Miko, 2026-10-27): leave the gap, documented; fill it later from the Climate Data Store if needed.
  - Fill (planned, not built): the Climate Data Store's `reanalysis-era5-land` (hourly, global, free; the queue the monthly build already uses), requests cut to the area south of 57.1°S (`area`: -57, -180, -90, 180), one month per request per variable group. Same accumulation convention (totals since 00 UTC), so the same de-accumulation; same columns (`era5land_*`), so the filled rows join the Earth Data Hub ones. Cells that straddle 57.1°S (partly covered, `era5land_valid` < 1) are recomputed whole from the Climate Data Store. Also worth asking Earth Data Hub whether a full-globe copy exists or is planned.
- The store is lossy: bitround with 13 mantissa bits, ~1 part in 16,000 (±0.02 K at 300 K).
- Validated 2026-10-27 against the Earth Engine build on the 786 acquisition days both wrote (2016 to 2017-11; 3.5M hourly and 4.4M daily rows; `docs/evidence/era5land_edh_validate.py`). All group-1 variables agree within ~4 x 10^-4 relative (t2m <= 0.125 K, sp <= 32 Pa; daily tp and sf sums 4 x 10^-4; hourly tp and sf 3-4 x 10^-3 of the largest hour, from rounding the totals before differencing), except where explained: (1) 35,672 rows of 728 cells south of 57.1°S (-61° to -83°) have Earth Engine values only: outside Earth Data Hub's grid; (2) 101,520 sea and coastal rows have Earth Engine `ssrd` only, with every other field empty on both sides: Earth Engine fills solar radiation over the sea, Earth Data Hub keeps it land-only like the other fields; (3) on a few days (e.g. 2016-03-04 to 03-07, around 17-19 UTC) Earth Engine's derived hourly `ssrd` is wrong: ERA5 `ssrd` from our ERA5 build agrees with Earth Data Hub (e.g. 283D_710L: Earth Data Hub 3.647, ERA5 3.646, Earth Engine 1.178 MJ m-2). The Earth Data Hub build replaces the Earth Engine one.
- Checked 2026-10-26 (`docs/evidence/edh_check.py`, and two acquisitions end to end): agrees with the Earth Engine build within 4 × 10⁻⁴ relative for every group-1 variable (t2m ≤ 0.12 K, sp ≤ 31 Pa, hourly tp ≤ 0.001 mm), identical rows and valid shares; no repeated fields (the fault Earth Data Hub fixed in its monthly ERA5-Land store on 2026-10-23).

**Build in groups** (`ERA5_LAND_GROUPS` in `sources.py`, by priority; ~30,000 requests per variable for the Core period):

| Group | Variables | Requests |
|---|---|---|
| 1 starter | t2m, d2m, u10, v10, sp, tp, sf, ssrd, snowc, sd, swvl1 | ~330k |
| 2 energy and water balance | skt, e, ro, ssr, str, strd, slhf, sshf | ~240k |
| 3 soil profile | swvl2-4, stl1-4 | ~210k |
| 4 evaporation parts, runoff split | pev, evabs, evatc, evavt, evaow, sro, ssro | ~210k |
| 5 snowpack, albedo | sde, rsn, tsn, asn, smlt, es, fal, src | ~240k |
| 6 lakes, leaf area | lblt, licd, lict, lmld, lmlt, ltlt, lai_hv, lai_lv | ~240k |

```
G=$(python -c "from majortom.climate.sources import ERA5_LAND_GROUPS as G; print(' '.join(G[2]))")
sbatch --cpus-per-task=8 --mem=64G -J core-weather-era5land-edh-g2 ~/slurm-templates/cpu.sbatch \
  .venv311/bin/python -u -m majortom.climate stream --samples <Core metadata.parquet ...> --source era5-land-edh \
  --variables $G --out /data/Major-TOM/Core-Weather/era5land-edh-g2 --cache /data/Major-TOM/Core-Weather/cache
```

A job that runs out of quota fails on its next request; a rerun continues from the acquisition days already written. Lake shape factor (`lshf`, fixed in time) belongs in the ERA5-Land static table.

## 7. Build and cost

- **Reads dominate.** One global ERA5 field (1440 × 721) is about 4 MB per variable and hour. The hourly and daily windows of Core cover about 2,600 days ≈ 62,000 hours: about 2.5 TB for 10 variables from ARCO-ERA5, streamed, not stored.
- **ERA5-Land is daily only.** Hourly fields are about 6× larger than ERA5's (3600 × 1801); the Climate Data Store's daily statistics would avoid reading them, if they cover our variables (section 2).
- **CAMS** is small: 0.75°, 3-hourly, from the Atmosphere Data Store.
- **Climate Data Store key** (ERA5-Land; CAMS): a personal access token, saved once with `python -m majortom.build.credentials cds` (owner-only file, `CDS_TOKEN` takes precedence); each dataset's licence is accepted once on its page.
- **Earth Data Hub key** (ERA5-Land hourly as zarr, from Destination Earth): a free account at <https://earthdatahub.destine.eu> gives an API key (account settings). Save it once with `python -m majortom.build.credentials edh`: hidden prompt, owner-only file `~/.config/majortom/edh-key`, then one read of the store's metadata (no data). `EDH_KEY` takes precedence, e.g. in Slurm jobs. Readers send it as HTTP basic auth with the user name `edh`, the same as Earth Data Hub's `https://edh:<key>@data.earthdatahub.destine.eu/...` form, so the key never appears in a logged URL. The store, `https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr` (zarr v3, needs `zarr>=3`), has 50 variables (our 49 time-varying ones and lake shape factor), hourly from 1950 to the last closed month, on the 0.1° grid from 57.1°S to 90°N (most of Antarctica is cut). Chunks are 60 days × 5° × 10°, so reading one hour fetches 60 days: readers go through time in 60-day blocks. No read quota; the network link is the limit.
- **Earth Engine costs, measured in production (2026-10-25)** from the project's in-progress EECU-seconds (Cloud Monitoring, 5-minute steps) against the Slurm job times: ERA5-Land hourly (11 bands) ~186 EECU-s per global hour (~7.8 per 600 × 600 tile), IMERG ~6.5 EECU-s per stored half-hour, Sentinel-5P ~45–60 per day and product. Earth Engine's profiler (`ee.profilePrinting`), used for the earlier estimates, matched the computed Sentinel-5P mosaics but under-reported reads of stored images by ~20× (IMERG) and ~30× (ERA5-Land). The Contributor quota (1,000 EECU-hours a month) ran out on 2026-10-25 at 10:56 UTC, about 80% of it spent by ERA5-Land hourly. Rule from now on: a cost is measured on a run of at least an hour, against the quota counter, before any budget.
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
- NOAA CPC day: CPC's daily totals end at a different UTC hour per country, published as a 0.5° end-of-day map (CPC's README: "based on very limited and UNCHECKED information"; 12Z for 23% of land boxes, 18Z 16%, 24Z 38%, 03Z/06Z the next day 22%). The reader uses that map per cell (`majortom/climate/readers/files.py`), and `cpc_start` records each day's UTC start. Our ERA5 comparison at Sydney, Brisbane, Denver and Glasgow (`docs/evidence/cpc_day.py`) agrees with the map within its broad peaks.
- IMERG source (2026-10-24, pending review): read from Earth Engine's copy of IMERG V07 (`NASA/GPM_L3/IMERG_V07`, stored half-hours read with `getPixels`), not GES DISC: GES DISC gave 0.09 MB/s from cirrus and reset parallel connections; Earth Engine gave 12.9 MB/s with 8 requests at once, ~0.33 EECU-s per global half-hour by the profiler (~13 EECU-hours for the Core period, ~3.7 TB moved); measured in production ~6.5 EECU-s (section 7), ~130 EECU-hours for the Core period. The GES DISC reader stays as `imerg-gesdisc`.
- ERA5-Land in Core-Weather (2026-10-24): hourly from Earth Engine's stored images, compressed GeoTIFF. It has 49 distinct time-varying variables (69 bands, 19 of them cumulative twins of the hourly amounts, plus lake shape factor, fixed in time). The first 11 (the starter list) are building now (~120 EECU-hours); the other 38 follow from 1 November on that month's quota (~395 EECU-hours; measured ~1.05 EECU-s per 600 x 600 px tile area for all 49 bands, 0.25 for 11). Decision: Miko, option 2. Superseded 2026-10-25: those costs came from the profiler and were ~30× low (section 7); the build stopped with the quota at 2017-11 (resumable). Proposed: ERA5-Land hourly, all 49 variables, from Earth Data Hub's zarr instead, after a throughput test; Earth Engine keeps IMERG.
- ERA5-Land hourly from Earth Data Hub (Miko, 2026-10-26): all 49 variables in six groups of decreasing priority (section 6.1), within the 500,000-request monthly quota: group 1 in October 2026, 2-3 in November, 4-5 in December, 6 in January. Replaces the Earth Engine source, which stays as a reference for 2016-2017.
- Output layout (Miko, 2026-10-23): each source is built into its own folder under `/data/Major-TOM/Core-Weather/` and joined on `grid_cell`, `acquired` and the step; for publishing, the sources are merged into one hourly and one daily table.
- ERA5-Land through Earth Engine (option b, measured 2026-10-23, `docs/evidence/era5land_ee_cost.py`): ~280 EECU-s per global 7-band daily field, so ~2,100 EECU-hours for ~3,000 days × 9 offset groups (~700 if cut to each group's area): months of the Community tier. Through the Climate Data Store, month requests waited up to ~3 h and ran ~16 min. Decided (Miko, 2026-10-23): option c, ERA5-Land only in ERA5Land-Monthly; Core-Weather takes its daily land variables from ERA5. The ERA5-Land daily reader (`majortom/climate/readers/queue.py`) stays, tested on Loch Lomond.

## 9. Open questions

1. Earth Engine tier for `major-tom-climate`: Contributor since 2026-10-24 (1,000 EECU-hours a month); used up on 2026-10-25 (section 7), restricted until 1 November. Remaining Earth Engine work: IMERG from 2016-09 (~120 EECU-hours at the measured cost).

## 10. Future options

- (Costs in this item and the next came from Earth Engine's profiler and are ~30× low for stored images; section 7.) ERA5-Land in Core-Weather from raw hours (measured 2026-10-23, suggested by a colleague's `getPixels` tip): reading one stored hourly image costs ~0.21 EECU-s per 600 x 600 seven-band tile with `getPixels` or `computePixels` alike; the expense of option b is the server-side 24-hour mean. Reading raw hours (18 tiles per global hour) and averaging on cirrus: ~3.8 EECU-s per hour, ~76 EECU-hours for ~3,000 days, one read serving every offset group, and it also allows an hourly ERA5-Land table. Cost moves to transfer: ~26 TB as stored (float64), about half as float32, less again without all-sea tiles (~1.5-3 days at ~64 MB/s, if Earth Engine serves that fast).
- ERA5-Land in Core-Weather (option b): Earth Engine's hourly ERA5-Land averaged over each offset group's day, ~280 EECU-s per global 7-band field, ~2,100 EECU-hours for Core (~700 if cut to each group's area; `docs/evidence/era5land_ee_cost.py`). Worth it with a Contributor or Partner tier, or if the Climate Data Store queue shortens (option a: ~800 month requests, measured waits up to ~3 h and ~16 min runs).
