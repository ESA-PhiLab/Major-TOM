> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-11-09 — Building Major TOM Climate and Core-Weather: time, credits, bottlenecks

Cost of building `major-tom/climate` (six monthly datasets) and `major-tom/core-weather` (weather around every Core
acquisition), measured from the builds of 2026-10-23 to 2026-11-09. Per source: route, limits, measured rates, cost per
TB, and the cost of pulling the source's whole catalogue at those rates.

Conventions:

- **Measured** numbers cite a file and line, or a Slurm job. **Estimated** numbers are derived from measured rates by
  `docs/evidence/climate_costs.py` and say so.
- TB = 10^12 bytes. "Per TB" means per TB of source data read (global float32 fields unless stated), not per TB published.
- Money: Earth Engine at the pay-as-you-go list price, $0.40 per EECU-hour (first 10,000 h a month)
  ([pricing](https://cloud.google.com/earth-engine/pricing)). Our noncommercial use cost nothing; dollar figures show
  what the same work costs a commercial user. Compute at $0.0446 per vCPU-hour (AWS c7i on-demand, us-east-1, from
  [holori](https://calculator.holori.com/aws/ec2/c7i.xlarge/us-east-1); not checked against AWS's own list).
- Machine: `cirrus`, a shared Slurm node (96 usable cores per task, NVMe RAID). All builds ran as CPU jobs.

## What was built

| Dataset | Source and route | Period | Cells | Variables | Published |
|---|---|---|--:|--:|--:|
| CPC-Monthly | NOAA CPC, NOAA PSL files | 1979–2025 | 1,657,095 | 4 | 21.2 GB |
| IMERG-Monthly | NASA IMERG V07, Earth Engine | 1998–2024 | 5,113,745 | 5 | 54.4 GB |
| CAMS-Monthly | CAMS EAC4, Atmosphere Data Store | 2003–2025 | 5,113,745 | 19 | 110.9 GB |
| ERA5-Static | ERA5, ARCO-ERA5 | — | 5,113,745 | 16 | 0.12 GB |
| ERA5Land-Monthly | ERA5-Land, Climate Data Store | 1950–2025 | 1,524,009 | 49 | 469.8 GB |
| ERA5-Monthly | ERA5, Climate Data Store | 1940–2025 | 5,113,745 | 201 | 5.17 TB |
| Core-Weather | ERA5 (ARCO), ERA5-Land (Earth Data Hub), CAMS, CPC, IMERG, Sentinel-5P (Earth Engine) | 2015–2024 | 3,704,013 acquisitions | 285 hourly / 309 daily columns | 395.2 GB |

Sources: `docs/datasets/climate/README.md:18-23`, `docs/reports/2026-11-02-core-weather-publish.md:26-28`.

- Calendar time: first build job 2026-10-23 (Slurm 1608); Core-Weather published 2026-11-02; ERA5-Monthly `by_cell/` 2026-11-04.
- Published in total: 5.83 TB (climate) + 0.40 TB (Core-Weather).
- Peak disk on `/data`: 6.4 TB climate + 3.89 TB Core-Weather on 2026-10-31; the RAID was 98% full
  (`docs/reports/2026-10-31-bycell-page-index.md:11-12`).
- Open: the ERA5-Land fill south of 57.1°S for Core-Weather (2016 and 2018 done, 2019–2020 running, 2021–2024 queued).

## Slurm totals (2026-10-23 to 2026-11-09)

| Jobs | Count | Wall time | CPU-hours allocated |
|---|--:|--:|--:|
| Completed | 181 | 995 h | 5,433 |
| Running (south fill) | 11 | 960 h | 3,842 |
| Failed | 38 | 165 h | 944 |
| Cancelled | 38 | 165 h | 1,184 |
| Out of memory | 3 | 9 h | 55 |
| **All** | **271** | **2,295 h** | **11,458** |

- 11,458 allocated CPU-hours ≈ $511 of on-demand compute.
- Failed, cancelled and out-of-memory jobs: 21% of allocated CPU-hours.
- **The builds wait on remote services, not on CPU.** CPU actually used, as a share of CPU allocated, finished jobs:

| Build | Wall | Allocated | Used | Share |
|---|--:|--:|--:|--:|
| Core-Weather ERA5 (8 groups) | 324 h | 2,594 CPU-h | 141 CPU-h | 5% |
| ERA5-Land south fill (2016, 2018, part of 2019) | 462 h | 1,848 CPU-h | 50 CPU-h | 3% |
| Core-Weather ERA5-Land, Earth Data Hub | 90 h | 721 CPU-h | 169 CPU-h | 23% |
| ERA5Land-Monthly (CDS) | 109 h | 435 CPU-h | 7 CPU-h | 2% |
| ERA5-Monthly (CDS) | 78 h | 312 CPU-h | 34 CPU-h | 11% |
| Core-Weather CAMS | 31 h | 245 CPU-h | 31 CPU-h | 13% |
| ERA5 `by_cell` repack (local) | 13 h | 104 CPU-h | 59 CPU-h | 57% |
| ERA5-Monthly release (local) | 3 h | 27 CPU-h | 22 CPU-h | 83% |

## Per source

### ERA5 hourly — ARCO-ERA5 (Google Cloud Zarr, anonymous)

- Access: no account, no quota, no egress charge to the reader.
- Measured: 882,144 fields in 10 h 45 min, job 1661 (`docs/reports/2026-10-24-climate-overnight.md:9`).
  22.8 fields/s average; link-bound at ~64 MB/s (`LOG.md:88`); 32 threads as fast as 64 (`LOG.md:89`).
- Per TB of float32 fields (4.15 MB each): **2.9 h**, 8 CPUs allocated.
- Incidents: reads stalled with no timeout (gcsfs `timeout=None`, `LOG.md:51`); groups hung for 15 h (2026-10-28) and
  were restarted (`LOG.md:57`).
- Whole catalogue (estimated): 262 hourly single-level variables × 758,232 h (1940 to 2026-06) = 198.7 million fields,
  **825 TB, ~101 days** at the measured rate. One variable: 3.15 TB, ~9 h.

### ERA5 and ERA5-Land monthly — Climate Data Store (CDS)

- Access: free personal token; licence accepted once per dataset.
- ERA5-Monthly: 1,032 months × 201 variables; download job 43.1 h, after a 27.1 h run killed by the 2026-10-25 environment rebuild.
- ERA5Land-Monthly: 912 months × 49 variables; download job 87.8 h; "CDS downloads keep breaking (hundreds of
  retries)" (`docs/reports/2026-10-30-climate-evening.md:64`), all recovered by retries.
- CDS daily statistics: one request at a time per user; a month request waited 2 h 49 min, then ran 16 min
  (`docs/reports/2026-10-23-climate-sources.md:48`). The full design needed ~7,000 requests ("weeks"); cut to ~800
  (`docs/CLIMATE.md:256`).

### ERA5-Land hourly — three routes tried

| Route | Limit | Measured | Per TB (estimated) | Whole catalogue: 49–50 variables, 1950–2026 (estimated) |
|---|---|---|---|---|
| Earth Engine (stored images) | Contributor 1,000 EECU-h/month | 186 EECU-s per global hour, 11 bands (`docs/CLIMATE.md:244`) | 181 EECU-h = **$72** | 33.0 M band-hours, 855 TB: **154,839 EECU-h** = 155 Contributor-months or 1.5 Partner-months; **~$62,000** at $0.40 (less with volume tiers) |
| Earth Data Hub (Destination Earth zarr) | 500,000 requests/month per key (`docs/CLIMATE.md:193`) | group 1: ~330,000 requests in 22.7 h (job 1808), 4.0 requests/s | 500,000 requests (at ~2 MB each) = **1 key-month**, ~34 h | 25.3 M requests before skipping sea chunks: **51 key-months**, ~51 TB compressed, ~72 days at 4 requests/s |
| CDS hourly (south fill) | one queue per user; 1,848 fields per request accepted, 8,184 refused (`docs/CLIMATE.md:212`) | 2016: 6 groups in 7.8–16.5 h; 2019–2020 together: ~520 group-days/day | **14 days**, ~71 GB/day | ~126 TB of GRIB: **~4.9 years** at the present pace; 18,203 requests |

- Earth Engine was used for 2016 to 2017-11, then the quota ran out (2026-10-25 10:56 UTC, ~80% spent on ERA5-Land,
  `docs/CLIMATE.md:244`). Earth Engine's profiler had under-reported the cost of stored-image reads ~30×.
- Earth Data Hub: the store stops at 57.1°S (~18% of grid rows), so cells further south need the CDS fill
  (`docs/CLIMATE.md:210-212`). Values are bit-rounded to 13 mantissa bits (~1 part in 16,000).
- Earth Data Hub keys: one free key per person; extra keys from colleagues ran the six variable groups as two chains
  (from 2026-10-28). Group jobs: 4.5–22.7 h each; group 4 failed after 19.6 h on 401/403/429 errors and was restarted.
- CDS south fill, measured from the day files (2026-11-06 to 2026-11-09): 508–565 group-days a day in total, shared
  between the two years running. When one year speeds up, the other slows, so the limit is the CDS per user, not the
  number of jobs. Remaining at that pace: 2019–2020 ~1.8 days; 2021–2024 ~2.8 days per year, ~2026-11-22 for all.
- CDS errors stop a job: 2019 group 5 failed after 36.5 h on "400 Bad Request … The job has failed" and was resubmitted
  (job 2737); `stream` does not resubmit a failed CDS request.

### IMERG — Earth Engine (GES DISC rejected)

- GES DISC: 0.09 MB/s with one connection; parallel connections reset by the server
  (`docs/reports/2026-10-24-climate-overnight.md:32`).
- Earth Engine `getPixels`: 12.9 MB/s with 8 requests at once (1.24 GB per day of 48 half-hours,
  `docs/reports/2026-10-24-climate-overnight.md:33`), 140× faster.
- Cost: 9.7 EECU-s per hourly field, 4.85 per stored half-hour (calibrated against the quota counter, `docs/CLIMATE.md:262`).
- Per TB (estimated): **52 EECU-h ($21), 21.5 h**.
- Whole catalogue 1998-06 to 2025 (estimated): 483,648 half-hours, 12.5 TB, **652 EECU-h** (0.7 Contributor-months,
  $261), **11 days**. Through GES DISC at 0.09 MB/s: ~1,600 days.
- Built on a second Earth Engine project (`precipitation-study`) after the first ran out of quota.

### CAMS EAC4 — Atmosphere Data Store (ADS)

- Access: free token (as CDS).
- Measured: 0.3 MB/s with one month in flight, 0.62 MB/s with three (`docs/reports/2026-10-24-climate-overnight.md:47`).
- Per TB (estimated): **18.7 days**.
- Our 19-variable selection, 2003–2025 (estimated): 276 months × 1.25 GB = 345 GB, **6.4 days**. Core-Weather's
  ~100 months: ~2.3 days measured-rate estimate; the fetch job ran 28.1 h.

### NOAA CPC — NOAA PSL files

- Anonymous yearly files; 29 GB cached; no limits met. CPC-Monthly build 3.8 h.

### Sentinel-5P L3 — Earth Engine

- Earth Engine mosaics on a 0.05° grid (32 tiles of 900 × 900), cell means on `cirrus`: ~50 EECU-s per day and product
  (`docs/CLIMATE.md:245`).
- Cell means on Earth Engine (`reduceRegions`) would cost ~50,000 EECU-h instead of ~200: rejected.
- Level-2 swaths from CDSE: ~18 TB for NO₂ alone over Core's days: rejected (`docs/CLIMATE.md:40`).
- Whole catalogue (estimated): 2,744 days × 7 products, **267 EECU-h ($107)**.

### Publishing — source.coop

| Upload | Size | Time | Rate | Per TB |
|---|--:|--:|--:|--:|
| Core-Weather (job 2240, 8 workers) | 395.2 GB | 4.0 h | 27 MB/s | 10.2 h |
| ERA5-Monthly (job 2106) | 2,800 GB | 18.8 h | 41 MB/s | 6.7 h |
| ERA5-Monthly `by_cell/` (jobs 2315, 2404, 2426) | 2,141 GB | 20.7 h | 29 MB/s | 9.7 h |

- Transient errors: `S3UploadFailedError` on 6 files, source.coop 520s outlasting 6 retries. Fixed with 10 retries per
  file and waits capped at 10 min (`LOG.md:20`).

## Summary: per TB and whole catalogue

| Source, route | Time per TB | Credits per TB | $ per TB (list) | Whole catalogue (estimated) |
|---|--:|--:|--:|---|
| ERA5 hourly, ARCO-ERA5 | 2.9 h | none | compute only (~$1) | 825 TB, ~101 days |
| ERA5-Land hourly, Earth Engine | quota-bound | 181 EECU-h | $72 | 155 Contributor-months, ~$62,000 |
| ERA5-Land hourly, Earth Data Hub | ~34 h | 1 key-month | free | 51 key-months, ~72 days of reading |
| ERA5-Land hourly, CDS | ~14 days | none | free | ~4.9 years |
| IMERG, Earth Engine | 21.5 h | 52 EECU-h | $21 | 652 EECU-h, 11 days |
| CAMS EAC4, ADS | ~18.7 days | none | free | our selection: 6.4 days |
| Sentinel-5P L3, Earth Engine | — | — | — | 267 EECU-h, $107 |
| Upload to source.coop | 6.7–10.2 h | — | — | — |

## What made it feasible

- Loop over time steps, not samples: each global field read once (2,245,886 Core-S2L2A samples fall on 2,883 days).
- Weights computed once per source grid (sparse cell × pixel table); each step is one `numpy.bincount`.
- IMERG through Earth Engine instead of GES DISC: 140× faster.
- Sentinel-5P as Earth Engine mosaics plus local cell means: ~200 instead of ~50,000 EECU-h; L3 instead of ~18 TB of L2.
- ERA5-Land: daily means and 3-hour offset groups cut CDS requests from ~7,000 to ~800; then Earth Data Hub for hourly
  data after Earth Engine cost ~30× its profiler estimate.
- Earth Data Hub: variables in six priority groups within the monthly request quota; extra per-person keys; sea chunks
  skipped; finished blocks reused on reruns.
- Second Earth Engine project for IMERG (+1,000 EECU-h a month).
- Memory requests sized from measured peaks (monthly builds 64 → 16 GB; south fill 32 → 8 GB), which stopped the
  builds holding back other users' jobs.
- `by_cell` layout (row groups of 1,024 cells, page index): one ERA5 variable for one cell, 11.4 MB → 0.8 MB per read
  (`docs/reports/2026-10-31-bycell-page-index.md:34-37`). Climatology page index: one cell, all 201 variables,
  51.35 MB → 3.60 MB (`docs/reports/2026-11-02-era5land-climatology-layout.md:53-54`).

## Bottlenecks and incidents

- **Remote services set the pace.** CDS: one request at a time per user, queue waits up to ~3 h. ADS: 0.3–0.62 MB/s.
  GES DISC: 0.09 MB/s. Earth Engine: quota (1,000 EECU-h a month on Contributor) used up in one day on 2026-10-25.
- **Profiler estimates were low.** Earth Engine's profiler under-reported stored-image reads ~30× (ERA5-Land) and ~20×
  (IMERG). Costs are now measured on runs of at least an hour against the quota counter.
- **Hangs without errors:** ARCO reads with no timeout; GDAL netCDF deadlock; ERA5 groups idle for 15 h; 207 stale
  joblib processes in swap (2026-10-24 to 2026-10-30).
- **Environment:** `uv` recreated `.venv` on 2026-10-25 12:53 and killed nine running builds; builds now run on
  `.venv311` by full path.
- **Memory:** the ERA5 repack ran out of memory with 6 bands at once (~30 GB each) and was re-run 2 at a time; the
  ERA5 `by_cell` build hit its 100 GB limit and was rewritten in slices of 4,096 cells (a whole band needed ~270 GB).
- **Disk:** the RAID reached 98% (2026-10-31); a 2.8 TB intermediate build had to be deleted before ERA5 `by_cell` could run.
- **CDS failures stop jobs** (2019 south group 5, 2026-11-06), and two years sharing a cache folder deleted each
  other's files (2019 group 4, 2026-11-05).

## How it scales

- **Linear and cheap:** sources on cloud object storage without quotas (ARCO-ERA5, NOAA PSL): time grows with bytes at
  ~3 h per TB on one node. ERA5 hourly in full (825 TB) is a ~3-month read on this link.
- **Quota-bound:** Earth Engine and Earth Data Hub scale with credits, not machines. More throughput means a higher tier
  (Earth Engine Partner: 100,000 EECU-h a month), more keys, or a paid plan.
- **Queue-bound:** CDS and ADS throughput is per user; more jobs do not help. The ERA5-Land hourly catalogue through the
  CDS would take ~4.9 years at the present pace.
- **Local steps are not the limit:** repacking, climatology and `by_cell` builds took hours, using the CPU fully, once
  memory was bounded.

## Decisions

- ARCO-ERA5 for ERA5 hourly: no quota, fastest route.
- Earth Engine for IMERG and Sentinel-5P L3: rejected GES DISC (0.09 MB/s) and CDSE L2 (~18 TB).
- Earth Data Hub for ERA5-Land hourly: rejected Earth Engine (155 Contributor-months for the catalogue) and CDS (years).
- CDS hourly only for the area Earth Data Hub lacks (south of 57.1°S).

## Look at first

- `docs/CLIMATE.md:237` (build and cost, measured Earth Engine costs)
- `docs/evidence/climate_costs.py` (every estimate in this report)
- `docs/reports/2026-11-02-core-weather-publish.md` (the published Core-Weather)
