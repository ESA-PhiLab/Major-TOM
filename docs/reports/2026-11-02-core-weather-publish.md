> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-11-02 — Core-Weather published to source.coop

Core-Weather uploaded to `source.coop://major-tom/core-weather` without the ERA5-Land fill south of 57.1°S. The fill comes later as an update of the affected day files.

## Validation (job 2236)

`docs/evidence/core_weather_validate.py` compared `merged/` with all 19 source folders. Output: `/data/Major-TOM/Core-Weather/tests/validate-merged-20261102/`.

- 2,976 days in both tables. Rows per day equal to every source.
- Every column's min, max and null share equal between source and merged. `cpc_start` and `s5p_start` have no statistics on either side.

## Upload (job 2240, 4 h 01 min)

Script: `core_weather_upload.py` in the session scratchpad. It copies `era5/acquisitions.parquet` into `merged/`, stages the README without review notes, then calls `majortom.release.publish`:

    publish(M, "source.coop://major-tom/core-weather", workers=8,
            include=["README.md", "acquisitions.parquet",
                     "hourly/year=*/*.parquet", "daily/year=*/*.parquet"])

Result: `Core-Weather 5954 uploaded, 0 already there, verified True`. Six daily files (2019-03-10, 2019-06-03, 2019-10-28, 2020-02-22, 2021-01-14, 2022-08-18) failed once with `S3UploadFailedError` and passed on retry.

| Table | Files | Rows | Columns | Size |
|:--|--:|--:|--:|--:|
| hourly | 2,976 | 181,496,637 | 285 | 137.4 GB |
| daily | 2,976 | 225,944,793 | 309 | 257.8 GB |
| acquisitions.parquet | 1 | 5,961,727 | | 76,718,357 B |

## README example over https

The job ran the README Reading example against `https://data.source.coop/major-tom/core-weather`. Acquisition: Core-S2L2A, cell 624U_29L (Loch Lomond), 2022-12-11 11:45:01 UTC.

    (49, 4) (61, 5)
    step_h 0: era5_t2m 274.016  era5land_t2m 273.580  imerg_precip 0.0
    step_d 0: era5land_t2m_mean 271.710  cpc_precip 0.1298  cams_pm2p5_mean 2.78e-09
    nulls: s5p_no2 32 of 61, all other columns 0

Same rows and values as the local run.

## Cleanup

Deleted from `/data/Major-TOM/Core-Weather/cache`: cams, cpc, era5, imerg, s5p, era5land/blocks, era5land/monthly. `/data` went from 2.2 TB to 4.0 TB free.

Kept `cache/era5land/hourly` (8.5 GB): the running 2016 south-fill jobs (2229-2234) write their CDS downloads there. `empty.json`, `means` and `raw` are kept too (under 1 MB).

Per-source build folders (`era5-g*`, `era5land-edh-g*`, `cams`, `cpc`, `imerg`, `s5p`) are kept until a decision.

## South fill

Fill job 2235 was held during the upload so it could not change `merged/`. Released after the upload passed. When it finishes, the 2016 day files it changed need re-uploading, and 2018-2024 are submitted with `docs/evidence/era5land_south.sh <year> --submit`.

## Decisions

- Published without the south fill: the fill takes weeks. Rejected: waiting for it.
- Kept the CDS download cache: deleting it would restart the running fill jobs.

## Look at first

- docs/datasets/Core-Weather.md (Reading section, daily column names, Copernicus disclaimer: not yet reviewed)
- docs/evidence/core_weather_validate.py
- docs/evidence/era5land_south.sh
