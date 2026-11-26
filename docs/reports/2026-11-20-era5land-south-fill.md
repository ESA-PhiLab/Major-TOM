> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-11-20 — ERA5-Land south of 57.1°S filled into Core-Weather

Core-Weather was published (2026-11-02) without ERA5-Land values south of 57.1°S: Earth Data Hub's copy stops there.
The values were fetched from the Climate Data Store's hourly `reanalysis-era5-land` and written into the merged day
files in place. Review page (private): https://claude.ai/artifact/HSUC5FPTKBP63EMx3RBi98. Background and source design:
docs/CLIMATE.md 6.1.

## Run

    Core samples south of 57.1°S (docs/evidence/era5land_south_samples.py)
      -> majortom.climate stream --source era5-land-cds-south, per year x 6 variable groups
         -> /data/Major-TOM/Core-Weather/era5land-cds-south-YYYY-gN/{hourly,daily}
      -> majortom.climate merge --fill (per year) -> merged/{hourly,daily}/year=YYYY/<day>.parquet, in place
      -> docs/evidence/south_fill_check.py (job 2931) -> review page -> re-upload

| Year | Groups ran | Fill completed |
|---|---|---|
| 2016 | 2026-11-02 18:05 to 11-03 11:20 | 11-03 11:21 |
| 2018 | 11-03 16:43 to 11-05 08:21 | 11-05 08:23 |
| 2019 | 11-05 08:23 to 11-11 20:41 | 11-11 20:51 |
| 2020 | 11-05 08:47 to 11-10 19:08 | 11-10 19:18 |
| 2021 | 11-10 19:18 to 11-13 23:38 | 11-13 23:49 |
| 2022 | 11-13 23:49 to 11-16 18:55 | 11-16 19:05 |
| 2023 | 11-16 19:05 to 11-19 10:21 | 11-19 10:29 |
| 2024 | 11-19 10:29 to 11-19 15:56 | 11-19 15:56 |

80 Slurm jobs (4 CPUs, 8 GB each; fills 4 CPUs, 12-32 GB), 2,673 h wall, 10,690 allocated CPU-h. The six groups of a
year ran in parallel; years were chained by dependencies, except 2019 and 2020, which overlapped. Throughput was set
by the Climate Data Store queue, not by the server: 40-150 days per group per day.

## Failures

Five group jobs stopped on a request the store reported as failed (`HTTPError: 400 ... The job has failed` on the
results URL): 2019 g4 and g5, 2021 g4 and g6, 2022 g5. Each was resubmitted with the same command and cache; days
already written are skipped, so no work was lost. The year's fill job was re-pointed at the new job with
`scontrol update Dependency=`. 2022 g5 stood 18 h before the failure was noticed (no watcher running).

`majortom/climate/readers/queue.py:44` does not retry this error. CDS 500 errors and dropped connections are retried by
the CDS client itself (2023 ran through a day of them). A retry around `retrieve` is proposed, not applied.

## Result

| | Value |
|---|--:|
| Day files changed (hourly + daily) | 2,450 (1,225 + 1,225) |
| Size of changed files | 241,933,060,233 bytes |
| Daily rows given ERA5-Land values | 3,235,318 |
| Cells | 52,776 (60.6°S to 82.8°S) |
| South rows over the sea, still empty | 3,021,269 |
| Fill rows not applied | 0 |
| ERA5-Land minus ERA5 t2m: mean, std, correlation | -0.42 K, 0.92 K, 0.998 |

The fills rewrote 2,684 day files. 234 of them had sea rows only on that day; they came out byte-identical to the
published files and are not re-uploaded. 2015 and 2017 have no Core acquisitions south of 57.1°S.

`era5land_valid` is the share of the cell with values (0 at sea), so "has values" is `era5land_valid > 0`, not
`notna()`: the published south rows carry 0, not empty. All 52 `era5land_*` columns are filled in every filled row.
ERA5-Land's glacier constants show up as expected: `sd` 10 m water equivalent, `sde` 33.33 m, `rsn` 300 kg m⁻³,
`lai_hv` and `lai_lv` 0.

Published copies of the three days with most filled rows, read over HTTPS before the upload, have 0 south rows with
values; the local copies have 14,335, 14,518 and 15,677.

## Documentation

- `docs/datasets/Core-Weather.md` (published as the README without its first 4 lines): ERA5-Land grid no longer
  "90°N–57.1°S"; south note rewritten with the figures above; credit row for the Climate Data Store; Size updated to
  137.8 GB hourly and 258.3 GB daily.
- `docs/CLIMATE.md` 6.1: fill marked as run, result bullet added.

## Upload

`publish(merged, "source.coop://major-tom/core-weather", include=[README.md, acquisitions.parquet,
hourly/year=*/*.parquet, daily/year=*/*.parquet])`, as for the first upload (job 2240). The CLI's `publish` takes TACO
zips only, so the call runs from a session script.

- Dry run, job 2934 (22 min 46 s): 2,451 files to upload (README, 1,225 hourly, 1,225 daily), 3,503 already there.
  The list equals the changed files found by the check.
- Upload, job 2935 (8 CPUs, 8 workers, 2 h 33 min 54 s, 2026-11-20): `2451 uploaded, 3503 already there, verified
  True`. Verification compares size and ETag of every file with the source.coop listing.
- Published README: 11,286 bytes, equal to the staged copy.
- 2026-11-26: README re-uploaded alone (`include=["README.md"]`, verified True, 11,425 bytes) with one added sentence:
  ERA5 (`era5_*`, 0.25°) always covered these cells; only `era5land_*` was empty there.

## Cleanup

2026-11-26: the 48 `era5land-cds-south-YYYY-gN` build folders and the ERA5-Land caches (`cache`, `cache-south-2020` to
`cache-south-2024`) under /data/Major-TOM/Core-Weather deleted, 43 GB. The merged day files hold everything they
produced.

## Decisions

- Fill in place, not a new version: same columns and files; the change is a set of day files.
- Check the fill against ERA5 2 m temperature: ERA5 covers these cells at 0.25° and was already published.
- Upload only changed files: `publish` compares size and ETag with the remote listing.

## Look at first

- docs/evidence/south_fill_check.py
- majortom/climate/merge.py:59 (`fill`)
- docs/datasets/Core-Weather.md (ERA5-Land note)
