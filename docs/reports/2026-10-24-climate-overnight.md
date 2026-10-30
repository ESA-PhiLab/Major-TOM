> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-10-24 — Climate overnight: Core-Weather ERA5 built, monthly datasets started, six fixes

Core-Weather ERA5 is complete and validated. Sentinel-5P stage 1 and the five [Source]-Monthly builds are running. NOAA CPC follows CPC's own end-of-day map. IMERG now comes from Earth Engine. Six bugs found and fixed, two of them silent. 90 offline tests pass.

## Core-Weather ERA5 (job 1661)

882,144 fields in 10 h 45 min. `docs/evidence/core_weather_validate.py` on `/data/Major-TOM/Core-Weather/era5/`:

    samples 5,961,727 (Core-S2L1C 2,245,886, Core-S2L2A 2,245,886, Core-S1RTC 1,469,955), acquisitions 3,704,013
    hourly: 181,496,637 rows; expected 181,496,637 (1.0000)
    daily: 225,944,793 rows; expected 225,944,793 (1.0000)

No empty values except land-only (34%: open-sea cells) and sea-only variables (64%). 13 GB daily, 6.3 GB hourly, 74 MB link table.

## Fixes

- Cached day means were keyed by position in a run's cell table: another run, or a restart that skipped finished days, would have read wrong cells without error. Now keyed by a fixed cell identity, extended when a run needs more cells, and checked on every lookup (`majortom/climate/days.py:46`, test `tests/test_climate_days.py:113`).
- Sentinel-5P stage 1 crashed on its first day: an empty collection has no bands. Empty days now give an empty field; days before 2018-06-28 are stored empty without asking Earth Engine (no quota spent on 2016-2018).
- Threads deadlocked in GDAL's netCDF driver (ERA5 monthly and the ERA5-Land daily test sat idle for hours). All netCDF access now goes through one lock; files are chosen by their variable list (`majortom/climate/readers/__init__.py:64`).
- Single-variable netCDF files list no subdatasets; their name comes from the band tag.
- CAMS monthly means are stamped 21:00 on the first day; monthly reads match the month.
- Downloads that failed every retry ended in a misleading "file not found"; they now back off and report the cause.

## NOAA CPC day boundaries

CPC's README: the end of day "differs from country to country", with a 0.5° end-of-day map "based on very limited and UNCHECKED information". Land boxes: 12Z 23%, 18Z 16%, 24Z 38%, 03Z/06Z next day 22%. The reader uses the map per cell; `cpc_start` records each day's UTC start (Glasgow: 18Z). Our ERA5 test (`docs/evidence/cpc_day.py`) agrees within its broad peaks.

## IMERG from Earth Engine

    GES DISC, 1 connection:      8 files, 63 MB in 681 s = 0.09 MB/s (8 connections: reset by the server)
    Earth Engine getPixels:      one day, 48 half-hours, 1.24 GB in 96 s = 12.9 MB/s; 0.33 EECU-s per half-hour

`imerg` now reads Earth Engine's IMERG V07 (`majortom/climate/readers/earthengine.py`); `imerg-gesdisc` keeps the NASA route. IMERG monthly from Earth Engine is identical pixel for pixel to the GES DISC file and starts in 1998, not 2000.

## ERA5-Land from raw hours

Reading one stored hourly ERA5-Land image costs ~0.21 EECU-s per 600 × 600 seven-band tile (`getPixels` or `computePixels`); the earlier ~41 EECU-s per tile was the server-side 24-hour mean. Raw hours averaged on cirrus: ~76 EECU-hours for Core, ~8-26 TB moved. Recorded as a future option in [docs/CLIMATE.md](../CLIMATE.md).

## Sentinel-5P stage 1 (job 1698)

7,700 fields (the budget, ~120 EECU-hours) in 9 h 28 min, source days to 2021-07-01. 2,000 complete acquisition days written, 124,199,111 of 225,944,793 daily rows (55%); stage 2 continues from the cache after the quota resets on 1 November. A typical day (2020-05-30): valid share O₃ 0.99, aerosol index 0.96, HCHO 0.77, CO 0.69, SO₂ 0.67, NO₂ 0.64, CH₄ 0.07. CH₄ median 1,856 ppb; O₃ median 0.124 mol m⁻².

## CAMS download speed

Atmosphere Data Store: requests run in ~2 min, downloads crawl. One month at a time: 1.31 GB in 76.5 min (0.3 MB/s). Three months in flight (`python -m majortom.climate fetch`): 3.9 GB in 105 min (0.62 MB/s). Core period (~100 months, 1.25 GB each): ~2.3 days with 3 in flight.

## [Source]-Monthly

New `majortom/climate/monthly.py`: series in a fixed cell order (`cells.parquet`), one file per month, resumable; climatology one variable at a time, written in grid row bands. 2021 tests:

| Source | Cells | 12 months in |
|---|---|---|
| CPC (from daily files) | 1,630,397 | 0.4 min (files cached) |
| ERA5 | 5,113,745 | 1.2 min after download |
| ERA5-Land | 1,524,009 | 65 min (464 MB download) |
| CAMS EAC4 | 5,113,745 | 1.5 min |
| IMERG | 5,113,546 | 73 min from GES DISC; ~2.4 min from Earth Engine |

Done by morning: IMERG-Monthly (324 months, 1998-2024, 14 GB; global mean 2.76-2.84 mm/day) and CAMS-Monthly (276 months, 2003-2025, 68 GB; global median ozone column 268-289 DU). Still running at 04:00 UTC: ERA5-Monthly (to 1955 of 2025, ~14 h left), CPC-Monthly (to 1988 of 2025, ~10 h), ERA5Land-Monthly (to 1955 of 2025, ~2 days). Full builds started (16 GB, 4 CPUs each): ERA5-Monthly 1940-2025, ERA5Land-Monthly 1950-2025, CAMS-Monthly 2003-2025, IMERG-Monthly 1998-2024, CPC-Monthly 1979-2025, under `/data/Major-TOM/<name>/`. Climatology waits for 1991-2020 to be complete.

## Also added

- `merge` command: per-source folders into one hourly and one daily table, one schema per kind (`majortom/climate/merge.py`).
- `fetch` command: several CAMS months in flight.
- Draft dataset card: [docs/datasets/Core-Weather.md](../datasets/Core-Weather.md).

## Decisions

- Monthly builds started without asking: approved design (option c), reversible, resumable; first sized at 64 GB each, cancelled and resubmitted at 16 GB after the tests showed 1.3-7.4 GB peaks.
- IMERG switched to Earth Engine: same NASA product, 140× faster here; GES DISC kept.
- Core-Weather builds for CAMS and IMERG not started: they need your decision (below).

## Look at first

- majortom/climate/days.py:46 (cell identity, shared cache)
- majortom/climate/readers/files.py (CPC end-of-day map)
- docs/datasets/Core-Weather.md (draft card)
