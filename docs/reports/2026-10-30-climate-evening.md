> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-10-30 — Climate release: CPC gauges, ERA5-Monthly, Core-Weather checks, app tests

Evening and overnight work. Uploads go to the Unlisted `major-tom/climate`; READMEs wait for review.

## CPC-Monthly: cpc_gauges

`majortom/climate/gauges.py`: per cell and month, the mean daily number of rain gauges in the 3 x 3 CPC pixels
around the cell (1.5° x 1.5°), from CPC's own daily files (`gnum`; NOAA PSL's netCDF copy leaves it out).

    field, days = month_gauges(raw, year, month)       # mean daily gnum per pixel, then 3 x 3 sum
    values, _ = cell_means(_w, field)                  # area weights, as cpc_precip

Raw files are south-up float32 (2 x 360 x 720), gzip-compressed up to 2005 and in 2006-2022. Added to all 564
months, climatology recomputed, release layouts rebuilt, published (1,168 files, verified). Every other value
unchanged against the previous release (5 months, 3 bands checked); `grid_cell` is now `large_string` (values equal).

## ERA5-Monthly

Repacked without `ptype` (`repack --drop`): 1,032 months, 201 variables. The first run ran out of memory in the
climatology (6 bands at once, ~30 GB each); `repack` now renames a file into place only after its check and takes
`--only climatology`; rerun with 2 at a time. Overviews for 8 surface variables (`overviews --columns`).

Validation: 5,113,745 cells, all months, no `ptype`, climatology 61,364,940 rows with 30 years everywhere. Largest
missing shares in ocean-only variables (up to 86%). Upload of 2.8 TB started (job 2106).

Figures: annual precipitation (same scale as CPC and IMERG) and January/July 2 m temperature. The overview collage
now has eight maps; the places chart has ERA5 as a third precipitation line.

## Core-Weather

`docs/evidence/core_weather_validate.py`, metadata only: all 2,976 days in every finished source, and on every day
the same number of rows in every source. Whole-day null columns are expected (no cloud, ocean-only, lakes, S5P
before 2018-06-28).

`majortom/climate/merge.py`: a column given by an earlier source is kept from it.

    part = part.drop(columns=[c for c in part.columns if c in table.columns and c not in KEYS[kind]])

`era5_land` and `era5land_valid` are identical across groups; `era5_valid` differs in groups 4-6 (counted on wave
and snow variables). With `era5` listed first, its `era5_valid` (2 m temperature, every cell) is kept.

Trial merge of four days, all finished sources: rows equal the source rows (no unmatched rows); hourly 260
columns, daily 284. Estimate for the whole: ~93 GB hourly, ~181 GB daily.

## Apps

`apps/climate-globe/tests/test_app.py`: 5 browser tests on the published data; on GitHub in `apps-tests.yml`
(preview test skipped there: images are not in git). The app has its own `pytest.ini`: the repository root's
`__init__.py` needs numpy. All three workflows pass.

## Decisions

- `era5` folder kept and merged first: it holds the 12 basic variables, none of which is in the 8 groups.
- `era5land` (old, 786 days) moved to /RUBBISH: every column is in the 6 complete groups.
- `tests` (7.6 GB, check scripts of 2026-10-23..26) left in place for review.
- Final Core-Weather merge waits for ERA5 groups 5-8 (about 2 November): merging now would mean merging twice.

## Open

- READMEs to review: docs/datasets/climate/*.md (with ERA5-Monthly), docs/datasets/Core-Weather.md.
- `era5_viten` reaches ±1.4e11: check against ERA5's definition and units.
- ERA5Land-Monthly-v2: 1998 reached; CDS downloads keep breaking (hundreds of retries).
- ERA5 group 5 hung at 16:36 and was restarted (job 2099); group 7 now starts after group 8.

## Look at first

- majortom/climate/gauges.py
- majortom/climate/merge.py:25 (columns kept from the earlier source)
- docs/datasets/climate/ERA5-Monthly.md
