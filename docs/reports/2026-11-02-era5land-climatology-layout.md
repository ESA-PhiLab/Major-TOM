> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-11-02 — ERA5-Land built; ERA5-Monthly-v2 removed; climatology with a page index; file layout documented

ERA5Land-Monthly built, validated, uploading. ERA5 overviews for all 201 variables. ERA5-Monthly-v2 removed after
a check, `era5_ptype` kept. ERA5 `by_cell/` building (first run killed at 100 GB: the check read whole files; fixed, rerun as job 2178). Climatology files rewritten with a page index. README section
on the file layout.

## Push and release-test

`refactor/v1` pushed (3261211); Apps to Spaces passed. Release-test README regenerated
(`docs/evidence/release_test_readme.py`, release-bench env: needs `taco`) and uploaded: commit links and the author
section changed, verified.

## ERA5Land-Monthly (job 2164)

Download (job 1919) ended 2026-11-01 05:25: 912 months, CDS timeouts all recovered. One job: climatology (29 min),
repack (21 min), overviews of every column (8 min), climatology overviews, `by_cell/` (47 min, 22.9 GB peak).

    climatology --out ERA5Land-Monthly-v2 --period 1991 2020; tables repack --src ... --out climate-release/...
    tables overviews ...; tables overviews --climatology ...; tables by-cell ... --workers 4

`docs/evidence/monthly_validate.py` (wrapper, published folder only): 1,524,009 cells, 1950-01 to 2025-12 without
gaps, 49 variables, no nulls, `era5land_valid` in [0, 1], climatology 30 years everywhere. Rows reach row -1002:
the CDS monthly copy covers Antarctica (the Earth Data Hub gap does not apply). 470 GB.

README draft `docs/datasets/climate/ERA5Land-Monthly.md`; table from `docs/evidence/era5_monthly_variables.py`,
which now takes a source name (ERA5's table reproduced unchanged).

## ERA5-Monthly-v2 removed

Checked before removal: 1940-01, 1987-05, 2024-07 compared by cell with the published copy, all 202 shared columns
identical, no NaN differences; 1,032 months and 20 bands in both. Only `era5_ptype` was not published.

`era5_ptype` kept at `/data/Major-TOM/ERA5-Monthly-ptype/` (same paths; `grid_cell`, `year`, `month`, ptype
columns; every file read back equal): 1,052 files, 8.9 GB. Then `find ERA5-Monthly-v2 -delete` (2.8 TB; too large
for /RUBBISH on the system disk). /data: 1.6 TB -> 4.1 TB free.

## ERA5 overviews and by_cell

`tables overviews` without `--columns` (job 2165): 1,032 files, all 201 variables, 221 GB, 1 h 39 min.
`apps/climate-globe/scales.json` rebuilt: every variable has overviews.

`tables by-cell --workers 3` (job 2168), 100 GB: peak RSS reached the limit during the first three bands, which
finished (band 00 checked). Estimated 16-20 h, ~2.1 TB.

## majortom/release/tables/layout.py: climatology page index

Test on ERA5 band 10 (scratch, Chromium, hyparquet): same row groups, page index, no value dictionaries.

| | File | Footer | One cell, one variable | One cell, all 201 |
|---|---|---|---|---|
| Before | 3.66 GB | 3.18 MB | 3.56 MB | 51.35 MB |
| After | 2.28 GB | 3.25 MB | 3.26 MB | 3.60 MB |

Row groups kept at 65,536 rows: with 1,024 cells, the footer (74 row groups x 407 columns) would grow about five
times. `_write_options` gives climatology files the by_cell settings; `relayout_climatology` rewrites a published
folder in place, each file checked equal before `os.replace`.

    return dict(write_page_index=True, use_dictionary=[c for c in KEYS if c in table.column_names],
                use_byte_stream_split=floats, data_page_size=4096, write_batch_size=12)

CLI `tables relayout --folder`. Job 2170: CPC 0.26 -> 0.19 GB, IMERG 2.32 -> 1.41, CAMS 2.91 -> 2.20, ERA5-Land
7.47 -> 4.47, ERA5 64.6 -> 38.0. Test `test_climatology_rows_by_rank_and_page_index`.

## apps/climate-globe: normals by rank

`loadCellClimatology` reads rows `12 x rank` through the page index when the band has 12 rows per cell and every
column an offset index; it checks the rows are the cell and otherwise reads by row-group statistics. 6/6 app tests
pass on the published data (CPC climatology already uploaded). Not pushed: `scales.json` points every ERA5 variable
to `overviews/`, still uploading.

## docs/datasets/climate/README.md: File layout

Diagram of the five folders, the rank rule, settings per folder, sizes and largest footers per dataset
(`layout_stats.py`, metadata only), measured reads. The by_cell write notes and the ERA5 band 00 table moved there.

## Uploads (job 2173)

CPC, IMERG, CAMS climatology: uploaded, verified. Then ERA5 climatology and overviews, then ERA5Land-Monthly data
(README after review).

## Core-Weather

Groups 6 and 7 finished. `docs/evidence/core_weather_validate.py`: 2,976 days in every source, hourly and daily,
no day with differing rows. Merge not started (~93 GB hourly, ~181 GB daily).

## Decisions

- Climatology row groups kept at 65,536 rows: ERA5 footer stays 3.3 MB, not ~16 MB.
- `era5_ptype` kept on /data, not uploaded: not published, 8.9 GB.
- Real delete of ERA5-Monthly-v2: /RUBBISH is on the system disk (911 GB free).
- App not pushed until ERA5 overviews are on source.coop.

## Open

- Push the app after the ERA5 overview upload; ERA5 `by_cell/` upload after the build.
- ERA5-Monthly README: overviews line, `by_cell/` note, climatology layout.
- Core-Weather merge: sources, order, ERA5-Land gap south of 57.1°S.
- Old ERA5-Land copies after the upload: `ERA5Land-Monthly-v2` (255 GB), `ERA5Land-Monthly` v1 (68 GB).
- `IMERG-Monthly-v2` (31 GB): build of the published copy; 1998-01, 2010-06, 2024-12 identical by cell, no extra
  columns. Removable.
- `Core-Weather/cache` (1.9 TB; `era5land/blocks` 1.4 TB, `era5land/monthly` 176 GB, `era5` 213 GB, `cams` 127 GB,
  `cpc` 29 GB, `s5p` 9 GB): after the merge upload. Rebuilding Core-Weather for new samples would download again.
- READMEs for review: top level, CPC, IMERG, CAMS, ERA5-Monthly, ERA5Land-Monthly.

## Look at first

- majortom/release/tables/layout.py:37 (`_write_options`)
- apps/climate-globe/data.js (`loadCellClimatology`)
- docs/datasets/climate/README.md (File layout)
