> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-10-23 — Climate readers for all six sources

Readers for ERA5-Land, CAMS, NOAA CPC, IMERG and Sentinel-5P, beside ERA5. One `stream` command builds any of them. Hourly sources go through the rolling buffer; daily sources through a new daily path. 82 offline tests pass. Real tests: Sentinel-5P and CPC work; ERA5-Land and CAMS are rerunning; IMERG is refused (401).

## Layout

`majortom/climate/readers/` holds one module per access method. `sources.py` keeps the descriptions and dispatches.

    zarr.py          ERA5 (ARCO), moved unchanged
    queue.py         ERA5-Land daily statistics, CAMS EAC4: a month per request, cached
    files.py         NOAA CPC yearly netCDF, IMERG half-hourly HDF5: downloaded once, resumable
    earthengine.py   Sentinel-5P: UTC-day mosaic, 32 tiles of 900 x 900 at 0.05°

Each `Source` now names its `reader` and `period` ("hour" or "day"). `python -m majortom.climate stream` picks the path from the period (`majortom/climate/__main__.py:71`).

## Hourly path: 3-hourly sources

Off-step hours are left empty and not read; daily statistics ignore empty hours.

    if not on_step(src, hours[i]):                  # e.g. CAMS at 01:00: no field, left empty
        values[i, :, j] = np.nan

`majortom/climate/stream.py:80`. Sums are empty, not 0, when no hour has a value (`majortom/climate/series.py:122`).

## Daily path: `majortom/climate/days.py`

Per acquisition, a group (ERA5-Land: its UTC-offset group); per daily step, the source day that fits the window. Each source day's cell means are computed once and cached as `<cache>/<source>/means/<key>.npz`; tables are built from the cache.

    centres = acquired[:, None] + np.arange(-daily, daily + 1) * DAY
    days = rdr.days(src, centres, lon[:, None], group[:, None])
    return days.astype(np.int64) * 100 + group[:, None] + OFFSET

`majortom/climate/days.py:47`. A restart reads nothing already cached (test: `tests/test_climate_days.py:87`). Each row records `<source>_start`, the UTC start of the source day used.

Source days per reader:
- ERA5-Land: group = 12 − UTC hour, in 3-hour steps; day = local date at that offset (`readers/queue.py:56`).
- CPC: the UTC day holding the window centre (`readers/files.py:62`).
- Sentinel-5P: the UTC day of the ~13:30 local overpass on the local day of the window centre (`readers/earthengine.py:43`).

## Real tests

Sentinel-5P, one acquisition near Loch Lomond (2021-08-25), ±2 days: all 7 products, ~5 s per global field, ~1,700 EECU-s. NO₂ 1.6 × 10⁻⁵ mol m⁻², O₃ 0.128 mol m⁻² (≈ 290 DU); CH₄ empty (quality filtering). Full build: ~14,800 fields, ~20 h, ~200 EECU-h.

NOAA CPC, 20 Loch Lomond acquisitions: 1,220 rows, no gaps. Against ERA5's daily windows: precipitation r = 0.78, daily maximum r = 0.87, CPC maximum 1.0 K warmer. Day alignment: same day r = 0.78, day before 0.30, day after 0.43, so CPC days are 00–24 UTC here.

ERA5-Land: the month request waited 2 h 49 min in the Climate Data Store queue, then ran 16 min (12:45 → 15:50 UTC). The test job had already stopped with the session. Rerun as job 1688.

CAMS: request ran in 2.5 min (Atmosphere Data Store; licence and token work). Download ~270 KB/s while the ERA5 run shares the link; stopped at 957 MB with the session. Rerun as job 1689.

IMERG: GES DISC answers 401 to the Earthdata token.

## Full ERA5 run (job 1661)

480,000 of ~882,000 fields at 30/s at 16:29 UTC; ~3.75 h left. Output: `/data/Major-TOM/Core-Weather/era5/`.

## Decisions

- Daily sources cache cell means per source day, not a rolling buffer: few periods, each read once, free restarts.
- CPC day start kept at 00 UTC: matches the files' labels and the ERA5 test.
- Sentinel-5P tiles fetched in 8 threads per field: ~5 s per field; not tuned further.
- IMERG and Sentinel-5P full downloads not started: decisions below.

## Open decisions

See the chat summary of 2026-10-23 (ERA5-Land queue, IMERG access, CPC test outside UTC, Sentinel-5P validity and build, CAMS download speed, output layout).

## Look at first

- majortom/climate/days.py:47 (source day per step)
- majortom/climate/readers/queue.py:56 (ERA5-Land offset groups)
- majortom/climate/stream.py:80 (3-hourly steps)
