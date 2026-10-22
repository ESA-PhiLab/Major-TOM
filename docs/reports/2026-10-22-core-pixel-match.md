> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed

# 2026-10-22 — Core pixel match against the sources

Which source pixels each Core sample holds, for 12 samples per Core dataset (48 samples, 93 band comparisons). Extends the single-sample check of `docs/reports/2026-09-30-refactor-research.md`. Script: `docs/evidence/pixel_match_set.py`; rows: `docs/evidence/pixel_match.csv`.

## Method

Samples are spread over processing versions, hemispheres and UTM zone edges. Each Core band is compared with the exact source product:

    S2 L1C/L2A  CDSE by product id (Core was built from the same EODATA store); the version Core used, or
                else the same acquisition reprocessed (geometry check only)
    S1 RTC      Planetary Computer, same product id
    DEM         Planetary Computer Copernicus DEM

- Native comparison: whole-pixel offsets within ±3 px of the stored origin, searched for 100% equal pixels (also allowing ±1000).
- Resampled comparison (source in another CRS, or no exact copy): source resampled onto the stored grid, shifted ±30 m (5 m, then 1 m steps), lowest RMSE.

## Results

| Dataset | Samples | Exact copy at the rounded origin | Other |
|---|---|---|---|
| Core-S2L2A | 12 | 4 (12 bands), all 100% equal | 6 version gone, reprocessed source; 1 found only in another zone; 1 no version on CDSE |
| Core-S2L1C | 12 | 4 (12 bands), all 100% equal | 8 version gone, reprocessed source |
| Core-S1RTC | 12 | 8, all 100% equal | 4 resampled from the same product: best shift within ±5 m |
| Core-DEM | 12 | (resampled from latitude/longitude) | RMSE ≤ 2.5 m on the stored grid; best shifts scatter within ±15 m |

Every exact copy holds the source pixels unchanged, read at the stored origin **rounded to the band's own pixel grid**. The stored origin is the unrounded value: errors up to half a pixel (B04 ≤ 5 m, B11 ≤ 10 m, B01 ≤ 30 m; S1 ≤ 5 m). The value shift is 0 for every band: Core keeps the source values, offset included.

    rounded_col = rounded_row = True   for all 24 S2 bands and 8 S1 samples matched exactly
    equal_share = 1.0                  same samples

## Source versions

CDSE no longer holds the versions most Core Sentinel-2 samples were cut from. Of 24 S2 samples, 8 matched their exact version (all 05.00). For the other 16, CDSE holds:

| Core version | CDSE holds now |
|---|---|
| 02.xx, 03.xx (2016-2021) | 05.00 |
| 04.00, 03.01 (2022) | 05.10 |
| 02.01 (one 2016 scene) | nothing |

The reprocessed versions correlate with Core at 0.97-0.9999, but at 10 m the best whole-pixel offset agrees with the rounding rule in only 43-50% of samples (B01: 100%, B11: 86-100%). Collection 1 reprocessing also refined geolocation, so reprocessed pixels can move by up to a pixel; they cannot verify Core's geometry at 10 m.

## DEM

Core-DEM was resampled from the latitude/longitude Copernicus DEM. The best shift does not follow the stored origin's distance to the 30 m grid (no rounding pattern). Shifts within ±15 m (half a source pixel) fit a different pixel-centre convention or resampling method. Two cells (one flat) sit at the ±30 m search edge and are unresolved.

## Implications for `correct_core_transform`

- Native samples (Core-S2L2A, Core-S2L1C, most Core-S1RTC): replace the stored origin with the origin rounded to the band's pixel grid. Exact for every matched sample.
- Resampled Core-S1RTC samples (4 of 12) and Core-DEM: keep the stored transform. Residual shifts are within half a pixel and have no rule to correct.
- A per-sample flag is needed to tell native from resampled Core-S1RTC samples. Not derivable from metadata yet.

## Decisions

- Matched S2 by exact product id on CDSE over date-and-tile matching: only the exact version can prove pixel equality.
- Reprocessed sources used for geometry only, and reported as inconclusive at 10 m.
- DEM search by RMSE over shifts over equality: the DEM is resampled, no exact copy exists.

## Look at first

- docs/evidence/pixel_match.csv (one row per sample and band)
- docs/evidence/pixel_match_set.py:118 (`native_match`)
- docs/evidence/pixel_match_set.py:167 (`shift_match`)
