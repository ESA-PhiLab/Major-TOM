> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed. Made in one batch after Miko's go-ahead; needs review.

# 2026-10-05 — Workflow GIF, Snowbird acquisition race, archive matrix, white-background fix

Adds `gif0_workflow.gif`, `gif5_acquisition.gif` and `archives_matrix.png` to `docs/animations/out/`. Fixes the off-white GIF background. Not committed.

## Background fix: docs/animations/style.py:114

GIF backgrounds were `#fcfcfc`, not white: Pillow's palette lookup works at about 6 bits per channel and picked a fitted near-white slot over the exact one. `to_palette` maps colours at full precision; brand slots come first, so they win ties.

    colours, inverse = np.unique(flat @ np.array([65536, 256, 1]), return_inverse=True)
    rgb = np.stack([colours >> 16, (colours >> 8) & 255, colours & 255], axis=1)
    nearest = ((rgb[:, None, :] - rgb_palette[None]) ** 2).sum(-1).argmin(1)

The header background slot is now 1 (white). `verify_gif` reports the decoded corner pixel; all GIFs give (255, 255, 255). Sizes grew 0.1–0.3 MB.

## Measurements: docs/animations/acquire_snowbird.py

Cell 451U_946L (Snowbird), Sentinel-2 L2A 2023-04-15, tile 12TVK: the product Major TOM Core-S2L2A holds (`data/core_meta_451U_946L.json`). Each lane runs in a fresh process, 3 rounds interleaved, GDAL caching off. Reads: B04, B03, B02 at 10 m + scene classes (SCL, or Core's `cloud_mask`) for the v2 1056 px window.

    def read_cogs(hrefs, clock):                      # acquire_snowbird.py:54
        with clock.step("read"):
            for band, href in hrefs.items():
                with rasterio.open(href) as src:
                    out[band] = src.read(1, window=from_bounds(*WINDOW, src.transform))

Median step times (s), from our server, 2026-10-05:

| Lane | Search | Access | Read 4 bands | Product found |
|---|---|---|---|---|
| Major TOM (HF) | 0.00 (cached metadata row) | — | 2.0 | `..._N0509_..._20230415T211527` |
| Planetary Computer | 0.23 | 0.10 (sign) | 2.35 | `..._R127_T12TVK_20240901T113744` |
| AWS Earth Search | 0.83 | 0.00 | 17.7 | `S2B_T12TVK_20230415T182242_L2A` (c1) |
| CDSE | 0.33 | login | — | `..._N0510_..._20240901T113744` |
| Google Earth Engine | — | login | — | not measured |

CDSE and Planetary Computer serve the 2024 reprocessing (N0510); Major TOM holds the original N0509. Earth Engine and CDSE credentials were not used: reading the credential files was blocked, and they need Miko's decision.

Chips (`data/chip_*.png`) show the same scene from all three readable lanes.

## docs/animations/gif5_acquisition.py

Replays the medians at 1× (`SPEED = 1.0`; 0.5 gives 2× slow motion). Each lane's image fills row by row during its read. Login lanes show a dashed lock box after their open steps.

    for name, seconds in steps:                        # gif5_acquisition.py:62 (draw_lane)
        shown = min(max(clock - start, 0.0), seconds)
        ...
        if name == "read":
            read_progress = shown / seconds

26 s loop, 520 frames, 1.4 MB, seam 0.000, decode check 0 error.

## docs/animations/gif0_workflow.py

Six stations (grid, footprint, archive query, read + cloud mask, rumi + GeoZL, TACO) drawn as flat icons. A sample travels left to right; stations light on arrival; brackets mark "majortom.build" (1–4) and "AI-ready format" (5–6). 12 s, 0.7 MB. rumi is described with its spec's wording: "stateless raster storage", "GeoTIFF-inspired format".

## docs/animations/fig_archives.py

Static 2560×1440 PNG. Facts verified 2026-10-05 by a research agent against live STAC `/collections` and docs:

- earth-search.aws.element84.com/v1/collections, registry.opendata.aws/sentinel-2/, /sentinel-1/
- documentation.dataspace.copernicus.eu (Sentinel2, Sentinel1, S3, Quotas)
- developers.google.com/earth-engine (S2_SR_HARMONIZED, S2_HARMONIZED, S1_GRD, access, computePixels, usage)
- planetarycomputer.microsoft.com/api/stac/v1/collections, docs/concepts/sas

Corrections to the draft claims, now on the slide:

- AWS: L1C exists (requester-pays JP2); S1 GRD is open too; L2A is free COG.
- CDSE: "slow" is not documented; the limit is 4 connections × 20 MB/s and 12 TB/month.
- GEE: needs a Cloud project plus noncommercial/commercial registration; L2A from 2017.
- Planetary Computer: no L1C (confirmed); only archive with stored S1 RTC files. CDSE offers gamma0 on the fly; GEE S1_GRD is terrain-corrected, not RTC.
- L2A +1000 offset: kept and declared (Earth Search c1, CDSE), removed (GEE harmonised), kept but undeclared (Planetary Computer).

## Polar facts (answer to Miko's question, not yet a figure)

From grid.py (`get_rows`, `subdivide_circumference`, d = 10 km): 2004 rows. Row 1002D is exactly −90° (one cell). The northernmost row 1001U is at 89.91° with 7 cells; 1000U has 13, 999U 19. Cells stay 9–10 km wide to the pole. `Grid()` defaults to `latitude_range=(-85, 85)`. UTM is defined only between 80°S and 84°N (UPS beyond). Sentinel-2 images systematically between 56°S and 82.8°N; Antarctica only on request.

## Decisions

- Same product in every lane over "latest clear scene": the race compares archives, not dates.
- Real time over slow motion: the slow lane already takes 18.6 s; `SPEED` changes it.
- Login lanes shown as locks, not estimated times: no invented numbers.
- Major TOM column added to the matrix: it is the point of the slide sequence.
- Matrix as a static PNG, not a GIF: comparisons read better standing still.

## Known gaps

- Times depend on network location; our server's location is not stated on the slides. AWS (Oregon) is close to Utah, so Earth Search will likely be faster at the workshop.
- Major TOM's "search" uses its metadata row cached locally; fetching metadata.parquet the first time is not timed.
- Major TOM reads its own v1 1068 px window, the others the v2 1056 px window.
- CDSE download and Earth Engine are unmeasured pending credentials.

## Look at first

- docs/animations/acquire_snowbird.py:77 (Major TOM lane) and :54 (COG lanes)
- docs/animations/out/archives_matrix.png
- docs/animations/style.py:114 (exact palette mapping)
