> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed. Made in one batch after Miko's go-ahead; needs review.

# 2026-10-05 — Acquisition v2: cloud-native reads, CDSE lane, race B, pole slide

Replaces the first acquisition measurement (sequential reads, cached Major TOM lookup). Adds race B (newest clear scene), the CDSE lane, a pole slide and a new GIF 0 grid icon. Not committed.

## docs/animations/acquire_snowbird.py

Reads now follow cloud-native practice: header in one request, bands concurrently, HTTP/2, exact window only.

    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",            # header in one request
    "GDAL_HTTP_VERSION": "2", "GDAL_HTTP_MULTIPLEX": "YES",
    ...
    with ThreadPoolExecutor(len(hrefs)) as pool:        # read_bands, one dataset per thread
        return dict(zip(hrefs, pool.map(read_window, hrefs.values())))

Requests and bytes come from GDAL's own network statistics, printed at process exit; each lane runs in its own process. Major TOM uses fsspec, not GDAL, so its bytes are the parquet column-chunk sizes and its request count is not reported.

    "CPL_VSIL_NETWORK_STATS_ENABLED": "YES", "CPL_VSIL_SHOW_NETWORK_STATS": "YES",
    ...
    requests, nbytes = network_stats(proc.stdout + proc.stderr)

Major TOM's lookup is now a real remote query of `metadata.parquet` (173 MB, 4,492 row groups). Filtering on `grid_cell` (string) took ~11 s: string statistics barely prune. Filtering on `grid_row_u` and `grid_col_r` (integers, in file order) takes ~2.5 s, mostly the footer.

    filters=[("grid_row_u", "==", row), ("grid_col_r", "==", col)]

CDSE: STAC search, then windowed JP2 reads from `/vsis3/eodata/...` with the keys from `~/.config/cdse/s3.env` (loaded into the job's environment, never printed). Earth Engine: `ee.Initialize(project="wiki-eo")` fails with "Please authorize access"; no stored login on this machine.

## Results (medians of 3 successful runs, 2026-10-05)

| Lane | Race A total | Requests | MB | Race B total | Race B scene |
|---|---|---|---|---|---|
| Planetary Computer | 1.95 s | 12 | 12.5 | 2.34 s | 2026-10-01 |
| Major TOM | 4.37 s | n/a | 6.8 | 4.53 s | 2023-04-15 (fixed) |
| CDSE | 4.88 s | ~223 | 20.0 | 14.2 s (search 9.7 s) | 2026-10-01 |
| AWS Earth Search | 6.79 s | 12 | 18.6 | 10.8 s | 2026-10-01 |
| Earth Engine | login missing | | | login missing | |

Concurrency cut Earth Search's read from 17.7 s to 5.6 s and Planetary Computer's from 2.35 s to 1.44 s. CDSE's search was rate-limited (HTTP 429, WAF) in 2 of 4 race B attempts. Previous results are kept in `data/acquisition_runs_naive.jsonl`.

## docs/animations/gif5_acquisition.py

One script, two outputs: `--race same` → `gif5a_same_product.gif`, `--race recent` → `gif5b_newest_scene.gif`. Each finished lane shows seconds, requests, MB and the acquisition date; Major TOM's date is flagged "fixed sample" in race B. Lanes without a successful run show a lock; CDSE's 429s are noted under its name. Chips get one contrast stretch per race (2–98th percentile over all lanes).

## docs/animations/fig_poles.py

Cells of the last rows around each pole (`cell_10km`), windows of 1056 px centred on each cell's lat/lon midpoint in UPS (EPSG:32661 / 32761), coverage counted per 60 m pixel within ±32 km.

    windows = [window_centroid(c, crs, snap=False) for c in cells]
    count = rasterize([(w, 1) for w in windows], ..., merge_alg=MergeAlg.add)

North: 1.7% of the shown area uncovered. South: 17.5%. The south-pole cell (1002D) is a disc; its v2 window sits on the lat/lon midpoint, 5 km off the pole. A straight strip along 0° longitude appears in both panels because column 0 starts at 0° in every row, so cell edges align there.

Caveats on the slide: grid not generated beyond ±85° by default; no UTM (UPS instead); no Sentinel-2 beyond 82.8°N, Antarctica on request; wedge cells; midpoint ≠ centre.

## docs/animations/gif0_workflow.py

Grid icon replaced by a small orthographic globe of the official grid at 2,500 km spacing, reusing `gif1_grid.project`. The matrix (`fig_archives.py`) now shows race A times and a "newest clear scene" row.

## Other changes

- `style.py` `to_palette`: colour matching in chunks of 8,192 colours. The first race render was killed (exit 137) by a multi-GB temporary from photo thumbnails.

## Update: Earth Engine lane

`lane_gee` authenticates with the wiki-eo service account (`~/.config/earthengine/ee-sa.json`, read only for `client_email`, never printed), as Miko specified.

    email = json.loads(key.read_text())["client_email"]          # read only; never printed
    ee.Initialize(ee.ServiceAccountCredentials(email, str(key)), project="wiki-eo")

Sums of per-step medians, 3 runs each: race A 4.7 s (access 1.37, search 0.50, computePixels 2.84); race B 4.6 s, scene 2026-10-01. No byte count: the traffic does not go through GDAL. The six earlier "login missing" records were moved out of `acquisition_runs.jsonl`. GIF 5 and the matrix now show sums of per-step medians, so both slides give the same numbers.

## Decisions

- Integer filters over `grid_cell` for the Major TOM lookup: 2.5 s vs 11 s; what an expert would write.
- GDAL network statistics over debug-log parsing: the S3 driver does not log downloads.
- CDSE raw S3 reads over Sentinel Hub: same credentials, shows the archive as stored.
- Failed attempts recorded, medians over successful runs only.

## Known gaps

- Times are from our server; location not stated on slides.
- Major TOM still reads its v1 1068 px window; the others the v2 1056 px window.
- A cell index for Major TOM metadata (sorted by cell, or a small lookup file) would cut its 2.5 s lookup; candidate for WS1/WS4.

## Look at first

- docs/animations/acquire_snowbird.py:101 (concurrent reads) and the env block at the top
- docs/animations/out/gif5b_newest_scene.gif
- docs/animations/out/poles_1056.png
