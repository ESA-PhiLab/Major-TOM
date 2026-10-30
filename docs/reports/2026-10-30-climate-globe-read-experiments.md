> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-10-30 — Climate globe: reading the published data in a browser

Four experiments with `apps/climate-globe`, reading `major-tom/climate` from source.coop in the page (hyparquet,
zarrita). Measured from cirrus in headless Chromium; a home connection adds download time on top. Every value read
in the page was checked against the files in Python.

## Cost of a request

Range requests to `data.source.coop` wait 0.2-1.3 s before the first byte; one connection downloads at ~28 MB/s.
Time follows the number of requests more than bytes. `apps/climate-globe/data.js` therefore:

- reads row groups 16 at a time (hyparquet reads those of one call one after another);
- opens a file with one request for its last 128 KB, which gives the size and the footer (no HEAD request);
- tells hyparquet the same tail size; its default is to fetch its own last 512 KB again
  (`initialFetchSize`), which tripled the bytes of a cell series (500 MB -> 153 MB).

## 1. Selected cell

Click a cell: its monthly normals from one climatology band file, then on request its full series.

    const [r0, r1] = bounds(group, 'grid_row_u'), [c0, c1] = bounds(group, 'grid_col_r')
    if (r0 <= row && row <= r1 && c0 <= col && col <= c1) reads.push([start, end])

`data.js:loadCellClimatology` reads only row groups whose statistics contain the cell (1-2; rows follow the
Hilbert curve). Loch Lomond (624U_29L): 738 KB, 3 requests, 1.2-1.9 s.

Full series: every monthly file, its footer plus the column chunk holding the cell. CPC, 564 months:
153 MB in 1,142 requests, 10.1 s, for 564 numbers.

## 2. Zarr test store

`docs/evidence/climate_zarr_store.py` writes CPC precipitation on the (row, column) array: map months in
512 x 512 chunks with 2 x 2 and 4 x 4 overviews, and the series in (564, 32, 32) chunks in shards.
Values equal the Parquet files (sampled check, four months).

| Read | Parquet | Zarr |
|---|---|---|
| Globe month | 4.2 MB, 26 requests, 2.0-2.5 s | 0.4 MB, 5 requests, 0.5-1.6 s (4 x 4 overview) |
| Full-resolution month | 4.2 MB, 26 requests | 2.1 MB, 35 requests, 1.7 s |
| One cell, 564 months | 153 MB, 1,144 requests, 10.7 s | 767 KB, 6 requests, 0.7 s |

Identical cell values from both.

## 3. Loading ahead

`app.js:loadAhead` loads the neighbouring months after one is drawn; `fetchField` joins a load under way.
A step then draws in 0.15 s (was 2-2.5 s for Parquet). Stepping faster than loads finish: 2.1 s Parquet,
0.37 s Zarr.

## 4. Larger row groups

`docs/evidence/climate_rowgroup_files.py`, January-June 2024, all columns:

| Rows per group | CPC month | IMERG month | One cell, per file (CPC / IMERG) |
|---|---|---|---|
| 65,536 (published) | 26 requests, 2.0 s | 79 requests, 6.2 s | 0.36 / 0.48 MB |
| 262,144 | 8 requests, 0.9 s | 21 requests, 1.9 s | 0.93 / 1.5 MB |
| 1,048,576 | 3 requests, 0.5 s | 6 requests, 1.1 s | 2.6 / 4.3 MB |

Map months get 2-6x faster; a cell's series 3-7x more bytes.

## 5. Parquet in two more layouts, instead of Zarr

`docs/evidence/climate_bycell_layout.py`, CPC, all four columns:

- `by_cell/band=NN.parquet`: rows by Hilbert curve, then year and month; row groups of 116 cells x 564 months.
  8.4 GB, against 11.6 GB for the monthly files (a cell's months compress better side by side).
- `overviews/level=1|2/year=YYYY/MM.parquet`: 2 x 2 and 4 x 4 block means, 5.2 and 1.6 MB a month.

Checked against the monthly files (sampled cells and months; one month of block means). In the app:

| Read | Monthly Parquet | Parquet test layouts | Zarr |
|---|---|---|---|
| Globe month (4 x 4) | 4.2 MB, 26 requests, 2-2.5 s | 0.7 MB, 2 requests, 0.3-1.7 s | 0.4 MB, 5 requests, 0.5-1.6 s |
| Month at 2 x 2 | – | 2.1 MB, 2 requests, 1.3 s | 1.2 MB, 11 requests, 0.5 s |
| One cell, 564 months | 153 MB, 1,140 requests, 10.2 s | 1.5 MB, 4 requests, 0.7 s | 767 KB, 6 requests, 0.6 s |

Training-style reads (`docs/evidence/climate_bycell_bench.py`; pyarrow over HTTP, 16 at a time):

| 1,000 cells, all months, precipitation | Time | Read | Cells/s |
|---|---|---|---|
| By cell, neighbouring | 1.8 s | 3.1 MB | 570 |
| By cell, random | 98 s | 362 MB | 10 |
| Monthly files, random | 240 s | 2,323 MB | 4 |

A random cell costs its whole row group (116 cells). Loaders should shuffle row groups, then cells within a
buffer, or download the band files first (8.4 GB for CPC).

## Findings

- Monthly Parquet files suit map months and SQL over months; they are poor for one place over time.
- A Zarr copy with overviews and long time chunks is faster for both uses in a browser, by 5x (globe month) and
  ~200x in bytes (cell series).
- Larger Parquet row groups help maps only, and cost point reads; with a Zarr copy for the app they are not needed.
- Parquet by cell plus overview tables matches Zarr for the app (within 2x in bytes, same time), keeps one
  format, the `grid_cell` key and streaming into Hugging Face `datasets`; it also serves time-series training.

## Decisions

- Kept 65,536-row groups in the release: point and region queries stay cheap; the app can use Zarr for speed.
- Zarr compared in our own renderer, not deck.gl-raster: same drawing, so only the format differs; deck.gl-raster
  has no globe until deck.gl v10.

## Open

- Test files stay at `major-tom/climate/_experiments/` while the Zarr option is tried in the app; to be removed
  before the release goes public.

## Look at first

- apps/climate-globe/data.js (loaders, `countedFile`, `loadCellClimatology`, `loadZarrMonth`)
- docs/evidence/climate_zarr_store.py
- apps/climate-globe/app.js (`fetchField`, `loadAhead`)
