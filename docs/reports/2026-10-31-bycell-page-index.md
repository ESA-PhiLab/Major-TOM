> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-10-31 — by_cell with a page index; climatology overviews; ERA5 in the climate globe

`by_cell/` rewritten for CPC, IMERG and CAMS in a layout that also works for ERA5 (201 variables). Climatology
overviews added to four datasets. The climate globe has an ERA5 layer with all 201 variables. Nothing pushed;
READMEs changed in the repo only.

## Disk use on /data (before this work)

Climate suite 6.4 TB (published copies 3.07 TB, builds 3.36 TB, of which ERA5-Monthly-v2 2.79 TB); Core-Weather
3.89 TB (cache 2.91 TB, of which `cache/era5land` 2.51 TB). RAID 98% full, 2.4 TB free.

## majortom/release/tables/bycell.py

A band was built in memory whole: ~270 GB for an ERA5 band. Now in slices of 4,096 cells, appended with one
`ParquetWriter`; renamed into place after the check.

    for s in range(0, len(at), SLICE):
        table = _slice(at[s:s + SLICE], months, cells, rows, schema, names, bounds)
        ...
        writer.write_table(table, row_group_size=CELLS * len(months))

Writer settings, from the ERA5 band 00 tests below:

    pq.ParquetWriter(tmp, table.schema, compression="zstd", write_statistics=True, write_page_index=True,
                     use_dictionary=POSITION, use_byte_stream_split=floats,
                     data_page_size=4096, write_batch_size=len(months))

ERA5 band 00 (31,712 cells, 201 variables, 1,032 months), one cell read with hyparquet 1.31.4 in Chromium:

| Layout | File | Footer | One variable | 201 variables |
|---|---|---|---|---|
| 63-cell row groups (old rule, 65,536 rows) | 18.5 GB | 10.4 MB | 11.4 MB | 119 MB |
| 1,008-cell row groups | 14.2 GB | 0.7 MB | 16.4 MB | ~1.8 GB |
| + page index | 14.7 GB | 0.8 MB | 1.8 MB | 109 MB |
| + no value dictionaries (used) | 12.8 GB | 0.7 MB | 0.8 MB | 3.3 MB |

Dictionary pages (~0.9 MB per ERA5 column chunk) are read whole before any page; removing them made the page index
pay off. Build: 18.3 min, 27.6 GB peak for band 00; full ERA5 estimated at 16-20 h with 3 workers, ~2.1 TB.

## Replaced by_cell/ (CPC, IMERG, CAMS)

Rebuilt, every band compared with the published file (all values equal), swapped, uploaded, verified.

| Dataset | New | Old |
|---|---|---|
| CPC-Monthly | 7.46 GB | 9.36 GB |
| IMERG-Monthly | 17.38 GB | 39.16 GB |
| CAMS-Monthly | 44.15 GB | 49.96 GB |

The old layout stored the band's whole `grid_cell` dictionary in every row group (CPC band 12: 334 MB of 746 MB).
Small CPC bands grew (up to +20%): rounded values compress better with dictionaries.

Reads of CPC on source.coop (README table updated):

| Read | Old | New |
|---|---|---|
| App, one cell's series | 4 requests, 1.5 MB, 0.7 s | 10 requests, 0.18 MB, 0.8 s |
| pandas README example | 0.9 s | 1.5 s |
| pyarrow, 1,000 random cells | 98 s, 0.4 GB | 60 s, 1.1 GB |
| pyarrow, 1,000 neighbouring cells | 1.8 s, 3 MB | 1.1 s, 2.3 MB |

pandas and pyarrow do not use the page index: they read the cell's 1,024-cell row group.

## apps/climate-globe/data.js: loadByCellSeries

Rows found by rank, not by row-group statistics: the cell's place among its band's cells in `cells.parquet`.

    const months = Number(metadata.num_rows) / count, first = rank * months
    await parquetRead({ file, metadata, compressors, useOffsetIndex: true, rowStart: first, rowEnd: first + months,
      columns: ['grid_row_u', 'grid_col_r', layer.column], ... })

`year` and `month` follow from the order and are not read (16 requests -> 10). Footers kept per band.

## majortom/release/tables/overviews.py: climatology_overviews

`climatology/band=NN.parquet` -> per calendar month, block means (4 x 4) of every `_mean` column ->
`climatology_overviews/MM.parquet`. `_std` left out (a block mean of spreads is hard to read). CLI:
`tables overviews --climatology`. Built and uploaded: CPC 25 MB, IMERG 71 MB, CAMS 362 MB, ERA5 2.5 GB.

## apps/climate-globe: ERA5 layer

One layer entry, a dropdown of 201 variables (`scales.json`, made by `build_scales.py` from the climatology
overviews: 2nd-98th percentile; K linear cold-hot; signed diverging around 0; skewed positive log; else linear).
Climatology mode reads `climatology_overviews/`; live mode reads `overviews/` at globe scale for the 8 variables
that have them, `series/` otherwise. Selected cell: normals from `climatology/`; no series button (no ERA5 by_cell).
Test `test_era5_variables_and_normals`; 6/6 app tests pass on the published data.

## ERA5-Monthly upload

Job 2106 finished: 2,085 files uploaded, verified. README example run as written: 74 s, July 2024 (5,113,745
cells, t2m 216-317 K) and Loch Lomond's normals (277.1 K in January, 287.5 K in July). App, live month of
`era5_swvl1` from `series/` (no overviews): 15.3 MB in 160 requests, 12 s with ERA5's cell positions.

## Decisions

- 1,024-cell row groups over 63-cell: footer of the largest ERA5 band ~9 MB, not ~130 MB.
- No value dictionaries: a dictionary page is read whole before any page of its column.
- `climatology_overviews/` as its own folder: `overviews/` is partitioned by `year=`.
- Old by_cell copies moved to `/RUBBISH/miko-bycell-tests-20261031`, not kept on /data (98% full).
- ERA5 by_cell not built: needs ~2.1 TB and the removal of ERA5-Monthly-v2 (2.79 TB) - Miko's call.
- Release-test README not uploaded: its generator links the current commit, which must be pushed first.

## Open

- ERA5 by_cell build: after removing ERA5-Monthly-v2 (needs approval).
- Push refactor/v1 (redeploys the climate globe Space with the ERA5 layer), then the release-test README.
- READMEs (top level, CPC, IMERG, CAMS, ERA5-Monthly): review, then publish.
- Climatology files have the same footer and dictionary costs as the old by_cell; same treatment possible.
- `era5_viten` reaches ±1.1e10 in the normals: still to check against ERA5's units.

## Look at first

- majortom/release/tables/bycell.py:68 (writer settings)
- apps/climate-globe/data.js (loadByCellSeries)
- docs/datasets/climate/README.md (How by_cell/ files are written)
