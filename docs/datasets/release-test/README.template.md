# Major TOM release test

> **Not for use.** A test of the Major TOM release framework (`majortom.release`), published to try the whole path,
> from building samples to reading them back from source.coop, before the first real release
> (Sentinel-2 L2A, Major TOM v2). Every number on this page is generated from the data it describes.

{{n_sources}} sources, {{n_samples}} samples, {{size_gb}} GB, each source its own TACO dataset
in its own folder, made exactly as a real release would be (`python -m majortom.release build`).

## Where

<img src="https://data.source.coop/major-tom/release-test/figures/cells.png" alt="The test cells" width="100%">

Three cells of the Major TOM 10 km grid: **Loch Lomond** (`624U_29L`), the **Dolomites near Corvara** (`518U_90R`)
and the **Amazon near Manaus** (`35D_667L`). {{fallback}} Per cell: two dates for scene products (the clearest of
May-June and of August-September 2023), two years for yearly products (2020, 2021), one sample for static products.
Fewer samples mean the archive had nothing for that cell and period.

## What

<img src="https://data.source.coop/major-tom/release-test/figures/gallery.png" alt="One quicklook per source" width="100%">

| Source | Kind | Pixel sizes | Bands | Data types | Time | Archive | Samples | Released (all files) | Compression (rumi) | Partitions | Check | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{{sources_table}}

"Compression" is the raw size of the arrays over the size of their rumi files (lossless, GeoZL); "Released" also
counts TACO's catalogue in every partition, which dominates for the smallest sources. Below 1, the data are
smaller than the rumi file's own header (MODIS 1 km: 12 x 12 pixels, ~300 bytes per image). "Check": `taco.validate` passes
and every file reads back identical to the built sample. "Code": the source's definition in the builder (bands,
cloud mask, offsets, resolution) and the catalogue it was read from.

## Access points, quotas and speeds

| Archive | Account or key | Quota | Access | Sources | Build time per sample (median) | Built sample, uncompressed (median) |
|---|---|---|---|---|---|---|
{{access_table}}

Per source (medians over its samples; fastest first):

| Source | Archive | Samples | Build time | Search | Scene choice | Reads | Mostly | Built sample, uncompressed |
|---|---|---|---|---|---|---|---|---|
{{per_source_table}}

Build times are measured per sample (search, scene choice, reads of every band; `timing` in each sample's sidecar),
built with 3 workers from `cirrus` on a ~63 MB/s link shared with other large downloads at the time: they show which
sources are slow and why, not their best speed (Sentinel-2 L2A took 97 s per sample here, 18 s in a quieter
benchmark with 16 workers, [docs/reports/2026-10-22-build-speed.md]({{repo}}/docs/reports/2026-10-22-build-speed.md)).
Notes from building Major TOM: Earth Engine bills reading stored images far above what its profiler reports
(ERA5-Land hourly: ~30x); Destination Earth's Earth Data Hub serves ERA5-Land as zarr with a free quota of
500,000 requests a month (one request per 60-day chunk).

## Reading it from source.coop: measured

Opened and read from `https://data.source.coop/major-tom/release-test/<source>/` without credentials, from `cirrus`
(UK); the data are stored in AWS eu-central-1 (Frankfurt). "Open": `taco.open_dataset` and the catalogue query;
"read": one file of one sample with `rumi.read` over HTTP (byte ranges inside the partition ZIP).

| Source | Open | Read (median) | Read (max) | Identical to build |
|---|---|---|---|---|
{{remote_table}}

{{remote_note}}

## Reading it

```python
import taco, rumi
ds = taco.open_dataset("https://data.source.coop/major-tom/release-test/sentinel-2-l2a/")
rows = ds.sql('SELECT id, "10m.rumi::location" AS loc, "10m.rumi::header" AS hdr FROM dataset').to_pylist()
array = rumi.read(rows[0]["loc"], rows[0]["hdr"])               # (bands, y, x), only the bytes of this sample

from majortom.release import read                                # the same sample as a GeoTIFF, for any GDAL tool
with read(rows[0]["loc"], rows[0]["hdr"], as_="cog") as mem, mem.open() as f:
    print(f.crs, f.transform, f.descriptions)
```

Whole datasets or selections as COG files: `python -m majortom.release to-cog --dataset <url> --out <folder>
--where "<SQL condition>"`. Each sample also carries STAC time and footprint, its Major TOM cell (`majortom:code`),
its source products (`source:*`) and environmental attributes (`geoenrich:*`).

## How this was made

`majortom.build` (one window per cell, Major TOM v2) -> `majortom.release`: raster groups by pixel size and data
type -> lossless rumi files (GeoZL, all predictor families tried) -> samples written in Hilbert order over cell
centroids -> taco-eo partitions by size and writes the TACOCAT -> published with `python -m majortom.release
publish`. All links point to the Major TOM repository at commit `{{commit_short}}`.

| Code | What it does |
|---|---|
| [`docs/evidence/release_test_dataset.py`]({{repo}}/docs/evidence/release_test_dataset.py) | this test: every source at the three cells (`build`), one TACO dataset per source (`release`), the read-back check (`check`), the upload (`publish`) |
| [`docs/evidence/release_test_readme.py`]({{repo}}/docs/evidence/release_test_readme.py), [`README.template.md`]({{repo}}/docs/datasets/release-test/README.template.md) | this page: the text is the template, the tables and figures are computed |
| [`majortom/build/`]({{repo}}/majortom/build) | the builder: `download(cells, collection, archive=...)` searches an archive, chooses a scene per cell and reads its window |
| [`majortom/build/collections.py`]({{repo}}/majortom/build/collections.py) | one definition per source: bands, cloud mask, offsets, pixel size, time ("Code" column above) |
| [`majortom/build/archives.py`]({{repo}}/majortom/build/archives.py), [`backends/`]({{repo}}/majortom/build/backends) | the catalogues and how each is read: STAC with GDAL, or Earth Engine |
| [`majortom/release/`]({{repo}}/majortom/release) | rumi and COG files, raster groups, Hilbert order, TACO packaging, publishing |
| [`docs/RELEASE.md`]({{repo}}/docs/RELEASE.md) | design and measurements behind the release format |

Building one source for any cell:

```python
from majortom.build import download
from majortom.cells import _table, cell_at

cells = _table([cell_at(-4.62, 56.08)])                         # Loch Lomond
download(cells, "sentinel-2-l2a", datetime="2023-05-01/2023-06-30", archive="planetary-computer",
         outdir="loch-lomond")                                   # samples as .npz with a JSON sidecar each
```
