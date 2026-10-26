> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review (Miko and the TACO / rumi / GeoZL maintainer), 2026-10-24. Updated 2026-10-26 after the review meeting with Cesar (sections 2, 3, 5, 6, 7, 10). Design only; no code yet.

# Releasing Major TOM datasets — design

How built Major TOM samples become a published dataset: raster payloads (rumi proposed as default, readable as COG; under benchmark), TACO containers, and upload to source.coop (Hugging Face optional). A new module, `majortom.release`, with an extra `[release]`, alongside `majortom.build`. First target: a new version of Sentinel-2 Core on source.coop.

## 1. Flow

```
majortom.build output: one sample per cell (bands as arrays + JSON sidecar: CRS, transform, times, sources)
  -> raster groups by pixel size (Sentinel-2: 10 m, 20 m, 60 m)
  -> payload writers: rumi per group (proposed default) or COG per group
  -> TACO FOLDER: contract (structure + metadata levels) + samples
  -> partitions: ZIPs of at most a few GB, cut by size along a space-filling curve (section 7), one contract for all
  -> TACOCAT: consolidated metadata over the partitions, opened as one collection
  -> upload: source.coop (default), Hugging Face (option)
```

## 2. Decisions so far (Miko, 2026-10-24)

| Topic | Default | Options |
|---|---|---|
| Payload | **rumi, lossless** (Miko, 2026-10-26, after the benchmark in section 10; as Core-DEM) | COG: release with `payload="cog"` or `"both"`, read any rumi sample as a COG, convert a selection to COG files (section 5) |
| Compression | lossless | near-lossless, heavily lossy (section 4) |
| Container | TACO ZIP partitions + TACOCAT; built as FOLDER | — |
| Partitions | by size: at most ~2 GB per ZIP, any modality (section 7) | — |
| Raster groups | one file per pixel size | — |
| Destination | source.coop | Hugging Face |
| zarr | not used for sample releases | data cubes (many dates per cell), later |
| Climate datasets | plain partitioned parquet, not TACO | — |
| Overviews | none | on request (map viewers) |

## 3. COG parameters

Chosen for training reads, not only for viewing. Sample size is 1056 px at 10 m (10,560 m windows, docs/SPEC.md).

| Setting | Value | Reason |
|---|---|---|
| Tiles | 176 x 176 px at 10 m (6 x 6 per sample) and 20 m (3 x 3); the 60 m image (176 px) is one tile | TIFF tiles must be multiples of 16 (not 32): GDAL wrote 176 and rejected 168 and 264 ("Bad value for TileWidth"), tested 2026-10-26. The sizes that divide 1056 are 96, 176, 352, 528. For random 256 px training crops, 176 decodes ~2.8x the crop on average (96: ~1.9x with 121 tiles per sample; 352: ~5.6x) |
| Interleave | tile (`INTERLEAVE=TILE`: all bands of a tile stored next to each other, each band compressed on its own) | Cesar's suggestion (2026-10-26): training reads all bands of a tile in one range request, and single-band reads still work, with better compression than pixel interleave. Needs GDAL >= 3.11; GDAL 3.10 silently writes pixel interleave instead (tested) |
| Compression | ZSTD, horizontal predictor (floating-point predictor for floats) | fast decode, good ratio for 16-bit EO data |
| Overviews | none | training reads full resolution; a full pyramid adds about a third |
| Data type | as delivered (Sentinel-2: uint16) | no conversion |
| Metadata | band names, nodata, scale and offset (e.g. Sentinel-2 L2A: -1000 offset, 1e-4 scale), CRS, transform with the corrected origin | values interpretable without the sidecar |

Tiles of the 10 m and 20 m groups cover different ground (1.76 km and 3.52 km). Equal ground per tile across 10, 20 and 60 m would need 96 px tiles at 10 m (16 px at 60 m).

## 4. Compression levels

| Level | COG | rumi (GeoZL) |
|---|---|---|
| Lossless (default) | ZSTD + predictor | lossless graph chosen by `geozl.profile` per raster group |
| Near-lossless | LERC with a maximum error of 1-2 DN (or LERC_ZSTD) | `error=1` or `error=2` (absolute bound, `quant_linear`) |
| Heavily lossy | JPEG XL lossy (16-bit, recent GDAL) | large bounds: relative (`error="5%"`, `quant_log`) or noise-scaled (`quant_sqrt`) |

GeoZL's lossy modes are error-bounded quantisation at the head of the graph, then lossless stages; there is no transform codec. So "heavily lossy" in rumi guarantees a per-pixel bound, while JPEG XL in COG does not. The dataset records the level and its bound in collection metadata.

## 5. rumi

**Decision (Miko, 2026-10-26): rumi lossless is the default payload**, as in Core-DEM (`https://source.coop/major-tom/core-dem`: `dem.rumi` per sample, read with `taco.open_dataset(...)` then `rumi.read(location, header)`). The benchmark (section 10) found it 19% smaller than the best COG, ~5.5x faster to write and 1.8x faster for training crops. Sentinel-2 bands stay lossless (Core-DEM uses a 2 m bound); near-lossless and lossy remain options (section 4).

COG stays a first-class option, three ways:

| Need | How | Result |
|---|---|---|
| Publish (or rebuild) a release as COGs | `release(samples, out, payload="cog")`, or `payload="both"` for rumi and COG side by side | the same TACO layout, `cog/<group>.tif` instead of (or beside) `rumi/<group>.rumi`, section 3's COG parameters |
| Open one rumi sample in GDAL-based tools | `majortom.release.read(sample, as_="cog")` | an in-memory GeoTIFF (rasterio `MemoryFile`) with CRS, geotransform, band names, scale and offset from the rumi header |
| Get COG files for part of a rumi release (e.g. for QGIS) | `majortom.release.to_cog(dataset, out, where="...")`, or `python -m majortom.release to-cog <dataset> <out> --where "..."` | one COG per sample and group, section 3's parameters, pixel for pixel equal to the rumi values (lossless) |

Reading, both ways, in the dataset card and the README:

```python
import taco, rumi
ds = taco.open_dataset("https://data.source.coop/major-tom/<dataset>")
rows = ds.sql("SELECT id, \"rumi/10m.rumi::location\" AS loc, \"rumi/10m.rumi::header\" AS hdr FROM ... LIMIT 8")
array = rumi.read(rows.loc[0], rows.hdr[0])                         # (bands, y, x) numpy

from majortom.release import read
with read(rows.iloc[0], as_="cog") as cog, cog.open() as f:        # any rasterio/GDAL code
    print(f.crs, f.transform, f.descriptions)
```

rumi is GPL-3.0 and pre-1.0: the release records the rumi and GeoZL versions in its collection metadata, and `to_cog` gives a stable, permissive copy of any part of it.

- One `.rumi` file per raster group per sample, beside or instead of the COG.
- Frames: tile size chosen by profiling (frames are not bound to multiples of 16, so 256-friendly sizes are possible, e.g. 264 = 1056 / 4).
- Pattern: all bands of a tile in one frame (`b (row h) (col w) -> row col (b h w)`), matching COG's pixel interleave.
- Time: one step per sample (the acquisition time); cubes later.
- Header stored per file in TACO's `children/rumi` level, as in the workshop notebook.

## 6. TACO contract

```
structure:
  cog/<group>.tif           e.g. cog/sentinel2-l2a-10m.tif
  rumi/<group>.rumi         when rumi is chosen
metadata levels:
  sample:         STAC (time, footprint), MajorTOM (grid_cell, row, col, d=10 km), GeoEnrich (optional),
                  majortom:* from the build sidecar: product id, archive, processing baseline, offset, scaling,
                  cloud share, nodata share, platform; optionally Core-Weather step 0 (weather at acquisition)
  children/rumi:  Rumi header and statistics
  children/cog:   none (the path is enough)
collection: COLLECTION.json — description, licence and attribution, compression level, Major TOM spec version
```

- Follow Core-DEM's layout so all Major TOM releases read the same way: metadata groups `spatial` (`crs`, `tensor_shape`, `geotransform`, `centroid`), `majortom` (`code`, `partition`), `geoenrich`, and `rumi:header` / `rumi:stats` per file.
- Sample id: Core-DEM's cell code plus the acquisition time, e.g. `MT10km_0624U_0029L_20210714T112119` (Core-DEM: `MT10km_0104D_0845L`; one DEM per cell, many Sentinel-2 dates). Unique across partitions.
- Added for time series: `stac:datetime` (acquisition), and `majortom:*` from the build sidecar.

## 7. Partitions

Core-DEM uses a fixed 3000 km shard: 63 partitions of ~0.4 GB on average (24 GB in all). A fixed shard only fits one sample size: Sentinel-2 L2A at ~50-100 MB per sample and ~2.2 million samples would give ZIPs of over a terabyte. Partitions are therefore cut by size, the same way for every modality:

```
samples (grid cell, time, bytes as written in the FOLDER)
  -> group by grid cell: all dates of a cell stay in one partition
  -> order the cells along a Hilbert curve over their centroids (neighbouring cells stay close in the order,
     also across grid rows and zones)
  -> walk the order, adding cells until the next one would pass the target (default 2 GB);
     a cell larger than the target on its own is its own partition
  -> partition id (p0000, p0001, ...) in majortom:partition; the TACOCAT records each partition's bounding box
     and first/last cell code
```

- Sizes stay within the target whatever the bytes per sample (DEM, Sentinel-2, a time series, a coarser grid).
- A regional query touches a few neighbouring partitions, as with fixed shards.
- Deterministic for a given sample set. A later release with more samples moves the boundaries: releases are versioned and not edited in place.
- Rejected: fixed shards (size depends on modality); grid-row bands (very uneven sizes, long thin areas); recursive splitting of coarse Major TOM cells (meaningful names, but leaves many tiny partitions over sparse areas such as oceans and islands).

## 8. Publishing

- **source.coop (default):** S3-compatible upload of the partitions and the TACOCAT, with credentials from `majortom.build.credentials` (a new `source-coop` entry, owner-only file, as for the other archives).
- **Hugging Face (option):** `huggingface_hub` upload of the same files.
- Resumable: a partition already uploaded with the same size and hash is skipped.

## 9. Module layout (proposed)

```
majortom/release/
  __init__.py    release(samples, out, payload="rumi" | "cog" | "both", level="lossless", ...) -> TACOCAT path
  groups.py      raster groups by pixel size, per product (from majortom.build.collections)
  cog.py         write one COG per group (section 3, levels from section 4)
  rumi.py        write one .rumi per group
  read.py        read(sample, as_="array" | "xarray" | "cog"), to_cog(dataset, out, where) (section 5)
  partition.py   size-targeted partitions along a Hilbert curve (section 7)
  taco.py        contract, samples, partitions, TACOCAT (taco package, own repository)
  publish.py     source.coop and Hugging Face upload, resumable
  __main__.py    python -m majortom.release build | publish | to-cog
```

Dependencies (`[release]`): `taco-eo[writer]`, `rumi-eo` (GPL-3.0), `geozl` (BSD-3-Clause), `huggingface_hub`, an S3 client for source.coop, and GDAL >= 3.11 for tile-interleaved COGs (rasterio 1.5 wheels, which need Python >= 3.12, or a conda GDAL). rumi and GeoZL need Python 3.11 or newer; the package moved to 3.11 on 2026-10-25.

## 10. Benchmark: rumi or COG as the default (run 2026-10-26)

Script: `docs/evidence/release_bench.py`; results in `/data/Major-TOM/release-bench/full/` (`results.csv`, `crops_1.csv`, `crops_8.csv`). Environment: Python 3.12, rasterio 1.5.2 with GDAL 3.12.2, rumi-eo 0.26.1, GeoZL 0.18.0, on `cirrus` (files on the NVMe RAID, warm in the page cache).

**Data.** 200 random Core-S2L2A samples (seeded, no missing pixels) cut to the v2 window with `crop_core_to_v2`: not a centre crop, since the v2 window sits at a different offset in each Core sample; 206 samples were skipped because their v2 window does not fit inside. Latitudes −74° to 65°. Three groups by pixel size: 10 m (B02 B03 B04 B08, 1056 px), 20 m (B05 B06 B07 B8A B11 B12, 528 px), 60 m (B01 B09, 176 px).

**Formats.** COG: ZSTD, horizontal predictor, 176 px tiles, no overviews, pixel or tile interleave. rumi: 176 px frames holding all bands of a tile (`b (row h) (col w) -> row col (b h w)`), GeoZL graph chosen by `geozl.profile` per group (`planar>zigzag>transpose>entropy` for 10 m and 20 m), lossless or with an absolute bound of 1 DN.

**Size and whole-sample speed** (all three groups, single thread):

| Format | Ratio | Size | Write | Full read | Lossless |
|---|---|---|---|---|---|
| COG, pixel interleave | 1.63 | 1,523 MB | 37.9 s | 4.4 s | yes |
| COG, tile interleave | 1.59 | 1,556 MB | 35.7 s | 5.6 s | yes |
| rumi lossless | 2.01 | 1,232 MB | 6.8 s | 3.8 s | yes |
| rumi, 1 DN bound | 2.25 | 1,101 MB | 11.5 s | 4.4 s | no (max error 1) |

**Training-style crops.** 20 batches of 32 random 256 px crops of the 10 m group, each crop from a different sample, opening each file as a data loader does:

| Format | Read | 1 thread | 8 threads | 8 processes |
|---|---|---|---|---|
| COG, pixel interleave | open + window | 261 crops/s | 154 | 1,428 |
| COG, tile interleave | open + window | 219 | 156 | 1,189 |
| rumi lossless | `read_many` per batch | 444 | 2,571 | — |
| rumi lossless | single `read` | 482 | 501 | — |
| rumi, 1 DN bound | `read_many` per batch | 362 | 2,243 | — |

"8 processes" is a pool of 8 worker processes started once, as PyTorch DataLoader workers; COG in 8 threads of one process is slower than in one thread. rumi's `read_many` runs its own thread pool inside one process.

**Findings.**

- Lossless rumi is 19% smaller than the best COG, ~5.5x faster to write, slightly faster to read whole, and 1.8x faster for training crops (2,571 against 1,428 crops/s with 8 workers).
- The 1 DN bound saves another 11% and gives up exactness: not worth it for Sentinel-2.
- Tile interleave was 2% larger and slower than pixel interleave for reads of all bands. Its advantage, reading a single band, was not tested.
- Supports rumi lossless as the default (as Core-DEM), with the COG read path and converter (section 5). To decide with Cesar.

**Not covered.** Cold reads from source.coop over HTTP (range requests, latency); GPU decode; reads of single bands; other modalities (Sentinel-1 float32, DEM int16).

## 11. Questions for the TACO / rumi / GeoZL maintainer

Answered 2026-10-26: COG interleave (tile); COG tile size (multiples of 16; 176 works); reference layout (Core-DEM); payload (rumi lossless by default, COG as an option: section 5).


1. TACO partitions: recommended partition size, and is partitioning by grid row band sensible for a global dataset?
2. Is a COG payload in TACO handled as an opaque asset, or does TACO offer COG-aware metadata (like `taco.extensions.Rumi` for rumi)?
3. rumi frame size for 1056 px samples and 256 px training crops: 264 (1056 / 4), 352, or other?
4. GeoZL near-lossless and lossy defaults for Sentinel-2 uint16: sensible bounds, and is the wire format stable enough for a long-lived release?
5. rumi before 1.0: should Core v2 ship rumi now (regenerable from COG later), or wait for 1.0?
6. source.coop upload: preferred client and any TACO-specific publishing steps (e.g. TACOCAT placement)?
