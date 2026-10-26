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
| Payload | open: rumi (as Core-DEM) with a COG read path and converter (section 5), or COG; decided by the benchmark (section 10) | both side by side (about double the storage) |
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

Core-DEM (`https://source.coop/major-tom/core-dem`) is published as rumi only: `dem.rumi` per sample, GeoZL `med>zigzag>transpose>entropy` with a 2 m error bound, read with `taco.open_dataset(...)` then `rumi.read(location, header)`. Sentinel-2 Core v2 follows it unless the benchmark (section 10) shows lossless rumi gains little over COG. Proposal discussed 2026-10-26: rumi as the default payload, with a COG path for everything that expects GeoTIFF:

- `majortom.release.read(sample, as_="array" | "xarray" | "cog")`: `"cog"` returns an in-memory GeoTIFF (rasterio `MemoryFile`) with CRS, geotransform, band names, scale and offset from the rumi header, so GDAL-based tools open it unchanged.
- `majortom.release.to_cog(dataset, out, ...)`: converts a selection (SQL over the TACOCAT) to COG files with section 3's parameters, for QGIS or other GIS use.
- Sentinel-2 bands are kept lossless by default (unlike Core-DEM's 2 m bound); near-lossless and lossy remain options (section 4).

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
  __init__.py    release(samples, out, payload="cog", rumi=False, level="lossless", ...) -> TACOCAT path
  groups.py      raster groups by pixel size, per product (from majortom.build.collections)
  cog.py         write one COG per group (section 3, levels from section 4)
  rumi.py        write one .rumi per group; read(as_="cog"), to_cog() (section 5)
  partition.py   size-targeted partitions along a Hilbert curve (section 7)
  taco.py        contract, samples, partitions, TACOCAT (taco package, own repository)
  publish.py     source.coop and Hugging Face upload, resumable
  __main__.py    python -m majortom.release build | publish
```

Dependencies (`[release]`): `taco-eo[writer]`, `rumi-eo` (GPL-3.0), `geozl` (BSD-3-Clause), `huggingface_hub`, an S3 client for source.coop, and GDAL >= 3.11 for tile-interleaved COGs (rasterio 1.5 wheels, which need Python >= 3.12, or a conda GDAL). rumi and GeoZL need Python 3.11 or newer; the package moved to 3.11 on 2026-10-25.

## 10. Benchmark: rumi or COG as the default (planned 2026-10-26)

Real Core-S2L2A samples cut to the v2 window with `crop_core_to_v2` (not a centre crop: the v2 window sits at a different offset in each Core sample, and fits inside only about half of them, so the set leans towards low latitudes; its latitude spread is reported), grouped by pixel size (10 m: 4 bands, 20 m: 6, 60 m: 2). Per format: bytes on disk, full-sample decode time, and random 256 px crops per second (all bands of the 10 m group), single process.

| Format | Settings |
|---|---|
| COG | ZSTD, horizontal predictor, 176 px tiles, tile interleave (GDAL >= 3.11) |
| rumi, lossless | GeoZL graph from `geozl.profile`, frames chosen by rumi |
| rumi, near-lossless | absolute error bound 1 DN |

## 11. Questions for the TACO / rumi / GeoZL maintainer

Answered 2026-10-26: COG interleave (tile); COG tile size (multiples of 16; 176 works); reference layout (Core-DEM). Open: rumi or COG as default (benchmark, section 10).


1. TACO partitions: recommended partition size, and is partitioning by grid row band sensible for a global dataset?
2. Is a COG payload in TACO handled as an opaque asset, or does TACO offer COG-aware metadata (like `taco.extensions.Rumi` for rumi)?
3. rumi frame size for 1056 px samples and 256 px training crops: 264 (1056 / 4), 352, or other?
4. GeoZL near-lossless and lossy defaults for Sentinel-2 uint16: sensible bounds, and is the wire format stable enough for a long-lived release?
5. rumi before 1.0: should Core v2 ship rumi now (regenerable from COG later), or wait for 1.0?
6. source.coop upload: preferred client and any TACO-specific publishing steps (e.g. TACOCAT placement)?
