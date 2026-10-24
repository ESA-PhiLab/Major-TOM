> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review (Miko and the TACO / rumi / GeoZL maintainer), 2026-10-24. Design only; no code yet.

# Releasing Major TOM datasets — design

How built Major TOM samples become a published dataset: raster payloads (COG by default, rumi for training), TACO containers, and upload to source.coop (Hugging Face optional). A new module, `majortom.release`, with an extra `[release]`, alongside `majortom.build`. First target: a new version of Sentinel-2 Core on source.coop.

## 1. Flow

```
majortom.build output: one sample per cell (bands as arrays + JSON sidecar: CRS, transform, times, sources)
  -> raster groups by pixel size (Sentinel-2: 10 m, 20 m, 60 m)
  -> payload writers: COG per group (default) and/or rumi per group (recommended for training)
  -> TACO FOLDER: contract (structure + metadata levels) + samples
  -> partitions: ZIP per N samples (built from the FOLDER), one contract and collection metadata for all
  -> TACOCAT: consolidated metadata over the partitions, opened as one collection
  -> upload: source.coop (default), Hugging Face (option)
```

## 2. Decisions so far (Miko, 2026-10-24)

| Topic | Default | Options |
|---|---|---|
| Payload | COG | rumi, advised for AI training; both side by side |
| Compression | lossless | near-lossless, heavily lossy (section 4) |
| Container | TACO ZIP partitions + TACOCAT; built as FOLDER | — |
| Raster groups | one file per pixel size | — |
| Destination | source.coop | Hugging Face |
| zarr | not used for sample releases | data cubes (many dates per cell), later |
| Climate datasets | plain partitioned parquet, not TACO | — |
| Overviews | none | on request (map viewers) |

## 3. COG parameters

Chosen for training reads, not only for viewing. Sample size is 1056 px at 10 m (10,560 m windows, docs/SPEC.md).

| Setting | Value | Reason |
|---|---|---|
| Tiles | 176 x 176 px at 10 m (6 x 6 per sample) and 20 m (3 x 3); the 60 m image (176 px) is one tile | GeoTIFF tiles are multiples of 16; the sizes that divide 1056 are 96, 176, 352, 528. For random 256 px training crops, 176 decodes ~2.8x the crop on average (96: ~1.9x with 121 tiles per sample; 352: ~5.6x) |
| Interleave | pixel (all bands of a tile together) | training reads all bands of a group: one range request per tile |
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

- Sample id: the grid cell plus the acquisition (e.g. `624U_29L_20210714T112119`), unique across partitions.
- Partitions: by grid row band (as `docs/CLIMATE.md` does for parquet), so a region's samples share partitions; target size of a few tens of GB per ZIP (to confirm).

## 7. Publishing

- **source.coop (default):** S3-compatible upload of the partitions and the TACOCAT, with credentials from `majortom.build.credentials` (a new `source-coop` entry, owner-only file, as for the other archives).
- **Hugging Face (option):** `huggingface_hub` upload of the same files.
- Resumable: a partition already uploaded with the same size and hash is skipped.

## 8. Module layout (proposed)

```
majortom/release/
  __init__.py    release(samples, out, payload="cog", rumi=False, level="lossless", ...) -> TACOCAT path
  groups.py      raster groups by pixel size, per product (from majortom.build.collections)
  cog.py         write one COG per group (section 3, levels from section 4)
  rumi.py        write one .rumi per group (section 5)
  taco.py        contract, samples, partitions, TACOCAT (taco package, own repository)
  publish.py     source.coop and Hugging Face upload, resumable
  __main__.py    python -m majortom.release build | publish
```

Dependencies (`[release]`): `taco-eo[writer]`, `rumi-eo` (GPL-3.0), `geozl` (BSD-3-Clause), `huggingface_hub`, an S3 client for source.coop. rumi and GeoZL need Python 3.11 or newer; the package moves from 3.10 to 3.11.

## 9. Questions for the TACO / rumi / GeoZL maintainer

1. TACO partitions: recommended partition size, and is partitioning by grid row band sensible for a global dataset?
2. Is a COG payload in TACO handled as an opaque asset, or does TACO offer COG-aware metadata (like `taco.extensions.Rumi` for rumi)?
3. rumi frame size for 1056 px samples and 256 px training crops: 264 (1056 / 4), 352, or other?
4. GeoZL near-lossless and lossy defaults for Sentinel-2 uint16: sensible bounds, and is the wire format stable enough for a long-lived release?
5. rumi before 1.0: should Core v2 ship rumi now (regenerable from COG later), or wait for 1.0?
6. source.coop upload: preferred client and any TACO-specific publishing steps (e.g. TACOCAT placement)?
