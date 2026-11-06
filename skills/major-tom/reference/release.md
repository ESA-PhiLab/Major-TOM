> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (added 2026-11-06).

# Releasing and publishing: majortom.release

Needs `majortom[release]` (rumi-eo, taco-eo, boto3, huggingface_hub; Python 3.12 or later for the COG profile).
Design and benchmarks: [docs/RELEASE.md](https://github.com/ESA-PhiLab/Major-TOM/blob/refactor/v1/docs/RELEASE.md).

    built samples (.npz + .json, one product)
      -> raster groups (bands with one pixel size and type) -> rumi files (lossless by default)
      -> samples in Hilbert order -> TACO partitions (<= 3 GB) + .tacocat catalogue -> publish

## release(samples, out, ...)

| Argument | Meaning |
|---|---|
| `samples` | the `.npz` files of one product (all with the same raster groups) |
| `out` | a `.zip` path (partitions and `.tacocat` beside it) or a folder |
| `id`, `title`, `description`, `licenses`, `providers`, `keywords` | catalogue entry |
| `profile` | `"lossless"` (default), `"near-lossless-1"`, `"near-lossless-2"` |
| `max_bytes` | partition size limit, default 3 × 10⁹ bytes |
| `with_time` | `False` for one sample per cell (DEMs): ids are the cell alone |
| `stats` | per-file band statistics (default `True`) |
| `geoenrich` | taco-eo GeoEnrich attributes: `True` (default), a list of names, or `False`; skipped with a warning offline |
| `stage`, `overwrite`, `progress` | working folder, replace, progress bar |

Sample ids: `MT10km_<row>_<col>_<time>`, e.g. `MT10km_0624U_0029L_20240603T114351` (`majortom.release.sample_id`).

## Reading

- `taco.open_dataset(path or URL)`; `.sql("SELECT ... FROM dataset")` (one row per file) or `FROM sample`.
- Columns: `id`, `majortom:code` (cell), `<group>.rumi::location`, `<group>.rumi::header`, `geoenrich:*`, `source:*`.
- `majortom.release.read(location, header)` -> array (bands, y, x); `as_="cog"` -> in-memory COG (rasterio
  MemoryFile).
- `taco.export(source, out, sql=...)` copies a selection into a new dataset (only selected files are fetched).

## publish(folder, destination, ...)

| Destination | Credentials |
|---|---|
| `source.coop://<account>/<repository>[/<prefix>]` | `python -m majortom.build.credentials source-coop` (a repository API key) |
| `hf://<user or organisation>/<repository>` | `huggingface-cli login` or `HF_TOKEN` |

Arguments: `dry_run`, `include` (other files by glob, e.g. `["README.md", "figures/*.png"]`), `workers`, `force`.
Resumable (same size and sha256 are skipped), every file checked after upload. Publishing is outward-facing:
confirm with the user before running it without `dry_run`.

## Command line

    python -m majortom.release build   --samples <folder> --out <out.zip> --id <id> --description <text>
    python -m majortom.release publish --folder <folder> --to source.coop://<account>/<repository>
    python -m majortom.release to-cog  --dataset <folder, .zip or URL> --out <folder> [--where "<SQL>"]
