> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review, 2026-10-30.

# majortom.release

Turns built data into published Major TOM datasets: the file formats and layouts, and the upload. Other packages
produce data ([majortom.build](../build/README.md): image samples; [majortom.climate](../climate/README.md): tables per cell); this one decides how it is
stored and published. Needs `pip install 'majortom[release]'`.

## Two kinds of data

```
image samples (majortom.build)
  -> raster groups by pixel size -> rumi (default, lossless) and/or COG per group
  -> TACO dataset: samples in Hilbert order, cut into ZIP partitions of just under 3 GB, + TACOCAT
  -> publish

tables keyed by grid_cell, one file per month (majortom.climate)
  -> tables/layout.py:    rows along the Hilbert curve, small row groups          series/, climatology/
  -> tables/overviews.py: 4 x 4 block means per month, for globe-wide maps       overviews/
  -> tables/bycell.py:    every cell's whole series together, by band             by_cell/
  -> publish
```

The monthly files suit one month over the globe; `by_cell` suits one place over time (an app's chart, training on
series); `overviews` suit a map of the whole globe from one small file. See
[the read experiments report](../../docs/reports/2026-10-30-climate-globe-read-experiments.md) for the measurements behind these layouts.

## Modules

| Module | What it does |
|---|---|
| `groups.py` | one built sample -> its raster groups (bands that share a pixel size) |
| `rumi.py`, `profiles.py` | a raster group as a `.rumi` file, with a compression profile (lossless by default) |
| `cog.py` | the Major TOM COG profile |
| `taco.py` | built samples -> a TACO dataset (partitions + TACOCAT), validated |
| `order.py` | the Hilbert curve order, shared by samples and tables |
| `read.py` | released rumi samples as arrays or COGs |
| `tables/` | layouts for tables keyed by `grid_cell` (above) |
| `publish.py` | upload a release folder: source.coop by default, Hugging Face as an option; resumable, checked |

## Command line

```
python -m majortom.release build   --samples <build folder> --out <out.zip> --id <id> --description <text> ...
python -m majortom.release publish --folder <release folder> --to source.coop://<account>/<repository>[/<prefix>]
python -m majortom.release to-cog  --dataset <folder, .zip or URL> --out <folder> [--where "<SQL condition>"]
python -m majortom.release tables repack    --src <dataset folder> --out <folder>
python -m majortom.release tables overviews --src <dataset folder> --out <dataset folder> [--level 2]
python -m majortom.release tables by-cell   --src <dataset folder> --out <dataset folder> [--workers 4]
```

## More

- [docs/RELEASE.md](../../docs/RELEASE.md): design, decisions and benchmarks for image samples
- [docs/CLIMATE.md](../../docs/CLIMATE.md): the climate tables and where they are published
