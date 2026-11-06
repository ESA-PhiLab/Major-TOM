> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (added 2026-11-05).

# Building new samples: majortom.build

Needs `majortom[build]` (`majortom[gee]` for Earth Engine). Full product and archive tables:
[docs/ARCHIVES.md](https://github.com/ESA-PhiLab/Major-TOM/blob/refactor/v1/docs/ARCHIVES.md).

    cells (from points or a box)  ->  per sample: window -> search -> pick a scene -> read -> write

## Call

```python
from majortom import build
cells = build.cells_from_points(points, d=10)      # or build.cells_in((west, south, east, north), d=10)
errors = build.download(cells, "sentinel-2-l2a", "2024-06-01/2024-08-31",
                        archive="planetary-computer", prefer="clearest", cloud=(0, 0.05), outdir="chips")
```

Output per sample: `chips/<collection>/<index>.npz` (one array per band) and a `.json` sidecar (cell, window, CRS,
source items, archive, dates, cloud share and how it was measured). `download` returns the samples that failed,
with the reason. Re-running reuses samples built with the same request.

## Arguments

| Argument | Meaning |
|---|---|
| `datetime` | `"start/end"` or one time; selects the version for yearly products; ignored for static ones |
| `days` | widen a single `datetime` by ± days |
| `prefer` | order scenes are tried: `"clearest"`, `"cloudiest"`, `"nearest"`, `"newest"` |
| `cloud` | accepted cloudy share of the window, `(low, high)`; `None` for any |
| `max_nodata` | accepted share of the window without data (default 0.01) |
| `strict` / `fallback` | no acceptable scene: fail (`strict=True`), or use the best tried: `"complete"` or `"clear"` |
| `platform` | e.g. `"terra"`, `["landsat-8", "landsat-9"]` |
| `window` | `"v2"` (default, 1056 px centred) or `"v1"` (lines up with Core v1 samples) |
| `workers`, `force` | parallel samples; rebuild everything |

Per-row overrides: columns `datetime`, `days`, `margin` in `cells`.

## Archives

| `archive` | Account | Notes |
|---|---|---|
| `planetary-computer` | none | default, most products |
| `earth-search` | none | Sentinel-2 L1C, Collection 1 L2A |
| `cdse` | S3 keys | ESA's archive, what Core was built from |
| `earthdata` | NASA token | LP DAAC, PO.DAAC, ASF |
| `earth-engine` | service account or user login | uses the Earth Engine quota of the project |

Credentials: `python -m majortom.build.credentials <cdse|earthdata|earth-engine|cds|edh>` prompts with hidden
input and writes owner-only files, or set the environment variables it names. Do not ask users to paste keys
into a chat, and never print them.

## Archives compared

Same processing, same pixels: Sentinel-2 L2A, Landsat L2, HLS and Copernicus DEM samples are identical across the
archives that hold them, once each sample's `offset` is removed (docs/ARCHIVES.md section 5). Speed (2026-10-22,
UK server, Sentinel-2 L2A samples per hour): Planetary Computer 1,177, Earth Search 384, Earth Engine 215, CDSE 207
(docs/reports/2026-10-22-build-speed.md).

## Adding a product or archive

One `Collection` in `majortom/build/collections.py` or one `Archive` in `archives.py`, test dates in
`tests/test_products.py`, then `python -m majortom.build.catalogue`: CONTRIBUTING.md.

## Cost

Each sample is a STAC search plus windowed reads of cloud-optimised files: seconds per sample on most archives
(Sentinel-3 swaths about 24 s). Earth Engine use counts against the project's quota: test on a few cells first.
