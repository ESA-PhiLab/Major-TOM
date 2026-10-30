> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** draft for review, 2026-10-30.

# majortom.build

Builds Major TOM samples from satellite archives: for each grid cell, find a scene, read its bands onto the cell's
window, write one sample. Planetary Computer is the default archive. Needs `pip install 'majortom[build]'`.

## Flow

```
cells (from points or an area)
  -> per cell: window (pixel grid of each band)
  -> search the archive for scenes over the window and time range
  -> pick a scene (order: clearest, cloudiest, nearest, newest; accepted cloud share; fallback)
  -> read every band onto the window (tiles stitched; masks first)
  -> chips/<collection>/<index>.npz (one array per band) + .json sidecar (georeferencing, scene, choices)
```

Built samples go to `majortom.release` to become a published dataset.

## Modules

| Module | What it does |
|---|---|
| `__init__.py` | `cells_from_points`, `cells_in`, `download`: the loop above |
| `collections.py` | products the builder knows: bands, cloud masks, resampling, where each archive keeps them |
| `archives.py` | archives: how they are reached, what they allow |
| `search.py` | scenes over a window, in any archive: order, passes, zones |
| `read.py` | one scene onto a sample window, from any archive |
| `backends/` | one module per access method (`stac`: STAC + GDAL; `earthengine`: computePixels), same four functions |
| `credentials.py` | set up credentials once: `python -m majortom.build.credentials <name>` with `cdse`, `earthdata`, `earth-engine`, `cds`, `edh` or `source-coop` |
| `catalogue.py` | the products table of `docs/ARCHIVES.md`, generated from `collections.py` |

## Use

```python
from majortom import build
cells = build.cells_from_points(points, d=10)              # or build.cells_in(bbox, d=10)
errors = build.download(cells, "sentinel-2-l2a", "2024-06-01/2024-08-31")
```

## More

- `CONTRIBUTING.md`: how the builder is organised, adding a product or an archive
- `docs/ARCHIVES.md`: archives and products
