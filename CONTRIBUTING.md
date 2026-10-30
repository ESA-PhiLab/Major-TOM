> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed

# Contributing to Major TOM

Contributions are welcome as pull requests to [ESA-PhiLab/Major-TOM](https://github.com/ESA-PhiLab/Major-TOM). This page covers the most common one: a new product or a new archive for `majortom.build`. Data sources and setup are described in [docs/ARCHIVES.md](docs/ARCHIVES.md).

## 1. Setup

```bash
git clone https://github.com/ESA-PhiLab/Major-TOM && cd Major-TOM
uv sync --extra build --extra gee --extra viz      # or: pip install -e '.[build,gee,viz]' pytest
uv run pytest -q                                   # offline tests: no network, no credentials
```

Tests that read live archives are marked `network` and skipped by default: `uv run pytest -m network tests/test_products.py`. Pairs whose archive needs credentials are skipped until those are set up ([docs/ARCHIVES.md](docs/ARCHIVES.md), section 4).

## 2. How the builder is organised

```
majortom/build/
  __init__.py      download(), build_one(), pick_scene()       the same for every archive
  collections.py   products: bands, kind, period, and one Source per archive (its names for the bands)
  archives.py      archives: which backend, address, settings, workers, rate limits
  credentials.py   looking up and setting up keys and tokens
  search.py        scene order, satellite passes, UTM zones, the same pass in another product
  read.py          window layout, stitching tiles
  backends/
    stac.py        STAC API + GDAL: Planetary Computer, Earth Search, CDSE, NASA Earthdata
    earthengine.py Google Earth Engine
```

A product is *what* is saved; an archive is *where* it comes from; a backend is *how* it is read. Each backend has the same four functions: `search`, `header`, `read_tile`, `session`.

## 3. Add a product

One entry in `COLLECTIONS` in `collections.py`. For a product already on Planetary Computer under its own name and band names, that is all:

```python
"modis-09q1": Collection(("sur_refl_b01", "sur_refl_b02"), res_m=250,
                         sources={"planetary-computer": Source("modis-09Q1-061", period=("2000-02-18", None))}),
```

- **Bands** are the names saved in every sample. If an archive names them differently, its `Source` maps them: `Source("sentinel-2-c1-l2a", {"B04": "red", ...})`. Band `n` of a multi-band file is `"ASSET#n"`.
- **`kind`** groups products in docs and benchmarks (`optical`, `radar`, `elevation`, `land cover`, ...).
- **`period`** is the time range the archive's catalogue declares; a network test checks it.
- **`time`**: `"scene"` (dates select acquisitions), `"yearly"` (dates select a version) or `"static"`.
- **Clouds:** `clouds=(mask band, function)` if the product has a mask, or `clouds_from="<product>"` to borrow the mask of the same pass (Sentinel-2 L1C and Dynamic World borrow the L2A SCL).
- **`res_m`** is needed when the product is not in UTM (latitude/longitude, MODIS sinusoidal): the pixel size it is resampled to.
- **`offset`** only for products whose stored values carry an offset that archives do not all declare (Sentinel-2).

For Earth Engine, also add an `EarthEngine` entry to `EARTH_ENGINE`: band names, and which properties hold cloud cover, satellite and tile.

Then add the product's test dates to `DATES` in `tests/test_products.py`, and run `python -m majortom.build.catalogue` to update the table in docs/ARCHIVES.md.

## 4. Add an archive

**Reachable by STAC and GDAL** (most archives): one `Archive` entry in `archives.py`. Earth Search, for example:

```python
"earth-search": Archive("earth-search", "https://earth-search.aws.element84.com/v1", sort_prefix="properties.",
                        gdal_env=unsigned_s3_env),
```

- `gdal_env` gives the GDAL settings for its files (`backends/stac.py` has public, unsigned S3, S3 keys, bearer token).
- Credentials are looked up in `credentials.py` when a session starts; never store them in an `Archive`.
- Rate limits: `workers`, `retries`, `search_gap`.
- Then add a `Source` for each product it has, in `collections.py`.

**Another access method** (not STAC and GDAL): one module in `backends/` with the four functions, named by `Archive.backend`. `backends/earthengine.py` is the example: it turns each image into a `pystac.Item` whose assets are the bands, so scene choice, layout and stitching work unchanged.

## 5. What a new product or archive must pass

1. The offline suite: `uv run pytest -q`.
2. `tests/test_products.py` for your pairs: a sample builds, band names and shapes are right, no-data stays under 5%, and the period matches the catalogue.
3. If another archive has the same data, the same acquisition from both should match once offsets are removed. `tests/test_sources.py` and `docs/evidence/` show how. If they do not match, find out why (processing version, units, tiles) and say so in docs/ARCHIVES.md, section 5.
4. A look at its row in the gallery: `MAJORTOM_EVIDENCE_DIR=<dir> uv run pytest -m network tests/test_products.py`, then `python docs/evidence/gallery.py <dir>`.
5. No credentials in code or tests: `tests/test_no_secrets.py` fails if something that looks like a key is committed.

## 6. Pull request checklist

- [ ] Product or archive entry, with a `period` for each source
- [ ] Test dates in `tests/test_products.py`; network tests pass for the new pairs
- [ ] Cross-archive comparison, if the data exist elsewhere
- [ ] `python -m majortom.build.catalogue` run (the table test fails otherwise)
- [ ] A setup section in docs/ARCHIVES.md for a new archive
- [ ] One line in [LOG.md](LOG.md): date, what changed
