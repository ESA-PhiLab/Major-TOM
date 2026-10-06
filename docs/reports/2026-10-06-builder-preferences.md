> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

> **Review status:** unreviewed. Built autonomously on Miko's request the evening before the workshop; needs review.

# 2026-10-06 — builder scene preferences, yearly products, alias fix

Follows [2026-10-06-majortom-build.md](2026-10-06-majortom-build.md). Samples can now ask for clear or cloudy scenes. Yearly and static products pick their version by date. `MajorTOM/` is replaced by one file, `MajorTOM.py`. The workshop Colab badge now opens this repo's notebook. STRATEGY WS5 covers other backends and related tools.

## Flow

    download(cells, collection, datetime, prefer, cloud, max_nodata, strict)
      -> Choice(...)                                      recorded in every sample's .json
      -> per sample: scenes()  STAC search sorted by prefer -> passes() groups tiles by pass
                     tiles_for() one option per CRS zone, covering zones first
                     pick_scene() first acceptable option, else closest (or fail if strict)

## majortom/build/__init__.py

`Choice` holds the scene rules (`__init__.py:67`). `cloud` is a range, so cloudy samples are as easy to ask for as clear ones; `None` accepts any.

    def cloud_gap(self, cloud):
        if cloud is None or self.cloud is None:
            return 0.0
        low, high = self.cloud
        return max(low - cloud, cloud - high, 0.0)

`pick_scene` (`__init__.py:93`) tries up to 5 scenes, each in every zone `tiles_for` offers. Data present but clouds out of range moves on to the next scene. Without an acceptable one: `strict=True` fails the sample, else the closest is used and its real cloud share recorded.

    return min(tried, key=lambda t: (t[5] > choice.max_nodata, choice.cloud_gap(t[4]), t[5]))

## majortom/build/search.py

`scenes` (`search.py:78`) sorts in the catalogue by preference; "nearest" re-sorts by distance to the date (the middle of a range). Products without clouds use "newest". Static products ignore dates. Yearly products with no version in the dates take the newest, and the note says so.

    items = search(datetime if collection.time != "static" else None)
    if not items and collection.time == "yearly" and datetime:
        items, note = search(None), f"no version in {datetime}: newest used"

`passes` (`search.py:105`) groups tiles by platform within 5 minutes. HLS tiles of one pass have sensing times seconds apart; grouping by exact time gave 71% no-data. `tiles_for` (`search.py:123`) now returns all zones in order, and repairs footprints with `shapely.make_valid` (an io-lulc footprint near 180° raised a GEOS TopologyException).

## majortom/build/collections.py

`Collection.time` is "scene", "yearly" or "static" (`collections.py:43`). Yearly: alos-palsar-mosaic, alos-fnf-mosaic, esa-worldcover, io-lulc-annual-v02, esa-cci-lc. Static: nasadem, alos-dem, hgb, jrc-gsw.

## MajorTOM.py (commit c41c3c8)

`MajorTOM/` and `majortom/` differed only in case: one overwrites the other on macOS and Windows checkouts. One file `MajorTOM.py` re-exports `majortom` and maps old submodule names with a meta-path finder. `tests/test_imports.py` checks no tracked paths differ only in case.

## docs/workshop/workshop.ipynb

Cell 0: badges pointed to the taco repo's copy; now `ESA-PhiLab/Major-TOM/blob/refactor/v1/docs/workshop/workshop.ipynb`. Cell 6 explains `prefer`, `cloud`, `strict` and dates per product kind. Cell 8 adds `PREFER` and `CLOUD`.

## docs/STRATEGY.md WS5

A `Backend` protocol (`scenes`, `read`), a table of backends (Planetary Computer done; Earth Search, CDSE, GEE, Major TOM datasets planned), and a table of related tools (odc-stac/odc-geo, stackstac, aereo, phidown, earthengine-api/xee, TorchGeo) with why `build` does not use them.

## Verification

- 19 offline tests pass; `tests/test_build.py` covers `Choice`, `interval`/`middle`, collection kinds, `passes`.
- Preferences at Snowbird, Sentinel-2: clearest 0% cloud; cloudiest 100%; nearest picks 30 Jun / 5 Jul around 1 Jul; newest with `cloud=None`.
- Yearly: 2018, 2020, 2015, 2017, 2016 pick their versions; 2025 falls back to 2023 (worldcover) with the note.
- HLS S30 at Björkliden: zone 33, 2 tiles, no-data 0 (was 71%).
- io-lulc at Manaus: 2024 → 2022 (newest), 2018 → 2018; no-data 2.5%.
- Contact sheet of 13 collections × 2 sites (Snowbird, Björkliden): 25 of 26 samples look right. NASADEM at 68°N is empty, correctly (no data north of 60°N). The sheet uses the earlier all-13 run; the full run was not repeated after `make_valid` and `passes`, only the targeted checks above.

## Decisions

- Cloud as a range over a maximum: cloudy datasets need a lower bound.
- Fallback to the closest scene by default, `strict` opt-in: workshop runs should not fail on a few cells.
- Yearly fallback to the newest version over failing: the note records it per sample.
- One `MajorTOM.py` over a folder: no case clash, old imports still work.
- Colab badge on `refactor/v1`: the notebook lives here until it moves to the taco repo.

## Known gaps

- Badge must change when the branch merges or the notebook moves.
- `planetary_computer` emits a pydantic deprecation warning; not ours.
- Notebook header says Tue Oct 6 2026, 9:00 MDT; check the date.

## Look at first

- majortom/build/__init__.py:93 (`pick_scene`)
- majortom/build/search.py:78 (`scenes`, yearly fallback)
- docs/workshop/workshop.ipynb cells 6 and 8
