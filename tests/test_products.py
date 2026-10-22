"""Every product from every archive builds a sample (online).

    (product, archive) pairs from COLLECTIONS  ->  one sample at Snowbird  ->  bands, shapes, no-data, records

Run with `pytest -m network tests/test_products.py`; pairs whose archive needs credentials (CDSE keys, an
Earthdata token) are skipped when those are not set up.
docs/evidence/build_speed.py uses the same places and dates to measure reading speed.
"""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pystac_client")
from majortom import build                                       # noqa: E402
from majortom.build.archives import ARCHIVES                     # noqa: E402
from majortom.build.collections import COLLECTIONS               # noqa: E402

pytestmark = pytest.mark.network

SNOWBIRD = (-111.65, 40.58)
DATES = {                                                        # dates with data at Snowbird; static: None
    "sentinel-2-l2a": "2024-07-01/2024-08-31", "sentinel-2-l1c": "2024-07-01/2024-08-31",
    "landsat-c2-l2": "2024-07-01/2024-08-31", "hls2-s30": "2024-07-01/2024-08-31",
    "hls2-l30": "2024-07-01/2024-08-31", "sentinel-1-rtc": "2023-06-01/2023-06-30",
    "landsat-c2-l1": "2024-07-01/2024-08-31", "dynamic-world": "2024-07-01/2024-08-31",
    "sentinel-1-grd": "2023-06-01/2023-06-30",
    "alos-palsar-mosaic": "2020", "alos-fnf-mosaic": "2020", "esa-worldcover": "2021",
    "io-lulc-annual-v02": "2022", "esa-cci-lc": "2020",
}
MAX_NODATA = 0.05


def has_credentials(archive: str) -> bool:
    """Whether the archive's credentials, if it needs any, are set up."""
    from majortom.build.backends import available
    return available(ARCHIVES[archive])


PAIRS = [pytest.param(name, archive, id=f"{name}@{archive}",
                      marks=pytest.mark.skipif(not has_credentials(archive), reason=f"no {archive} credentials"))
         for name, collection in COLLECTIONS.items() for archive in collection.archives()]


@pytest.mark.parametrize("name, archive", PAIRS)
def test_product_builds_from_archive(name, archive, tmp_path):
    collection = COLLECTIONS[name]
    cells = build.cells_from_points(pd.DataFrame({"lon": [SNOWBIRD[0]], "lat": [SNOWBIRD[1]]}), d=10)
    errors = build.download(cells, name, DATES.get(name), archive=archive, outdir=tmp_path, cloud=None)
    assert errors.empty, errors.to_dict("records")
    stem = tmp_path / name / "000000"
    meta, arrays = json.loads(stem.with_suffix(".json").read_text()), np.load(stem.with_suffix(".npz"))
    assert meta["archive"] == archive and meta["collection"] == name
    assert meta["bands"] == list(arrays.files) and set(collection.assets) <= {b.rsplit("_", 1)[0] for b in
                                                                            arrays.files} | set(arrays.files)
    for band in arrays.files:                                    # square, and as wide as the window at its pixel size
        side = arrays[band].shape[-1]
        assert arrays[band].shape[-2:] == (side, side) and side > 0
    assert meta["window_m"] == pytest.approx(meta["res"] * meta["size"])
    assert meta["nodata"] <= MAX_NODATA, meta["nodata"]
    assert set(meta["timing"]) == {"search", "scene_choice", "bands", "total", "retries"}
    if collection.offset:
        assert all(meta["offset"][b] in (0.0, 1000.0) for b in collection.offset[0] if b in meta["offset"])


@pytest.mark.parametrize("name, archive", PAIRS)
def test_period_matches_the_catalogue(name, archive):
    source = COLLECTIONS[name].source(name, archive)
    if source.period is None:
        pytest.skip("static product: one version, no period")
    start, end = declared_period(ARCHIVES[archive], source.collection)
    assert start == source.period[0]
    assert source.period[1] is None or end in (source.period[1], None)   # ongoing: catalogues give the latest date


def declared_period(archive, collection: str) -> tuple[str, str | None]:
    """First and last day the archive's catalogue declares; a merged Earth Engine search: the earliest start."""
    import requests
    from majortom.build.backends import retry

    if archive.backend == "earthengine":                 # Earth Engine's public STAC files, one per collection
        periods = []
        for cid in collection.split("+"):
            url = (f"https://storage.googleapis.com/earthengine-stac/catalog/{cid.split('/')[0]}/"
                   f"{cid.replace('/', '_')}.json")
            periods.append(retry(lambda: requests.get(url, timeout=30).json())["extent"]["temporal"]["interval"][0])
        return min(p[0][:10] for p in periods), None
    from majortom.build.backends.stac import catalog
    entry = retry(lambda: catalog(archive).get_collection(collection), 6)
    first, last = entry.extent.temporal.intervals[0]
    return first.date().isoformat(), (last.date().isoformat() if last else None)
