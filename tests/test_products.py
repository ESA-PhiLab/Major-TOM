"""Every product from every archive builds a sample (online).

    (product, archive) pairs from COLLECTIONS  ->  one sample at Loch Lomond  ->  bands, shapes, no-data, records

Run with `pytest -m network tests/test_products.py`; pairs whose archive needs credentials (CDSE keys, an
Earthdata token, Earth Engine) are skipped when those are not set up. With MAJORTOM_EVIDENCE_DIR set, the
samples are kept in <dir>/samples/<archive>/<product>/ for docs/evidence/gallery.py.
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

PLACE = (-4.58, 56.09)            # cell 624U_29L: south Loch Lomond, its islands, Conic Hill, Luss and the A82
SUMMER = "2024-06-01/2024-08-31"                                 # one window for every product that allows it
DATES = {                                                        # dates with data at PLACE; static: None
    **{name: SUMMER for name in ("sentinel-2-l2a", "sentinel-2-l1c", "landsat-c2-l2", "landsat-c2-l1", "hls2-s30",
                                 "hls2-l30", "sentinel-1-rtc", "sentinel-1-grd", "sentinel-1-grd-tc",
                                 "dynamic-world", "modis-09a1", "modis-09q1", "modis-43a4", "modis-11a2",
                                 "modis-13q1", "viirs-dnb-monthly", "sentinel-3-olci-l1", "opera-rtc-s1",
                                 "opera-dswx-hls", "opera-dswx-s1", "opera-dist-alert-hls", "sentinel-3-olci-l2-lfr")},
    "aster-l1t": "2004-07-01/2004-09-30",                        # Planetary Computer: 2000-2006 only
    "alos-palsar-mosaic": "2020", "alos-fnf-mosaic": "2020", "esa-worldcover": "2021",   # yearly: nearest version
    "io-lulc-annual-v02": "2022", "esa-cci-lc": "2020",
}
MAX_NODATA = 0.05
MASKED = {"dynamic-world": 0.4,                                  # masks cloud and shadow pixels itself (mountains)
          "sentinel-3-olci-l2-lfr": 0.9}                         # masks clouds itself; one cloudy Scottish summer


def has_credentials(archive: str) -> bool:
    """Whether the archive's credentials, if it needs any, are set up."""
    from majortom.build.backends import available
    return available(ARCHIVES[archive])


# archive gaps at PLACE (docs/ARCHIVES.md): the sample builds but is partial; the test xpasses once a gap is filled
KNOWN_GAPS = {
    ("hls2-l30", "planetary-computer"): "Planetary Computer lacks HLS tile T30VUH for most of summer 2024",
}
PAIRS = [pytest.param(name, archive, id=f"{name}@{archive}",
                      marks=[pytest.mark.skipif(not has_credentials(archive), reason=f"no {archive} credentials")]
                      + ([pytest.mark.xfail(reason=KNOWN_GAPS[name, archive], strict=False)]
                         if (name, archive) in KNOWN_GAPS else []))
         for name, collection in COLLECTIONS.items() for archive in collection.archives()]


@pytest.mark.parametrize("name, archive", PAIRS)
def test_product_builds_from_archive(name, archive, tmp_path):
    import os
    from pathlib import Path

    collection = COLLECTIONS[name]
    if os.environ.get("MAJORTOM_EVIDENCE_DIR"):                  # keep the sample for the gallery
        tmp_path = Path(os.environ["MAJORTOM_EVIDENCE_DIR"]) / "samples" / archive
    cells = build.cells_from_points(pd.DataFrame({"lon": [PLACE[0]], "lat": [PLACE[1]]}), d=10)
    ranked = collection.clouds or collection.clouds_from          # with a cloud ranking: a clear scene, as users
    errors = build.download(cells, name, DATES.get(name), archive=archive, outdir=tmp_path,
                            **({} if ranked else {"cloud": None}))
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
    assert meta["nodata"] <= MASKED.get(name, MAX_NODATA), meta["nodata"]
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
    provider, collection = collection.split("/", 1) if archive.providers else (None, collection)
    entry = retry(lambda: catalog(archive, provider).get_collection(collection), 6)
    first, last = entry.extent.temporal.intervals[0]
    return first.date().isoformat(), (last.date().isoformat() if last else None)
