"""Sentinel-2 L2A from different archives agrees, once each sample's recorded offset is removed (online).

    fixed acquisitions (place, day)  ->  one sample per archive  ->  same grid, day, satellite?
                                                                ->  values minus offset: per band median, p99 |diff|

Archives can hold different processings of one acquisition: Earth Search has ESA's Collection 1
reprocessing (baseline 05.xx), Planetary Computer the original (02.12 to 05.xx). Same baseline: identical
values. Older originals differ more (first run, 2026-10-21: p99 up to ~800 at 03.00, ~180 at 04.00), which
the builder cannot correct, so those differences are recorded, not tested. What is tested: the offset rule
(a wrong offset leaves a median difference near 1000) and identity where baselines match.
Run with `pytest -m network`; set MAJORTOM_EVIDENCE_DIR to write the table (docs/evidence/s2_sources.csv).
"""
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pystac_client")
from majortom import build                                       # noqa: E402

pytestmark = pytest.mark.network


def available_archives() -> list[str]:
    """Archives to compare: CDSE only when its keys are set up (it needs credentials)."""
    from majortom.build.archives import ARCHIVES
    from majortom.build.backends import available
    return ["planetary-computer", "earth-search"] + (["cdse"] if available(ARCHIVES["cdse"]) else [])


ARCHIVES = available_archives()
CASES = {                                                        # place: (lon, lat), clear day in every archive
    "snowbird-2021": ((-111.65, 40.58), "2021-08-25"),           # before the +1000 offset on Planetary Computer
    "snowbird-2024": ((-111.65, 40.58), "2024-08-29"),
    "cape-town-2021": ((18.45, -33.94), "2021-01-25"),           # southern hemisphere; whole tile covered
    "cape-town-2023": ((18.45, -33.94), "2023-01-25"),
}
MEDIAN_MAX = 300       # largest accepted median |difference|; measured up to 136, a wrong offset gives ~1000
VALID_MIN = 0.9        # share of pixels with data in both samples; less means a poor test case


def sample(case: str, archive: str, outdir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """One L2A sample of the case's day from one archive: (sidecar, arrays)."""
    (lon, lat), day = CASES[case]
    cells = build.cells_from_points(pd.DataFrame({"lon": [lon], "lat": [lat]}), d=10)
    where = outdir / case / archive
    errors = build.download(cells, "sentinel-2-l2a", f"{day}T00:00:00Z/{day}T23:59:59Z", archive=archive,
                            cloud=None, outdir=where, force=True)
    assert errors.empty, errors.to_dict("records")
    stem = where / "sentinel-2-l2a" / "000000"
    return json.loads(stem.with_suffix(".json").read_text()), dict(np.load(stem.with_suffix(".npz")))


def compare(case: str, a: tuple, b: tuple) -> list[dict]:
    """Per band: offset-corrected differences between two samples of one acquisition."""
    (ma, xa), (mb, xb) = a, b
    rows = []
    for band in ma["bands"]:
        valid = (xa[band] > 0) & (xb[band] > 0)                  # 0 is no data in both archives
        ya = xa[band].astype(float) - (ma["offset"] or {}).get(band, 0)
        yb = xb[band].astype(float) - (mb["offset"] or {}).get(band, 0)
        diff, corr = np.abs(ya - yb)[valid], np.corrcoef(ya[valid], yb[valid])[0, 1]
        rows.append(dict(case=case, band=band, a=ma["archive"], b=mb["archive"], item_a=ma["items"][0],
                         item_b=mb["items"][0], baseline_a=ma["processing_baseline"],
                         baseline_b=mb["processing_baseline"], offset_a=(ma["offset"] or {}).get(band),
                         offset_b=(mb["offset"] or {}).get(band), valid=float(valid.mean()),
                         equal=float((diff == 0).mean()), p50=float(np.percentile(diff, 50)),
                         p99=float(np.percentile(diff, 99)), corr=float(corr)))
    return rows


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> pd.DataFrame:
    """All cases, all archive pairs; written to MAJORTOM_EVIDENCE_DIR/s2_sources.csv if set."""
    outdir = tmp_path_factory.mktemp("sources")
    rows, grids = [], {}
    for case in CASES:
        samples = {archive: sample(case, archive, outdir) for archive in ARCHIVES}
        grids[case] = {a: (m["crs"], m["left"], m["top"], m["size"], m["datetime"][:10], m["items"][0][:3])
                       for a, (m, _) in samples.items()}             # archives stamp a pass at different times
        for x, y in combinations(ARCHIVES, 2):
            rows += compare(case, samples[x], samples[y])
    result = pd.DataFrame(rows)
    result.attrs["grids"] = grids
    if os.environ.get("MAJORTOM_EVIDENCE_DIR"):
        result.to_csv(Path(os.environ["MAJORTOM_EVIDENCE_DIR"]) / "s2_sources.csv", index=False)
    return result


def test_same_acquisition_on_the_same_grid(table):
    for case, grids in table.attrs["grids"].items():
        assert len(set(grids.values())) == 1, (case, grids)     # CRS, corner, size, day and satellite
    assert (table["valid"] >= VALID_MIN).all(), table.loc[table["valid"] < VALID_MIN, ["case", "valid"]]


def test_offsets_are_removed_correctly(table):
    bad = table[table["p50"] > MEDIAN_MAX]
    assert bad.empty, bad[["case", "band", "p50", "offset_a", "offset_b"]].to_string()


def test_same_processing_baseline_gives_identical_values(table):
    same = table[table["baseline_a"] == table["baseline_b"]]
    assert not same.empty                                        # snowbird-2024: 05.11 in both
    assert (same["equal"] == 1).all(), same[["case", "band", "equal", "p99"]].to_string()
