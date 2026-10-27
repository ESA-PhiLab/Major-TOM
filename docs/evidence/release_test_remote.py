"""Reading major-tom/release-test back from source.coop as any user would (no credentials), with timings.

    https://data.source.coop/major-tom/release-test/<source>/ -> taco.open_dataset (time to open the catalogue)
      -> every sample and file: rumi.read(location, header) over HTTP -> identical to the built sample? (time each)
      -> one sample as a COG (majortom.release.read) -> CRS and transform
      -> the same open-and-read timings for Core-DEM (us-west-2), for comparison
      -> <root>/release-test/remote.csv
    python docs/evidence/release_test_remote.py --root /data/Major-TOM/release-test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE = "https://data.source.coop/major-tom/release-test"


def main(root: Path) -> None:
    import rumi
    import taco

    from majortom.release import groups, read
    from majortom.release.taco import sample_id

    rows = []
    for folder in sorted(f for f in (root / "release-test").iterdir() if f.is_dir() and any(f.glob("*.zip"))):
        name = folder.name
        t = time.perf_counter()
        ds = taco.open_dataset(f"{BASE}/{name}/")
        table = ds.sql("SELECT * FROM dataset").to_pandas()
        open_s = time.perf_counter() - t
        static = all(len(i.split("_")) == 3 for i in table["id"])
        exact = files = 0
        reads = []
        for npz in (root / "build" / name).rglob("*.npz"):
            found, side = groups(npz)
            match = table[table["id"] == sample_id(side["grid_cell"], side["grid_km"], None if static else side.get("datetime"))]
            if match.empty:
                continue
            r = match.iloc[0]
            for g in found:
                t = time.perf_counter()
                back = np.asarray(rumi.read(r[f"{g.name}.rumi::location"], r[f"{g.name}.rumi::header"]))
                reads.append(time.perf_counter() - t)
                exact += bool(np.array_equal(back.reshape(g.data.shape), g.data, equal_nan=True))
                files += 1
        cog = ""
        if files:
            g0 = found[0]
            with read(r[f"{g0.name}.rumi::location"], r[f"{g0.name}.rumi::header"], as_="cog") as mem, mem.open() as f:
                cog = f"{f.crs.to_epsg()} {f.transform.c:.0f},{f.transform.f:.0f}"
        rows.append(dict(source=name, open_s=round(open_s, 2), files=files, exact=exact,
                         read_median_s=round(float(np.median(reads)), 3) if reads else None,
                         read_max_s=round(float(np.max(reads)), 3) if reads else None, cog=cog))
        print(rows[-1], flush=True)
    # Core-DEM (us-west-2) for comparison: open, then read 10 samples
    t = time.perf_counter()
    dem = taco.open_dataset("https://data.source.coop/major-tom/core-dem/")
    sample = dem.sql('SELECT id, "dem.rumi::location" AS loc, "dem.rumi::header" AS hdr FROM dataset LIMIT 10').to_pylist()
    open_s = time.perf_counter() - t
    reads = []
    for s in sample:
        t = time.perf_counter()
        rumi.read(s["loc"], s["hdr"])
        reads.append(time.perf_counter() - t)
    rows.append(dict(source="core-dem (us-west-2, for comparison)", open_s=round(open_s, 2), files=len(reads),
                     exact=None, read_median_s=round(float(np.median(reads)), 3), read_max_s=round(float(np.max(reads)), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(root / "release-test" / "remote.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    main(Path(p.parse_args().root))
