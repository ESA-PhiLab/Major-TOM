"""End-to-end check of majortom.release.taco on real samples (docs/RELEASE.md 6-7).

    built samples -> release() -> partitions + TACOCAT -> taco.validate (inside release)
      -> taco.open_dataset -> every sample and raster group: rumi.read(location, header) equals groups() data
      -> majortom:code (taco's MajorTOM extension) equals our cell; partition sizes; one sample read as COG

    python docs/evidence/taco_release_check.py <build folder with *.npz> <out.zip> <max_bytes> [--no-time]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.release import groups, read                       # noqa: E402
from majortom.release.taco import release, sample_id            # noqa: E402


def main(build: str, out: str, max_bytes: str, *flags: str) -> None:
    import rumi
    import taco

    with_time = "--no-time" not in flags
    samples = sorted(Path(build).rglob("*.npz"))
    t = time.time()
    path = release(samples, out, id="major-tom-check", description="End-to-end check of majortom.release",
                   licenses=["CC-BY-4.0"], providers=["Major TOM"], max_bytes=int(max_bytes), with_time=with_time,
                   overwrite=True)
    print(f"{len(samples)} samples -> {path} in {time.time() - t:.0f} s (validated)")
    zips = sorted(Path(out).parent.glob(f"{Path(out).stem}*.zip"))
    print("partition files:", [(z.name, round(z.stat().st_size / 1e6, 1)) for z in zips])
    ds = taco.open_dataset(str(path))
    table = ds.sql("SELECT * FROM dataset").to_pandas()
    print("columns:", list(table.columns))
    checked = 0
    for npz in samples:
        found, side = groups(npz)
        sid = sample_id(side["grid_cell"], side["grid_km"], side.get("datetime") if with_time else None)
        row = table[table["id"] == sid].iloc[0]
        code = sample_id(side["grid_cell"], side["grid_km"], None)
        assert row["majortom:code"] == code, (row["majortom:code"], code)
        for g in found:
            back = np.asarray(rumi.read(row[f"{g.name}.rumi::location"], row[f"{g.name}.rumi::header"]))
            assert np.array_equal(back.reshape(g.data.shape), g.data, equal_nan=True), (sid, g.name)
            checked += 1
    print(f"{checked} files read back from the partitions: identical; majortom:code matches every cell")
    print(table.groupby("partition:id").size().to_string())
    g = found[0]
    with read(row[f"{g.name}.rumi::location"], row[f"{g.name}.rumi::header"], as_="cog") as mem, mem.open() as f:
        print("COG from the partition:", f.crs, f.transform.c, f.transform.f, f.descriptions)


if __name__ == "__main__":
    main(*sys.argv[1:])
