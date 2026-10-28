"""The Earth Data Hub reader's direct hour read equals the old mapped read on real cached blocks (no requests).

    <cache>/era5land/blocks/<var>/<block>.f32 with its .filled.npy (complete blocks only)
      -> per block, random hours: np.array(memmap[hour]) against edh._hour(path, hour)
      -> identical (NaN-aware) or not, per variable; peak memory of this check

    python docs/evidence/edh_hour_check.py /data/Major-TOM/Core-Weather/cache    (Slurm CPU job)
"""
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.climate.readers.edh import T, _hour        # noqa: E402
from majortom.climate.sources import SOURCES             # noqa: E402


def main(cache: str, per_block: int = 24) -> None:
    grid = SOURCES["era5-land-edh"].grid
    rng = np.random.default_rng(0)
    blocks = sorted(Path(cache, "era5land", "blocks").glob("*/*.f32"))
    checked = same = 0
    for path in blocks:
        if not path.with_suffix(".filled.npy").exists():  # still being written by a running job
            continue
        try:
            mapped = np.memmap(path, np.float32, "r", shape=(T, grid.height, grid.width))
            for i in rng.choice(T, per_block, replace=False):
                same += np.array_equal(np.array(mapped[i]), _hour(path, int(i), grid.height, grid.width), equal_nan=True)
                checked += 1
            del mapped
        except FileNotFoundError:                          # removed by the running job's cleanup meanwhile
            continue
        print(f"{path.parent.name} block {path.stem}: ok so far {same}/{checked}", flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"{same}/{checked} hours identical over {len(blocks)} block files; peak memory {peak:.1f} GB")
    assert checked and same == checked


if __name__ == "__main__":
    main(sys.argv[1])
