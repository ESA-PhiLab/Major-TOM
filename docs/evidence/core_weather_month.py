"""Speed and memory of majortom.climate.stream on one month of Core-S2L2A acquisitions (ERA5, all variables).

    Core-S2L2A metadata -> samples acquired in one month (all cells)
      -> stream() to <out>: hourly ±24 h, daily ±30 days, one parquet file per acquisition day
      -> fields read, reads per second, wall time, peak memory, output size

    python -u docs/evidence/core_weather_month.py <out> [YYYY-MM] [workers]
First run, 2026-10-23, July 2021: 31,494 samples (31,487 acquisitions); 22,320 fields in 13.0 min, 29/s with 64
threads (no faster than 32: the limit is not the threads); peak memory 3.5 GB; 164 MB in 62 files. 34% of the
acquisitions have no ERA5 land pixel: Core-S2L2A includes open-sea cells, so soil water and snow are empty there.
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.climate.stream import stream                       # noqa: E402

META = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"


def month_samples(month: str) -> pd.DataFrame:
    with fsspec.open(META).open() as f:
        meta = pq.read_table(f, columns=["grid_cell", "grid_row_u", "grid_col_r", "timestamp"]).to_pandas()
    acquired = pd.to_datetime(meta["timestamp"], format="mixed", utc=True)
    return meta[acquired.dt.strftime("%Y-%m") == month]


def main(out: Path, month: str, workers: int) -> None:
    samples = month_samples(month)
    print(f"{month}: {len(samples):,} samples, {samples.grid_cell.nunique():,} cells", flush=True)
    start = time.time()
    reads = stream(samples, out, "era5", workers=workers, progress=2400)
    wall = time.time() - start
    size = sum(p.stat().st_size for p in out.rglob("*.parquet"))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6                  # kB -> GB
    print(f"fields read {reads:,} in {wall / 60:.1f} min ({reads / wall:.0f}/s with {workers} threads); "
          f"peak memory {peak:.1f} GB; output {size / 1e6:.0f} MB in {len(list(out.rglob('*.parquet')))} files")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "2021-07", int(sys.argv[3]) if len(sys.argv) > 3 else 64)
