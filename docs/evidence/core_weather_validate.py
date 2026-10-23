"""Checks on a built Core-Weather source folder: completeness from file metadata, values from a sample of days.

    <root>/<source>/acquisitions.parquet, hourly/, daily/
      -> rows per table (parquet metadata, no data read) against acquisitions x steps
      -> a sample of day files: steps per acquisition, share of empty values per column, value ranges
      -> printed summary

    python -u docs/evidence/core_weather_validate.py <source folder> [sample days]
First run, 2026-10-23, ERA5 (/data/Major-TOM/Core-Weather/era5, built in 10 h 45 min, 882,144 fields): 5,961,727
samples, 3,704,013 acquisitions; hourly 181,496,637 rows and daily 225,944,793 rows, exactly acquisitions x steps;
no empty values except land-only (34%: open-sea cells) and sea-only variables (64%: cells without sea).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def rows(folder: Path) -> tuple[int, int]:
    files = sorted(folder.glob("year=*/*.parquet"))
    return len(files), sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def main(root: Path, sample: int) -> None:
    links = pd.read_parquet(root / "acquisitions.parquet", columns=["grid_cell", "acquired", "dataset"])
    acquisitions = links.drop_duplicates(["grid_cell", "acquired"])
    print(f"samples {len(links):,} ({links.dataset.value_counts().to_dict()}), acquisitions {len(acquisitions):,}")
    for kind, steps in (("hourly", 49), ("daily", 61)):
        if not (root / kind).exists():
            continue
        files, n = rows(root / kind)
        print(f"{kind}: {files:,} day files, {n:,} rows; expected {len(acquisitions) * steps:,} "
              f"({n / (len(acquisitions) * steps):.4f})")
        picked = sorted((root / kind).glob("year=*/*.parquet"))
        picked = [picked[i] for i in np.linspace(0, len(picked) - 1, min(sample, len(picked))).astype(int)]
        table = pd.concat([pd.read_parquet(f) for f in picked])
        per = table.groupby(["grid_cell", "acquired"]).size()
        print(f"   sample of {len(picked)} days: {len(per):,} acquisitions, steps per acquisition {sorted(per.unique())}")
        values = table.select_dtypes("number").drop(columns=[c for c in table if c.startswith("step")], errors="ignore")
        summary = pd.DataFrame({"empty": values.isna().mean(), "p01": values.quantile(0.01), "median": values.median(),
                                "p99": values.quantile(0.99)})
        print(summary.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 40)
