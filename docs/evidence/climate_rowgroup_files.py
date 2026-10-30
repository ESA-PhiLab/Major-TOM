"""Row-group test files for the climate globe: published series months rewritten with larger row groups.

    <release>/<dataset>/series/year=2024/MM.parquet (MM = 01..06; rows along the Hilbert curve, 65,536-row groups)
      -> the same table, every column and row unchanged, written with ROW_GROUP rows per group (zstd, statistics)
      -> <out>/rowgroups-<ROW_GROUP>/<dataset>/series/year=2024/MM.parquet, and cells.parquet copied alongside
    A map month then takes fewer requests (fewer row groups), a cell's value a larger chunk per file. Uploaded to
    major-tom/climate/_experiments and removed after the tests.

    python docs/evidence/climate_rowgroup_files.py /data/Major-TOM/climate-release /data/Major-TOM/experiments
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

DATASETS, MONTHS, SIZES = ["CPC-Monthly", "IMERG-Monthly"], range(1, 7), [262_144, 1_048_576]


def build(release: Path, out: Path) -> None:
    for size in SIZES:
        for dataset in DATASETS:
            target = out / f"rowgroups-{size}" / dataset
            (target / "series" / "year=2024").mkdir(parents=True, exist_ok=True)
            shutil.copy2(release / dataset / "cells.parquet", target / "cells.parquet")
            for m in MONTHS:
                rel = Path("series") / "year=2024" / f"{m:02d}.parquet"
                table = pq.ParquetFile(release / dataset / rel).read()
                pq.write_table(table, target / rel, row_group_size=size, compression="zstd", write_statistics=True)
                back = pq.ParquetFile(target / rel)
                assert back.read().equals(table), f"{target / rel} differs from its source"
                print(f"{size} {dataset} {rel}: {back.metadata.num_row_groups} row groups", flush=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
