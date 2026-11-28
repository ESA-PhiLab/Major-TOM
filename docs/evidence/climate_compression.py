"""How much of the duplication from mapping coarse sources onto Major TOM cells the Parquet compression recovers.

    per monthly dataset: series/year=2020/07.parquet + cells.parquet (local release copies)
      -> A. footer: compressed bytes of the value columns (`*_valid` left out), no data read
      -> B. cells.parquet: cell centres -> index of the source pixel under each centre -> distinct pixels covered
      -> C. duplication = cells / pixels covered; baseline = pixels covered x 4 bytes (float32, uncompressed)
             per value column
      -> D. each value column: share of cells equal to the previous cell in the file's (Hilbert) order
      -> docs/evidence/climate_compression.json and a table on stdout

    Approximate: a cell straddling two source pixels counts under the one at its centre. The baseline is not
    compressed, so stored / baseline credits compression with more than the duplication alone.

    python docs/evidence/climate_compression.py /data/Major-TOM/climate-release      (Slurm CPU job)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

R_KM, D_KM = 6378.137, 10.0                      # majortom/cells.py
DLAT = 180 / math.ceil(math.pi * R_KM / D_KM)
OUT = Path(__file__).with_suffix(".json")

# dataset -> (prefix, source resolution in degrees, offset of the first pixel centre from -90 / -180)
SOURCES = {
    "CPC-Monthly": ("cpc_", 0.5, 0.25),
    "IMERG-Monthly": ("imerg_", 0.1, 0.05),
    "CAMS-Monthly": ("cams_", 0.75, 0.0),
    "ERA5Land-Monthly": ("era5land_", 0.1, 0.0),
    "ERA5-Monthly": ("era5_", 0.25, 0.0),
}


def centres(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of each cell's midpoint, from its signed row and column (majortom.cells.Cell.centroid)."""
    lat = row * DLAT
    n_cols = np.maximum(np.ceil(2 * np.pi * R_KM * np.cos(np.radians(lat)) / D_KM), 1)
    dlon = 360 / n_cols
    return col * dlon + dlon / 2, lat + DLAT / 2


def pixels_covered(lon: np.ndarray, lat: np.ndarray, res: float, offset: float) -> int:
    """Distinct source pixels under the cell centres (pixel centres at offset + k * res)."""
    n_lon = round(360 / res)
    i = np.floor((lat + 90 - offset) / res + 0.5).astype(np.int64)
    j = np.floor((lon + 180 - offset) / res + 0.5).astype(np.int64) % n_lon
    return int(np.unique(i * n_lon + j).size)


def measure(folder: Path) -> dict:
    prefix, res, offset = SOURCES[folder.name]
    path = folder / "series" / "year=2020" / "07.parquet"
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    values = [n for n in names if n.startswith(prefix) and not n.endswith("_valid")]
    index = [names.index(n) for n in values]
    meta = pf.metadata
    stored = sum(meta.row_group(g).column(i).total_compressed_size
                 for g in range(meta.num_row_groups) for i in index)                    # A
    cells = pq.read_table(folder / "cells.parquet", columns=["grid_row_u", "grid_col_r"])
    lon, lat = centres(cells["grid_row_u"].to_numpy().astype(np.float64), cells["grid_col_r"].to_numpy())
    n_cells, covered = len(lon), pixels_covered(lon, lat, res, offset)                   # B
    repeats = []
    for name in values:                                                                 # D, one column at a time
        v = pf.read(columns=[name]).column(0).to_numpy(zero_copy_only=False)
        repeats.append(float(np.mean(v[1:] == v[:-1])))
    return {
        "dataset": folder.name, "resolution_deg": res, "cells": n_cells, "pixels_covered": covered,
        "duplication": n_cells / covered, "value_columns": len(values),
        "stored_bytes": stored, "raw_on_cells_bytes": n_cells * 4 * len(values),
        "baseline_bytes": covered * 4 * len(values),                                    # C
        "stored_over_baseline": stored / (covered * 4 * len(values)),
        "repeat_share_median": float(np.median(repeats)), "repeat_share_min": min(repeats),
        "repeat_share_max": max(repeats),
    }


def main(release: Path) -> None:
    rows = [measure(release / name) for name in SOURCES]
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print("| Dataset | Source grid | Cells | Source pixels | Duplication | Columns | Stored | Raw on cells | "
          "Source-grid baseline | Stored / baseline | Repeats (median, min-max) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    mb = lambda b: f"{b / 1e6:,.1f} MB"                                                 # noqa: E731
    for r in rows:
        print(f"| {r['dataset']} | {r['resolution_deg']}° | {r['cells']:,} | {r['pixels_covered']:,} | "
              f"{r['duplication']:.2f}× | {r['value_columns']} | {mb(r['stored_bytes'])} | "
              f"{mb(r['raw_on_cells_bytes'])} | {mb(r['baseline_bytes'])} | {r['stored_over_baseline']:.2f} | "
              f"{r['repeat_share_median']:.1%} ({r['repeat_share_min']:.1%}-{r['repeat_share_max']:.1%}) |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
