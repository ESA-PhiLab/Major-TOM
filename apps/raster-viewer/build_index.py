"""Footprint index of each Core dataset, for the Raster Viewer: where its samples are and where each one is stored.

    <core>/Core-<name>/metadata.parquet (one row per sample: grid_row_u, grid_col_r, parquet_url, parquet_row, ...)
      -> parquet_url -> part number (every URL is <base>/images/part_NNNNN.parquet; checked)
      -> rows sorted by grid_row_u, grid_col_r
      -> footprints/<name>.parquet: grid_row_u, grid_col_r, part, parquet_row, timestamp, + cloud_cover / orbit_state
         row groups of 65,536 rows with statistics and a page index, no value dictionaries (as the climate files):
         the app reads the two position columns for the coverage map and, for a click, the page holding the cell
      -> footprints/datasets.json: per dataset its label, the URL base of its image files, its samples

    python build_index.py /data/Major-TOM <out folder>        (Slurm, .venv311)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASETS = {"S2L2A": "Sentinel-2 L2A", "S2L1C": "Sentinel-2 L1C", "S1RTC": "Sentinel-1 RTC", "DEM": "COP-DEM"}
EXTRA = ["timestamp", "cloud_cover", "orbit_state"]                      # kept where a dataset has them
ROW_GROUP = 65_536


def index(metadata: Path, name: str) -> tuple[pa.Table, str]:
    """The footprint table of one dataset, and the URL base of its image files."""
    have = pq.read_schema(metadata).names
    table = pq.read_table(metadata, columns=["grid_row_u", "grid_col_r", "parquet_url", "parquet_row",
                                             *[c for c in EXTRA if c in have]])
    base = f"https://huggingface.co/datasets/Major-TOM/Core-{name}/resolve/main/images/"
    urls = table.column("parquet_url").to_pylist()
    part = [re.fullmatch(re.escape(base) + r"part_(\d{5})\.parquet", u) for u in urls]
    if not all(part):
        raise ValueError(f"{metadata}: a parquet_url outside {base}part_NNNNN.parquet")
    out = pa.table({"grid_row_u": pc.cast(table.column("grid_row_u"), pa.int16()),
                    "grid_col_r": pc.cast(table.column("grid_col_r"), pa.int16()),
                    "part": pa.array(np.array([int(m[1]) for m in part], np.int16)),
                    "parquet_row": pc.cast(table.column("parquet_row"), pa.int16()),
                    **{c: table.column(c) for c in EXTRA if c in have}})
    if "cloud_cover" in out.column_names:
        out = out.set_column(out.schema.get_field_index("cloud_cover"), "cloud_cover",
                             pc.cast(out.column("cloud_cover"), pa.float32()))
    return out.sort_by([("grid_row_u", "ascending"), ("grid_col_r", "ascending")]), base


def write(table: pa.Table, path: Path) -> None:
    keys = ["grid_row_u", "grid_col_r", "orbit_state"]
    pq.write_table(table, path, row_group_size=ROW_GROUP, compression="zstd", write_statistics=True,
                   write_page_index=True, use_dictionary=[c for c in keys if c in table.column_names],
                   data_page_size=4096)


def main(core: str, out: str) -> None:
    core, out = Path(core), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    listing = {}
    for name, label in DATASETS.items():
        table, base = index(core / f"Core-{name}" / "metadata.parquet", name)
        write(table, out / f"{name}.parquet")
        listing[name] = {"label": label, "images": base, "samples": table.num_rows,
                         "columns": table.column_names}
        print(f"{name}: {table.num_rows:,} samples -> {(out / f'{name}.parquet').stat().st_size / 1e6:.1f} MB", flush=True)
    (out / "datasets.json").write_text(json.dumps(listing, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
