"""Footprint index of each Core dataset, for the Raster Viewer: where its samples are and where each one is stored.

    <core>/Core-<name>/metadata.parquet (one row per sample: grid_row_u, grid_col_r, parquet_url, parquet_row, ...)
      -> parquet_url -> part number (every URL is <base>/images/part_NNNNN.parquet; checked)
      -> rows sorted by grid_row_u, grid_col_r
      -> footprints/<name>.parquet: grid_row_u, grid_col_r, part, parquet_row, timestamp, + cloud_cover / orbit_state,
         centre_lat, centre_lon, epsg: every image is 10,680 m square in its UTM zone (epsg), centred there exactly
         (checked against the GeoTIFF bounds), so the app places it on the map without reading a band
         row groups of 65,536 rows with statistics and a page index, no value dictionaries (as the climate files):
         the app reads the two position columns for the coverage map and, for a click, the page holding the cell
      -> footprints/datasets.json: per dataset its label, the URL base of its image files, its samples
    Core-VIIRS-Nighttime-Light/INDEX.parquet on Hugging Face (one row per tile: id MT10_<cell>, year, shard, offset,
    size, in no cell order)
      -> footprints/VIIRS.parquet: grid_row_u, grid_col_r, year, shard, offset, size, sorted by cell; the app reads a
         tile with one range request into its zip shard (viirs.js)

    AlphaEarth (tge-labs/aef on source.coop): aef_index.parquet, one row per COG tile and year (81,920 m UTM squares,
    the same 34,266 squares every year) + the Core-S2L2A cells
      -> footprints/AEF.parquet: per cell grid_row_u, grid_col_r, centre_lat, centre_lon, year (its Sentinel-2 year,
         within 2017-2025), tile_row, tile_rows: where its tile's rows are in AEF-tiles.parquet; the tile holding the
         cell centre, in the cell's own UTM zone where there is one
      -> footprints/AEF-tiles.parquet: tile, year, epsg, west, north, path: one row per tile and year, by tile

    python build_index.py /data/Major-TOM <out folder> [name ...]   (Slurm, .venv311; names: rebuild only those)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASETS = {"S2L2A": "Sentinel-2 L2A", "S2L1C": "Sentinel-2 L1C", "S1RTC": "Sentinel-1 RTC", "DEM": "COP-DEM"}
EXTRA = ["timestamp", "cloud_cover", "orbit_state"]                      # kept where a dataset has them
ROW_GROUP = 65_536
VIIRS = "https://huggingface.co/datasets/Major-TOM/Core-VIIRS-Nighttime-Light/resolve/main/"
AEF = "https://data.source.coop/tge-labs/aef/v1/annual/"
AEF_TILE = 81_920                                            # metres: every AlphaEarth COG


def index(metadata: Path, name: str) -> tuple[pa.Table, str]:
    """The footprint table of one dataset, and the URL base of its image files."""
    have = pq.read_schema(metadata).names
    table = pq.read_table(metadata, columns=["grid_row_u", "grid_col_r", "parquet_url", "parquet_row",
                                             *[c for c in EXTRA if c in have], "centre_lat", "centre_lon", "crs"])
    base = f"https://huggingface.co/datasets/Major-TOM/Core-{name}/resolve/main/images/"
    urls = table.column("parquet_url").to_pylist()
    part = [re.fullmatch(re.escape(base) + r"part_(\d{5})\.parquet", u) for u in urls]
    if not all(part):
        raise ValueError(f"{metadata}: a parquet_url outside {base}part_NNNNN.parquet")
    out = pa.table({"grid_row_u": pc.cast(table.column("grid_row_u"), pa.int16()),
                    "grid_col_r": pc.cast(table.column("grid_col_r"), pa.int16()),
                    "part": pa.array(np.array([int(m[1]) for m in part], np.int16)),
                    "parquet_row": pc.cast(table.column("parquet_row"), pa.int16()),
                    **{c: table.column(c) for c in EXTRA if c in have},
                    "centre_lat": pc.cast(table.column("centre_lat"), pa.float32()),     # 4e-6 degrees: under 1 m
                    "centre_lon": pc.cast(table.column("centre_lon"), pa.float32()),
                    "epsg": pa.array(epsg(table.column("crs").to_pylist()), pa.int16())})
    if "cloud_cover" in out.column_names:
        out = out.set_column(out.schema.get_field_index("cloud_cover"), "cloud_cover",
                             pc.cast(out.column("cloud_cover"), pa.float32()))
    return out.sort_by([("grid_row_u", "ascending"), ("grid_col_r", "ascending")]), base


def epsg(crs: list[str]) -> np.ndarray:
    """EPSG codes from the metadata's crs strings ("EPSG:32701"; Core-S1RTC writes "EPSG:EPSG:32752")."""
    codes = np.array([int(c.rsplit(":", 1)[1]) for c in crs])
    if not np.all(((codes > 32600) & (codes <= 32660)) | ((codes > 32700) & (codes <= 32760))):
        raise ValueError("a crs outside the UTM zones (EPSG 32601-32660, 32701-32760)")
    return codes.astype(np.int16)


def viirs() -> pa.Table:
    """The VIIRS footprint table, from the dataset's INDEX.parquet: the cell from the tile id, the tile's place."""
    with fsspec.open(f"{VIIRS}INDEX.parquet", "rb", block_size=2**24) as f:
        table = pq.read_table(f, columns=["id", "year", "shard", "offset", "size"])
    parts = pc.extract_regex(table.column("id"), r"^MT10_(?P<r>\d+)(?P<u>[UD])_(?P<c>\d+)(?P<l>[LR])$")
    if parts.null_count:
        raise ValueError("a VIIRS id outside MT10_<row><U|D>_<col><L|R>")
    sign = lambda letter, plus: pc.if_else(pc.equal(pc.struct_field(parts, letter), plus), 1, -1)
    row = pc.multiply(pc.cast(pc.struct_field(parts, "r"), pa.int16()), pc.cast(sign("u", "U"), pa.int16()))
    col = pc.multiply(pc.cast(pc.struct_field(parts, "c"), pa.int16()), pc.cast(sign("l", "R"), pa.int16()))
    out = pa.table({"grid_row_u": row, "grid_col_r": col, "year": pc.cast(table.column("year"), pa.int16()),
                    "shard": table.column("shard"), "offset": pc.cast(table.column("offset"), pa.int64()),
                    "size": pc.cast(table.column("size"), pa.int32())})
    return out.sort_by([("grid_row_u", "ascending"), ("grid_col_r", "ascending"), ("year", "ascending")])


def aef(core: Path) -> tuple[pa.Table, pa.Table]:
    """AlphaEarth: the cell table (Core-S2L2A cells -> the tile holding each centre) and the tile table."""
    import pandas as pd
    from pyproj import Transformer

    with fsspec.open(f"{AEF}aef_index.parquet", "rb", block_size=2**24) as f:
        idx = pq.read_table(f, columns=["path", "year", "crs", "utm_west", "utm_south"]).to_pandas()
    idx["epsg"] = idx.crs.str.split(":").str[-1].astype(int)
    idx["path"] = idx.path.str.replace("s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/", "", regex=False)
    squares = idx.drop_duplicates(["epsg", "utm_west", "utm_south"])[["epsg", "utm_west", "utm_south"]]
    squares = squares.sort_values(["epsg", "utm_south", "utm_west"]).reset_index(drop=True)
    squares["tile"] = np.arange(len(squares), dtype=np.int32)
    tiles = idx.merge(squares, on=["epsg", "utm_west", "utm_south"]).sort_values(["tile", "year"]).reset_index(drop=True)

    cells = pq.read_table(core / "Core-S2L2A" / "metadata.parquet",
                          columns=["grid_row_u", "grid_col_r", "centre_lat", "centre_lon", "crs", "timestamp"]).to_pandas()
    cells["own"] = epsg(cells.crs.tolist()).astype(int)
    cells["tile"] = -1
    for own_zone_first in (True, False):                     # a cell near a zone edge may lie in a neighbour's square
        for code, group in squares.groupby("epsg"):
            pick = (cells.tile < 0) & ((cells.own == code) if own_zone_first else (abs(cells.own - code) == 1))
            if not pick.any():
                continue
            x, y = Transformer.from_crs(4326, int(code), always_xy=True).transform(
                cells.centre_lon[pick].to_numpy(), cells.centre_lat[pick].to_numpy())
            west, south = group.utm_west.to_numpy(), group.utm_south.to_numpy()
            inside = ((x[:, None] >= west) & (x[:, None] < west + AEF_TILE) &
                      (y[:, None] >= south) & (y[:, None] < south + AEF_TILE))
            hit = inside.any(1)
            cells.loc[cells.index[pick][hit], "tile"] = group.tile.to_numpy()[inside[hit].argmax(1)]
    cells = cells[cells.tile >= 0]
    first = tiles.reset_index().groupby("tile")["index"].agg(["min", "count"])
    year = cells.timestamp.str[:4].astype(int).clip(2017, 2025)
    cell_table = pa.table({"grid_row_u": pa.array(cells.grid_row_u.to_numpy(np.int16)),
                           "grid_col_r": pa.array(cells.grid_col_r.to_numpy(np.int16)),
                           "centre_lat": pa.array(cells.centre_lat.to_numpy(np.float32)),
                           "centre_lon": pa.array(cells.centre_lon.to_numpy(np.float32)),
                           "year": pa.array(year.to_numpy(np.int16)),
                           "tile_row": pa.array(first.loc[cells.tile, "min"].to_numpy(np.int32)),
                           "tile_rows": pa.array(first.loc[cells.tile, "count"].to_numpy(np.int16))})
    tile_table = pa.table({"tile": pa.array(tiles.tile.to_numpy(np.int32)), "year": pa.array(tiles.year.to_numpy(np.int16)),
                           "epsg": pa.array(tiles.epsg.to_numpy(np.int32)),
                           "west": pa.array(tiles.utm_west.to_numpy(np.float64)),
                           "north": pa.array((tiles.utm_south + AEF_TILE).to_numpy(np.float64)),
                           "path": pa.array(tiles.path.tolist())})
    return cell_table.sort_by([("grid_row_u", "ascending"), ("grid_col_r", "ascending")]), tile_table


def write(table: pa.Table, path: Path) -> None:
    keys = ["grid_row_u", "grid_col_r", "orbit_state", "epsg", "shard"]
    pq.write_table(table, path, row_group_size=ROW_GROUP, compression="zstd", write_statistics=True,
                   write_page_index=True, use_dictionary=[c for c in keys if c in table.column_names],
                   data_page_size=4096)


def main(core: str, out: str, *only: str) -> None:
    core, out = Path(core), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    listing = json.loads((out / "datasets.json").read_text()) if only else {}
    for name in only or [*DATASETS, "VIIRS", "AEF"]:
        if name == "AEF":
            table, tile_table = aef(core)
            write(tile_table, out / "AEF-tiles.parquet")
            entry = {"label": "AlphaEarth Embeddings", "kind": "aef", "files": AEF, "tiles": "AEF-tiles.parquet",
                     "years": [2017, 2025], "link": "https://source.coop/tge-labs/aef"}
        elif name == "VIIRS":
            table = viirs()
            entry = {"label": "VIIRS Nighttime Light", "kind": "viirs", "files": VIIRS,
                     "link": "https://huggingface.co/datasets/Major-TOM/Core-VIIRS-Nighttime-Light"}
        else:
            table, base = index(core / f"Core-{name}" / "metadata.parquet", name)
            entry = {"label": DATASETS[name], "images": base}
        write(table, out / f"{name}.parquet")
        listing[name] = {**entry, "samples": table.num_rows, "columns": table.column_names}
        print(f"{name}: {table.num_rows:,} samples -> {(out / f'{name}.parquet').stat().st_size / 1e6:.1f} MB", flush=True)
    order = [*DATASETS, "VIIRS", "AEF"]
    listing = {k: listing[k] for k in sorted(listing, key=order.index)}
    (out / "datasets.json").write_text(json.dumps(listing, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
