"""Measure how one Major TOM sample is obtained from each archive (input for GIF 5).

Cell 451U_946L (Snowbird, Utah; workshop venue), Sentinel-2 L2A, window = the v2 1056 px footprint.
Two races:
  same    the product Major TOM Core-S2L2A holds (2023-04-15, tile 12TVK) from every archive
  recent  the newest scene under MAX_CLOUD % in the last RECENT_DAYS days; Major TOM returns its fixed sample
Per lane: search, access, read 4 bands (B04 B03 B02 at 10 m + scene classes), cloud mask; plus the HTTP
requests and bytes GDAL fetched. Reads follow cloud-native practice: one request per header
(GDAL_INGESTED_BYTES_AT_OPEN), bands fetched concurrently, HTTP/2 multiplexing, exact window only.
Lanes that cannot log in record the error instead of a time.

Usage (from docs/animations; CDSE keys loaded into the environment, never printed):
  set -a; source ~/.config/cdse/s3.env; set +a
  python acquire_snowbird.py --race same --runs 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CELL = "451U_946L"
CRS = "EPSG:32612"
WINDOW = (434640.0, 4484220.0, 445200.0, 4494780.0)   # v2 1056 px window of the cell
BBOX = (-111.80, 40.49, -111.63, 40.61)               # lon/lat box around it, for searches
TILE = "12TVK"                                        # the MGRS tile that contains the whole window
SAME_DATE = "2023-04-15"
RECENT_DAYS, MAX_CLOUD = 60, 10
CORE_META = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet"
LANES = ["major-tom", "planetary-computer", "earth-search", "cdse", "gee"]

os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",       # no directory listing per file
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",            # header in one request
    "GDAL_HTTP_VERSION": "2", "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "FALSE",                               # measure cold reads (each run is a fresh process)
    "CPL_VSIL_NETWORK_STATS_ENABLED": "YES", "CPL_VSIL_SHOW_NETWORK_STATS": "YES",   # requests + bytes at exit
    "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu", "AWS_HTTPS": "YES", "AWS_VIRTUAL_HOSTING": "FALSE",
})


# ---------- bookkeeping ----------

def network_stats(stderr: str) -> tuple[int, int]:
    """(requests, bytes) from the JSON GDAL prints at exit ("Network statistics: {...}")."""
    if "Network statistics:" not in stderr:
        return 0, 0
    stats = json.JSONDecoder().raw_decode(stderr.split("Network statistics:", 1)[1].strip())[0]
    methods = stats.get("methods", {})
    return (sum(m.get("count", 0) for m in methods.values()),
            sum(m.get("downloaded_bytes", 0) for m in methods.values()))


class Clock:
    """Records named steps: with clock.step("read"): ..."""
    def __init__(self) -> None:
        self.steps: list[tuple[str, float]] = []
        self.source, self.acquired = "", ""          # product id and acquisition date the lane found
        self.parquet_bytes = None                     # Major TOM: size of the column chunks read

    def step(self, name: str):
        clock = self

        class _Step:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                clock.steps.append((name, time.perf_counter() - self.t0))
        return _Step()


def search_dates(race: str) -> str:
    if race == "same":
        return SAME_DATE
    today = dt.date.today()
    return f"{today - dt.timedelta(days=RECENT_DAYS)}/{today}"


# ---------- reading ----------

def read_window(href: str) -> np.ndarray:
    """One band, exactly the cell's window (GDAL fetches only the tiles it touches)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(href) as src:
        return src.read(1, window=from_bounds(*WINDOW, src.transform))


def read_bands(hrefs: dict[str, str], clock: Clock) -> dict[str, np.ndarray]:
    """All bands concurrently: one dataset per thread; GDAL releases the GIL while waiting."""
    with clock.step("read"):
        with ThreadPoolExecutor(len(hrefs)) as pool:
            return dict(zip(hrefs, pool.map(read_window, hrefs.values())))


def stac_item(url: str, collection: str, race: str, clock: Clock):
    """Search a STAC API: the target product (race 'same') or the newest clear scene (race 'recent')."""
    from pystac_client import Client
    with clock.step("search"):
        search = Client.open(url).search(collections=[collection], bbox=BBOX, datetime=search_dates(race),
                                         max_items=200)
        items = [i for i in search.item_collection() if TILE in i.id
                 and (race == "same" or i.properties.get("eo:cloud_cover", 100) < MAX_CLOUD)]
        items.sort(key=lambda i: i.datetime, reverse=True)
    if not items:
        raise RuntimeError("no matching scene")
    clock.source, clock.acquired = items[0].id, items[0].datetime.date().isoformat()
    return items[0]


# ---------- lanes ----------

def lane_major_tom(race: str, clock: Clock) -> dict[str, np.ndarray]:
    """Major TOM Core: find the cell in metadata.parquet, then read one row group's band columns.

    The sample is fixed: both races return the same 2023 product.
    """
    import fsspec
    import pyarrow.parquet as pq
    from fsspec.parquet import open_parquet_file
    from rasterio.io import MemoryFile
    row, col = 451, -946                             # CELL as signed integers (451U, 946L)
    with clock.step("search"):                       # integer filters match the file's row order, so
        with fsspec.open(CORE_META, block_size=2 ** 20) as f:   # row-group statistics skip all but 1-2 groups
            meta = pq.read_table(f, columns=["grid_cell", "product_id", "timestamp", "parquet_url", "parquet_row"],
                                 filters=[("grid_row_u", "==", row), ("grid_col_r", "==", col)]).to_pylist()[0]
    clock.source = meta["product_id"]
    clock.acquired = dt.datetime.strptime(meta["timestamp"][:8], "%Y%m%d").date().isoformat()
    cols = ["B04", "B03", "B02", "cloud_mask"]
    out = {}
    with clock.step("read"):
        with open_parquet_file(meta["parquet_url"], columns=cols, row_groups=[meta["parquet_row"]]) as f:
            pf = pq.ParquetFile(f)
            table = pf.read_row_group(meta["parquet_row"], columns=cols)
            group = pf.metadata.row_group(meta["parquet_row"])
        for c in cols:
            with MemoryFile(table[c][0].as_py()) as m, m.open() as src:
                out[c] = src.read(1)
    clock.parquet_bytes = sum(group.column(i).total_compressed_size for i in range(group.num_columns)
                              if group.column(i).path_in_schema in cols)
    return out


def lane_planetary_computer(race: str, clock: Clock) -> dict[str, np.ndarray]:
    import planetary_computer
    item = stac_item("https://planetarycomputer.microsoft.com/api/stac/v1", "sentinel-2-l2a", race, clock)
    with clock.step("access"):                       # free token: sign the asset URLs
        item = planetary_computer.sign(item)
    return read_bands({b: item.assets[b].href for b in ["B04", "B03", "B02", "SCL"]}, clock)


def lane_earth_search(race: str, clock: Clock) -> dict[str, np.ndarray]:
    item = stac_item("https://earth-search.aws.element84.com/v1", "sentinel-2-c1-l2a", race, clock)
    keys = {"B04": "red", "B03": "green", "B02": "blue", "SCL": "scl"}
    return read_bands({b: item.assets[k].href for b, k in keys.items()}, clock)


def lane_cdse(race: str, clock: Clock) -> dict[str, np.ndarray]:
    """CDSE: STAC search, then windowed JP2 reads straight from the eodata S3 bucket (keys from env)."""
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise RuntimeError("no CDSE S3 keys in the environment")
    item = stac_item("https://stac.dataspace.copernicus.eu/v1", "sentinel-2-l2a", race, clock)
    keys = {"B04": "B04_10m", "B03": "B03_10m", "B02": "B02_10m", "SCL": "SCL_20m"}
    return read_bands({b: item.assets[k].href.replace("s3://", "/vsis3/") for b, k in keys.items()}, clock)


def lane_gee(race: str, clock: Clock) -> dict[str, np.ndarray]:
    """Earth Engine: one computePixels call on the window's exact UTM grid (no resampling)."""
    import ee
    with clock.step("access"):
        ee.Initialize(project="wiki-eo")
    with clock.step("search"):
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(ee.Filter.eq("MGRS_TILE", TILE))
        if race == "same":
            col = col.filterDate(SAME_DATE, "2023-04-16")
        else:
            today = dt.date.today()
            col = (col.filterDate(str(today - dt.timedelta(days=RECENT_DAYS)), str(today + dt.timedelta(days=1)))
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD)).sort("system:time_start", False))
        image = ee.Image(col.first())
        props = image.toDictionary(["PRODUCT_ID", "system:time_start"]).getInfo()
    clock.source = props.get("PRODUCT_ID", "")
    clock.acquired = dt.datetime.utcfromtimestamp(props["system:time_start"] / 1000).date().isoformat()
    with clock.step("read"):
        arr = ee.data.computePixels({
            "expression": image.select(["B4", "B3", "B2", "SCL"]), "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": 1056, "height": 1056}, "crsCode": CRS,
                     "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": WINDOW[0],
                                         "shearY": 0, "scaleY": -10, "translateY": WINDOW[3]}}})
    return {"B04": arr["B4"] + 1000, "B03": arr["B3"] + 1000, "B02": arr["B2"] + 1000,   # harmonised: offset removed
            "SCL": arr["SCL"]}


# ---------- one run ----------

def cloud_fraction(bands: dict[str, np.ndarray], clock: Clock) -> float:
    """Share of cloud pixels: SCL classes 8, 9, 10 (cloud medium/high, cirrus), or Core's cloud mask."""
    with clock.step("cloud mask"):
        if "SCL" in bands:
            return float(np.isin(bands["SCL"], [8, 9, 10]).mean())
        return float((bands["cloud_mask"] > 0).mean())


def save_chip(path: Path, bands: dict[str, np.ndarray]) -> None:
    """Small true-colour PNG of what the lane returned (same stretch for every lane)."""
    from PIL import Image
    rgb = np.stack([bands[b].astype(np.float32) for b in ("B04", "B03", "B02")], -1)
    rgb = np.clip((rgb - 1000) / 14000, 0, 1) ** (1 / 1.4)         # L2A DN keep the +1000 offset
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((352, 352), Image.Resampling.LANCZOS).save(path)


def run_lane(lane: str, race: str, run: int) -> None:
    """One lane, once, in this process; prints the record as JSON (the parent adds network stats)."""
    clock = Clock()
    record = {"race": race, "lane": lane, "run": run}
    t0 = time.perf_counter()
    try:
        bands = globals()[f"lane_{lane.replace('-', '_')}"](race, clock)
        record["cloud_fraction"] = cloud_fraction(bands, clock)
        if run <= 0:
            save_chip(DATA / f"chip_{race}_{lane}.png", bands)
    except Exception as e:                           # login missing, quota, no scene: record, do not invent
        record["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    record.update({"source": clock.source, "acquired": clock.acquired, "steps": clock.steps,
                   "total_s": time.perf_counter() - t0, "parquet_bytes": clock.parquet_bytes,
                   "measured": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(json.dumps(record), flush=True)


def measure(lane: str, race: str, run: int) -> dict:
    """Run one lane in a fresh process; add GDAL's network statistics (Major TOM: parquet chunk sizes)."""
    proc = subprocess.run([sys.executable, __file__, "--race", race, "--lane", lane, "--run", str(run)],
                          capture_output=True, text=True)
    record = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith('{"race"')))
    requests, nbytes = network_stats(proc.stdout + proc.stderr)     # GDAL prints its stats at exit
    if nbytes:
        record.update(requests=requests, bytes=nbytes, bytes_from="GDAL network statistics")
    else:
        record.update(requests=None, bytes=record["parquet_bytes"], bytes_from="parquet column chunks")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--race", choices=["same", "recent"], default="same")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--lane", help="internal: run one lane once in this process")
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--lanes", nargs="+", default=LANES, help="subset of lanes to measure")
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    if args.lane:
        run_lane(args.lane, args.race, args.run)
        return
    for run in range(args.runs):                     # rounds interleave lanes, so network drift hits all
        for lane in args.lanes:
            record = measure(lane, args.race, run)
            with open(DATA / "acquisition_runs.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(lane, record.get("error", "ok"), round(record["total_s"], 2), "s", record["requests"], "req",
                  round((record["bytes"] or 0) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
