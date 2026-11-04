"""Intro animation for Major TOM Climate and Major TOM Core-Weather, version 2: 66 s at 30 fps, as a square
(1080 x 1080, the main version) and a wide (1920 x 1080) MP4 from one code path.

    logos   SVGs (docs/brand, apps/climate-globe) -> transparent PNGs, rasterised by Chromium     (python-last env)
    prep    release files and one live read -> small arrays in WORK/data                          (miko-torch env)
              ERA5-Monthly climatology_overviews/MM.parquet -> 12 monthly 2 m temperature normals, 4 x 4-cell blocks
              ERA5-Monthly by_cell/ -> monthly 2 m temperature 1940-2025 at Loch Lomond (cell 625U_30L)
              Core-S2L2A metadata + thumbnail + Core-Weather hourly/ -> the Loch Lomond sample of 2022-08-10
              source.coop CPC-Monthly by_cell/, read three times with pandas + fsspec -> bytes, requests, seconds
              Basemap coastlines (longitude, latitude) for the globe
    shapes  Natural Earth 1:10m land and European lakes around Loch Lomond -> WORK/data/region.npz   (python-last env)
    render  frame(i, layout) -> WORK/frames-<layout>/%05d.png (one process per CPU) -> ffmpeg (H.264, yuv420p)

Scenes (seconds): the flag fills large and centred, cell by cell (observations, then the fill), moves into the
logo, "Major TOM" and "climate" fade in 0-7.6; globe zooming into Loch Lomond, where 21 rows of Major TOM cells
(majortom/cells.py) fill column by column and cell 625U_30L is picked out 7.6-15.2; card "Dataset 1 of 2"
15.2-19.2; ERA5 temperature normals on the globe, then six normals on six globes, 0.3 s a month, and the release's
variable count 19.2-26.7; monthly ERA5 temperature at Loch Lomond 1940-2025 26.7-31.1; card "Dataset 2 of 2" 31.1-35.1; a Sentinel-2 image and its hourly weather 35.1-43.1; sources
by kind of data, with emojis 43.1-50.1; reading one cell's series over HTTPS 50.1-57.2; size and access 57.2-62.7;
end card 62.7-66.

Every position comes from LAYOUTS (one table per layout), so square and wide share all drawing code.

Usage (from the repository root, through Slurm):
  srun -c 4 --mem=8G /home/miko/miniconda3/envs/python-last/bin/python docs/animations/intro_climate_weather.py logos
  srun -c 2 --mem=8G /home/miko/miniconda3/envs/python-last/bin/python docs/animations/intro_climate_weather.py shapes
  srun -c 4 --mem=16G /home/miko/miniconda3/envs/miko-torch/bin/python docs/animations/intro_climate_weather.py prep
  srun -c 1 --mem=2G /home/miko/miniconda3/envs/miko-torch/bin/python docs/animations/intro_climate_weather.py emojis
  srun -c 2 --mem=8G /home/miko/miniconda3/envs/miko-torch/bin/python docs/animations/intro_climate_weather.py six
  srun -c 16 --mem=32G /home/miko/miniconda3/envs/miko-torch/bin/python docs/animations/intro_climate_weather.py render
  (render square|wide renders one layout; render square 400 one frame, for checking)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "docs" / "evidence"))

RELEASE = Path("/data/Major-TOM/climate-release")
CORE = Path("/data/Major-TOM")
WEATHER = CORE / "Core-Weather" / "merged"
WORK = CORE / "animations" / "intro-v2"
DATA = WORK / "data"
SHOTS = REPO / "screenshots"
FPS = 30
LOGO_SVG = REPO / "docs" / "brand" / "mt-climate.svg"

SCENES = {"flag": (0.0, 7.6), "grid": (7.6, 15.2), "card1": (15.2, 19.2), "normals": (19.2, 26.7),
          "series": (26.7, 31.1), "card2": (31.1, 35.1), "weather": (35.1, 43.1), "sources": (43.1, 50.1),
          "access": (50.1, 57.2), "size": (57.2, 62.7), "end": (62.7, 66.0)}
# Explanatory text stays fully visible for at least 2.5 s (cards 3 s) before its scene fades out.
FADE = 0.3                                              # through the page colour, at both ends of each scene
PLACE, CELL, ROW, COL = "Loch Lomond", "625U_30L", 625, -30
BAND = (ROW + 1002) * 20 // 2004                        # by_cell/ band of grid rows (16)

# The Climate Globe's loading flag (apps/climate-globe/index.html, #loader): 20 bars, 4 columns of 5, in the order
# the app lights them (apps/climate-globe/app.js, animateFlag: one bar per 70 ms, column after column, top to bottom)
FLAG_ON = ["#2e3192", "#0071bc", "#39b54a", "#f15a24"]  # --mt-navy, --mt-blue, --mt-green, --mt-orange
FLAG_OFF = "#e5e7eb"                                    # --line
FLAG_BARS = [
    "420.45 14.5 402.24 23.6 402.24 34.39 420.45 25.29", "420.45 28.89 402.24 37.99 402.24 48.79 420.45 39.69",
    "420.45 43.28 402.24 52.38 402.24 63.18 420.45 54.08", "420.45 57.67 402.24 66.78 402.24 77.57 420.45 68.47",
    "420.45 72.07 402.24 81.17 402.24 92.6 420.45 83.04",
    "424.04 14.26 446.45 22.88 446.45 33.67 424.04 25.05", "424.04 28.65 446.45 37.26 446.45 48.06 424.04 39.45",
    "424.04 43.04 446.45 51.66 446.45 62.46 424.04 53.84", "424.04 57.43 446.45 66.05 446.45 76.85 424.04 68.23",
    "424.04 71.83 446.45 80.45 446.45 91.84 424.04 82.79",
    "450.04 22.91 472.45 14.29 472.45 25.08 450.04 33.7", "450.04 37.3 472.45 28.68 472.45 39.48 450.04 48.1",
    "450.04 51.69 472.45 43.07 472.45 53.87 450.04 62.49", "450.04 66.08 472.45 57.47 472.45 68.27 450.04 76.88",
    "450.04 80.48 472.45 71.86 472.45 82.82 450.04 91.88",
    "476.04 14.5 494.24 23.6 494.24 34.39 476.04 25.29", "476.04 28.89 494.24 37.99 494.24 48.79 476.04 39.69",
    "476.04 43.28 494.24 52.38 494.24 63.18 476.04 54.08", "476.04 57.67 494.24 66.78 494.24 77.57 476.04 68.47",
    "476.04 72.07 494.24 81.17 494.24 92.6 476.04 83.04"]

# Text sizes in points (1 pt = 1.39 px at dpi 100). The square is seen about 360 px wide on a phone, a third of its
# size, so nothing but the licence line is below 16 pt (7 px on the phone).
T_TITLE, T_SUB, T_BODY, T_SMALL, T_FOOT = 30, 19, 18, 16, 12

# One table per layout: canvas size and every box a scene draws in, in pixels (x, y from the top left).
LAYOUTS = {
    "square": {
        "size": (1080, 1080), "margin": 48,
        "logo": (90, 400, 900), "logo_sub": 690,
        "grid_view": (48, 150, 984, 805), "grid_r": 375, "grid_text": (540, 992),
        "card": (540, 470),
        "normals_globe": (540, 535, 345), "normals_month": (540, 965), "normals_bar": (290, 1000, 500),
        "normals_slots": [(212, 345, 140), (540, 345, 140), (868, 345, 140), (212, 735, 140), (540, 735, 140),
                          (868, 735, 140)], "normals_month_six": 975, "normals_total": (540, 1040),
        "series_chart": (100, 230, 840, 680), "series_note": 1040,
        "weather_image": (48, 160, 420), "weather_info": (500, 180, 74),
        "weather_t": (100, 650, 800, 140), "weather_c": (100, 880, 800, 110), "weather_note": 1050,
        "sources_cols": [["Reanalysis"], ["Satellite", "Rain gauges"]], "sources_box": (48, 165, 984, 820),
        "sources_note": 1045, "tile_h": 196,
        "code": (48, 160, 984, 300), "access_chart": (100, 540, 900, 300), "access_counts": (48, 960),
        "size_blocks": [(48, 230), (48, 510)], "size_formats": (48, 790), "size_icon": (790, 190, 200),
        "end_logo": (190, 120, 700), "end_lines": (540, 400, 100), "end_rule": 640, "end_logos": (150, 690),
        "end_credit": (540, 845), "end_foot": 1020,
    },
    "wide": {
        "size": (1920, 1080), "margin": 96,
        "logo": (360, 360, 1200), "logo_sub": 700,
        "grid_view": (840, 60, 1000, 960), "grid_r": 420, "grid_text": (96, 520),
        "card": (960, 470),
        "normals_globe": (1320, 560, 440), "normals_month": (96, 640), "normals_bar": (96, 760, 600),
        "normals_slots": [(900, 335, 170), (1280, 335, 170), (1660, 335, 170), (900, 795, 170), (1280, 795, 170),
                          (1660, 795, 170)], "normals_month_six": 640, "normals_total": (96, 760),
        "series_chart": (130, 200, 1660, 700), "series_note": 1040,
        "weather_image": (96, 180, 520), "weather_info": (96, 735, 40),
        "weather_t": (870, 220, 860, 300), "weather_c": (870, 640, 860, 240), "weather_note": 1000,
        "sources_cols": [["Reanalysis"], ["Satellite"], ["Rain gauges"]], "sources_box": (96, 165, 1728, 820),
        "sources_note": 1040, "tile_h": 196,
        "code": (96, 180, 900, 300), "access_chart": (1100, 200, 720, 420), "access_counts": (96, 600),
        "size_blocks": [(96, 300), (1000, 300)], "size_formats": (96, 680), "size_icon": (560, 240, 200),
        "end_logo": (510, 110, 900), "end_lines": (960, 400, 100), "end_rule": 640, "end_logos": (560, 690),
        "end_credit": (960, 845), "end_foot": 1020,
    },
}
L: dict = {}                                            # the layout being drawn (set per frame)


# ---------- logos (Chromium: the SVGs use paths, matplotlib cannot read them) ----------

def logos() -> None:
    """Transparent PNGs of the wordmark (the logo without its bars), the Earth Compress logo and the Asterisk mark."""
    from playwright.sync_api import sync_playwright

    DATA.mkdir(parents=True, exist_ok=True)
    wordmark = re.sub(r"<polygon[^>]*/>", "", LOGO_SVG.read_text())     # the bars are drawn by the animation
    jobs = [(wordmark, 4000, "wordmark.png"), ((REPO / "docs/brand/earth-compress.svg").read_text(), 1600,
            "earth-compress.png"), ((REPO / "apps/climate-globe/asterisk-mark.svg").read_text(), 600, "asterisk-mark.png"),
            ((REPO / "docs/brand/mt-climate-icon.svg").read_text(), 800, "mt-climate-icon.png")]
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        for text, width, name in jobs:
            text = re.sub(r"<\?xml[^>]*\?>", "", text)
            page.set_content(f'<html><body style="margin:0;background:transparent">'
                             f'<div id="box" style="width:{width}px;line-height:0">{text}</div></body></html>')
            page.eval_on_selector("#box svg", "s => { s.style.width = '100%'; s.style.height = 'auto'; }")
            page.locator("#box svg").screenshot(path=str(DATA / name), omit_background=True)
            print(name, "written", flush=True)


REGION = (-8.5, 54.8, -1.0, 57.6)                       # lon/lat box of the 1:10m land and lakes (grid scene)


def shapes() -> None:
    """Natural Earth 1:10m land and European lakes (public domain, through cartopy's downloader) clipped to REGION:
    exterior rings as (lon, lat) arrays. Loch Lomond is in ne_10m_lakes_europe."""
    import cartopy.io.shapereader as shp
    from shapely.geometry import box

    DATA.mkdir(parents=True, exist_ok=True)
    region = box(*REGION)
    out = {}
    for layer, prefix in (("land", "land"), ("lakes_europe", "lake")):
        k = 0
        for geom in shp.Reader(shp.natural_earth("10m", "physical", layer)).geometries():
            if not geom.intersects(region):
                continue
            part = geom.intersection(region)
            for poly in getattr(part, "geoms", [part]):
                if poly.geom_type == "Polygon" and poly.area > 0:
                    out[f"{prefix}{k}"] = np.asarray(poly.exterior.coords, np.float32)
                    k += 1
        print(layer, k, "rings", flush=True)
    np.savez_compressed(DATA / "region.npz", **out)


# ---------- prep ----------

def _live_read(runs: int = 3) -> dict:
    """Read one cell's CPC series from source.coop with pandas + fsspec, `runs` times, counting HTTP requests and
    response bytes at aiohttp (which fsspec's https filesystem uses)."""
    import aiohttp
    import fsspec
    import pandas as pd

    calls = {"requests": 0, "bytes": 0}
    request, read = aiohttp.ClientSession._request, aiohttp.ClientResponse.read

    async def counted_request(self, *args, **kwargs):
        calls["requests"] += 1
        return await request(self, *args, **kwargs)

    async def counted_read(self):
        body = await read(self)
        calls["bytes"] += len(body)
        return body

    aiohttp.ClientSession._request, aiohttp.ClientResponse.read = counted_request, counted_read
    url = "https://data.source.coop/major-tom/climate/CPC-Monthly"
    out = []
    for _ in range(runs):
        calls.update(requests=0, bytes=0)
        start = time.perf_counter()
        https = fsspec.filesystem("https", skip_instance_cache=True)   # a new connection each run, no cache
        history = pd.read_parquet(f"{url}/by_cell/band={BAND}.parquet", filesystem=https,
                                  columns=["year", "month", "cpc_precip"],
                                  filters=[("grid_row_u", "=", ROW), ("grid_col_r", "=", COL)])
        out.append({"seconds": time.perf_counter() - start, **calls, "rows": len(history)})
        print("live read", out[-1], flush=True)
    aiohttp.ClientSession._request, aiohttp.ClientResponse.read = request, read
    history.sort_values(["year", "month"]).to_parquet(DATA / "access-history.parquet")
    file_size = fsspec.filesystem("https").size(f"{url}/by_cell/band={BAND}.parquet")
    return {"runs": out, "seconds": float(np.median([r["seconds"] for r in out])), "requests": out[-1]["requests"],
            "bytes": out[-1]["bytes"], "file_bytes": int(file_size), "rows": len(history),
            "measured": time.strftime("%Y-%m-%d")}


def _du(paths: list[Path]) -> int:
    return sum(int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True, check=True).stdout.split()[0])
               for p in paths if p.exists())


def emojis() -> None:
    """Each emoji of EMOJI as a PNG (Noto Color Emoji, a bitmap font matplotlib cannot draw), all on the same
    140 x 128 canvas so they share a baseline. Variation selectors are left out (PIL here has no raqm)."""
    from PIL import Image, ImageDraw, ImageFont

    DATA.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)   # its only bitmap size
    for key, char in EMOJI.items():
        im = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2, 2), char, font=font, embedded_color=True)
        im.crop((0, 4, 140, 132)).save(DATA / f"emoji-{key.replace(' ', '-')}.png")
    print(len(EMOJI), "emojis", flush=True)


def prep() -> None:
    import pandas as pd
    import pyarrow.parquet as pq
    from climate_figures import inside_earth
    from core_weather_figures import stretch, thumbnail, weather_rows
    from mpl_toolkits.basemap import Basemap

    DATA.mkdir(parents=True, exist_ok=True)

    # ERA5 2 m temperature normals, 1991-2020, blocks of 4 x 4 cells (block_row counts from the south)
    normals = np.full((12, 501, 1002), np.nan, np.float32)
    for m in range(1, 13):
        t = pq.read_table(RELEASE / "ERA5-Monthly" / "climatology_overviews" / f"{m:02d}.parquet",
                          columns=["block_row", "block_col", "era5_t2m_mean"]).to_pandas()
        normals[m - 1, 500 - t.block_row.to_numpy(), t.block_col.to_numpy()] = t.era5_t2m_mean.to_numpy() - 273.15
    share4 = inside_earth().astype(np.float32).reshape(501, 4, 1002, 4).mean((1, 3))
    np.savez_compressed(DATA / "normals.npz", t2m=normals, share=share4)
    print("normals", np.isfinite(normals[0]).sum(), "blocks", flush=True)

    # ERA5 monthly 2 m temperature at the place, every month of the record
    t = pq.read_table(RELEASE / "ERA5-Monthly" / "by_cell" / f"band={BAND:02d}.parquet",
                      columns=["year", "month", "era5_t2m"],
                      filters=[("grid_row_u", "=", ROW), ("grid_col_r", "=", COL)]).to_pandas()
    t = t.sort_values(["year", "month"]).reset_index(drop=True)
    t["era5_t2m"] -= 273.15
    t.to_parquet(DATA / "series-era5.parquet")
    print("series", len(t), "months", int(t.year.min()), int(t.year.max()), flush=True)

    # the Core-S2L2A sample, its thumbnail (contrast stretched) and the hourly weather around it
    s2 = pd.read_parquet(CORE / "Core-S2L2A" / "metadata.parquet", filters=[("grid_cell", "=", CELL)]).iloc[0]
    acquired = pd.Timestamp(s2.timestamp)
    stretch(thumbnail(s2.parquet_url, int(s2.parquet_row))).save(DATA / "sample-thumbnail.png")
    hourly = weather_rows(WEATHER, "hourly", CELL, acquired,
                          ["era5_t2m", "era5land_t2m", "era5_tcc", "era5_tp", "imerg_precip"])
    hourly.to_parquet(DATA / "sample-hourly.parquet")
    (DATA / "sample.json").write_text(json.dumps({
        "cell": CELL, "place": PLACE, "acquired": f"{acquired:%Y-%m-%d %H:%M}", "product_id": s2.product_id,
        "cloud_cover": float(s2.cloud_cover), "era5_tp_max_mm": float(hourly.era5_tp.max() * 1000),
        "imerg_max_mm": float(hourly.imerg_precip.max())}))
    print("sample", acquired, len(hourly), "hours", flush=True)

    # one cell's series read live from source.coop
    access = _live_read()
    access["climate_gb"] = {d: _du([RELEASE / d / p for p in ("cells.parquet", "series", "climatology", "by_cell",
                                                               "overviews", "climatology_overviews", "static.parquet")])
                            / 1e9 for d in ("CPC-Monthly", "IMERG-Monthly", "CAMS-Monthly", "ERA5-Static",
                                            "ERA5Land-Monthly", "ERA5-Monthly")}
    access["weather_gb"] = _du([WEATHER / "hourly", WEATHER / "daily", WEATHER / "acquisitions.parquet"]) / 1e9
    (DATA / "access.json").write_text(json.dumps(access, indent=1))
    print("access", json.dumps(access), flush=True)

    # coastlines in longitude and latitude, for the globe
    m = Basemap(projection="cyl", resolution="l")
    segments = [np.asarray(s, np.float32) for s in m.coastsegs]
    np.savez_compressed(DATA / "coast.npz", **{f"s{i}": s for i, s in enumerate(segments)})
    print("coast", len(segments), "segments", flush=True)


# ---------- drawing helpers ----------

def _setup() -> None:
    """IBM Plex Mono for everything (as the README figures), League Spartan registered for the Asterisk wordmark."""
    from climate_figures import use_major_tom_font
    from matplotlib import font_manager

    use_major_tom_font()
    for ttf in sorted((HERE / "fonts").glob("LeagueSpartan-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))


class Assets:
    """Everything a frame needs, loaded once per worker."""

    def __init__(self) -> None:
        import pandas as pd
        from climate_figures import COLS, D_KM, R_KM, ROWS
        from PIL import Image

        n = np.load(DATA / "normals.npz")
        self.normals, self.share4 = n["t2m"], n["share"]
        x = np.load(DATA / "six.npz")
        self.six = {"t2m": self.normals, **{k: x[k] for k in x.files}}
        self.scales = _six_scales()
        self.variables = json.loads((DATA / "variables.json").read_text())
        c = np.load(DATA / "coast.npz")
        self.coast = [c[k] for k in c.files]
        self.series = pd.read_parquet(DATA / "series-era5.parquet")
        self.hourly = pd.read_parquet(DATA / "sample-hourly.parquet")
        self.sample = json.loads((DATA / "sample.json").read_text())
        self.access = json.loads((DATA / "access.json").read_text())
        self.history = pd.read_parquet(DATA / "access-history.parquet")
        self.thumb = np.asarray(Image.open(DATA / "sample-thumbnail.png").convert("RGB"))
        self.wordmark = np.asarray(Image.open(DATA / "wordmark.png"))
        self.earth_compress = np.asarray(Image.open(DATA / "earth-compress.png"))
        self.mark = np.asarray(Image.open(DATA / "asterisk-mark.png"))
        self.climate_icon = np.asarray(Image.open(DATA / "mt-climate-icon.png"))
        self.emoji = {k: np.asarray(Image.open(DATA / f"emoji-{k.replace(' ', '-')}.png")) for k in EMOJI}
        self.bars = [np.array(p.split(), float).reshape(-1, 2) for p in FLAG_BARS]
        cells, self.flag_start, _ = pixel_fill_schedule()
        self.flag_bar = np.array([b for b, _ in cells])
        self.flag_quads = [q for _, q in cells]
        self.grid = dict(rows=ROWS, cols=COLS, r_km=R_KM, d_km=D_KM)
        self._grid_cells()
        r = np.load(DATA / "region.npz")
        self.land = [r[k] for k in r.files if k.startswith("land")]
        self.lakes = [r[k] for k in r.files if k.startswith("lake")]

    def _grid_cells(self) -> None:
        """21 rows of Major TOM cells around CELL, from majortom.cells (the grid formulas): in each row the cells
        within 10.5 of CELL's column widths of its centre. Fill order: column by column (west to east), each from
        north to south. Colour: the flag's four colours by column number, repeating, so each column's shift from row
        to row shows."""
        import importlib.util                                        # the module alone: majortom/__init__.py
                                                                     # needs Python 3.10+, miko-torch has 3.8
        spec = importlib.util.spec_from_file_location("majortom_cells", REPO / "majortom" / "cells.py")
        grid = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = grid                                # dataclasses look the module up by name
        spec.loader.exec_module(grid)
        Cell, cell_from_name, cells_in = grid.Cell, grid.cell_from_name, grid.cells_in

        self.target = cell_from_name(CELL)
        t = self.target
        lon_c = t.centroid[0]
        west, east = lon_c - 10.5 * t.dlon, lon_c + 10.5 * t.dlon
        south, north = Cell(t.row - 10, 0).lat + 1e-6, Cell(t.row + 10, 0).lat + 1e-6
        table = cells_in((west, south, east, north))
        self.cells = [Cell(int(r), int(c)) for r, c in zip(table.row, table.col)]
        self.cell_quads = [np.array([(c.lon, c.lat), (c.lon + c.dlon, c.lat), (c.lon + c.dlon, c.lat + c.dlat),
                                     (c.lon, c.lat + c.dlat)]) for c in self.cells]
        order = sorted(range(len(self.cells)), key=lambda k: (self.cells[k].col, -self.cells[k].row))
        self.fill_rank = np.empty(len(order), int)
        self.fill_rank[order] = np.arange(len(order))
        self.fill_band = np.array([c.col % 4 for c in self.cells])  # the four flag colours, repeating by column
        self.target_k = next(k for k, c in enumerate(self.cells) if c.name == CELL)
        self.target_quad = self.cell_quads[self.target_k]


ASSETS: Assets | None = None


def _canvas():
    """A figure of the layout's size with one axes in pixel coordinates, y pointing down."""
    import matplotlib.pyplot as plt
    from climate_figures import PAGE

    w, h = L["size"]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    return fig, ax


def _axes(fig, box: tuple) -> object:
    """Axes for a box (x, y, w, h) in pixels from the top left."""
    W, H = L["size"]
    x, y, w, h = box
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def _lvl(level: float, colour: str) -> str:
    """`colour` mixed from the page colour: `level` 0 is invisible, 1 is the colour itself."""
    from climate_figures import PAGE
    from style import mix

    return mix(PAGE, colour, float(np.clip(level, 0, 1)))


def _head(ax, title: str, subtitle: str, level: float = 1.0) -> None:
    from climate_figures import INK, INK_2

    m = L["margin"]
    ax.text(m, 82, title, fontsize=T_TITLE, fontweight="bold", color=_lvl(level, INK), va="baseline")
    ax.text(m, 124, subtitle, fontsize=T_SUB, color=_lvl(level, INK_2), va="baseline")


def _flag(ax, x0: float, y0: float, scale: float, lit: int, level: float = 1.0) -> None:
    """The flag's 20 bars at (x0, y0) px, `scale` px per SVG unit (logo viewBox units); the first `lit` bars on."""
    from matplotlib.patches import Polygon

    for k, points in enumerate(ASSETS.bars):
        colour = FLAG_ON[k // 5] if k < lit else FLAG_OFF
        ax.add_patch(Polygon(np.c_[x0 + points[:, 0] * scale, y0 + points[:, 1] * scale], closed=True,
                             facecolor=_lvl(level, colour), edgecolor="none", zorder=4))


def _logo(ax, x0: float, y0: float, width: float, lit: int = 20, t_word: float = 1.0) -> None:
    """The Major TOM Climate logo (viewBox 502.24 x 114.72) at (x0, y0) px: the flag's bars, `lit` of them on, and
    the wordmark wiped in from the left (`t_word` 0..1)."""
    from matplotlib.patches import Rectangle

    scale = width / 502.24
    if t_word > 0:
        im = ax.imshow(ASSETS.wordmark, extent=(x0, x0 + width, y0 + 114.72 * scale, y0), zorder=3,
                       interpolation="lanczos")
        im.set_clip_path(Rectangle((x0, y0), width * t_word, 114.72 * scale, transform=ax.transData))
    _flag(ax, x0, y0, scale, lit)


def _image(ax, img: np.ndarray, x: float, y: float, height: float, zorder: int = 3, alpha: float = 1.0) -> float:
    """Draw `img` with its top-left at (x, y) px and the given height; returns its width."""
    width = height * img.shape[1] / img.shape[0]
    ax.imshow(img, extent=(x, x + width, y + height, y), zorder=zorder, interpolation="lanczos", alpha=alpha)
    return width


# ---------- the globe (orthographic, by hand: cartopy is not in miko-torch) ----------

def _globe_lookup(lon0: float, lat0: float, n: int):
    """For an n x n picture of the globe centred on (lon0, lat0): which pixels are on the Earth, the 4 x 4 block
    (north-up row, column) and the grid cell (row, column) each shows, and a shade that darkens towards the limb."""
    g = ASSETS.grid
    v = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(v, -v)
    rho2 = x * x + y * y
    on = rho2 <= 1
    z = np.sqrt(np.clip(1 - rho2, 0, 1))
    p0, l0 = np.radians(lat0), np.radians(lon0)
    lat = np.arcsin(np.clip(z * np.sin(p0) + y * np.cos(p0), -1, 1))
    lon = l0 + np.arctan2(x, z * np.cos(p0) - y * np.sin(p0))
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    row = np.clip(np.floor(np.degrees(lat) * g["rows"] / 180).astype(int), -g["rows"] // 2, g["rows"] // 2 - 1)
    col = np.floor(lon * g["r_km"] * np.cos(lat) / g["d_km"]).astype(int)
    col = np.clip(col, -g["cols"] // 2, g["cols"] // 2 - 1)
    br, bc = 500 - (row + g["rows"] // 2) // 4, (col + g["cols"] // 2) // 4
    return on, br, bc, row, col, 0.82 + 0.18 * z


def _globe(ax, centre: tuple, rgb_of, lon0: float, lat0: float = 22.0, level: float = 1.0) -> None:
    """A globe of radius r px at (cx, cy): pixel colours from rgb_of(block rows, block cols) -> (..., 3) and a mask
    of pixels without a colour (drawn in the empty colour); coastlines on top, only on the visible side."""
    from climate_figures import PAGE
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb

    cx, cy, r = centre
    n = int(2 * r * 1.5)                                               # 1.5 x supersampled, smoothed by imshow
    on, br, bc, row, col, shade = _globe_lookup(lon0, lat0, n)
    rgb = rgb_of(br, bc, row, col) * shade[..., None]
    page = np.array(to_rgb(PAGE))
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = page + (rgb - page) * level
    rgba[..., 3] = on
    ax.imshow(rgba, extent=(cx - r, cx + r, cy + r, cy - r), interpolation="antialiased", zorder=2)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    lines = []
    for s in ASSETS.coast:
        lam, phi = np.radians(s[:, 0]), np.radians(s[:, 1])
        visible = np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(lam - l0) > 0
        if not visible.any():
            continue
        px = cx + r * np.cos(phi) * np.sin(lam - l0)
        py = cy - r * (np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(lam - l0))
        pts = np.c_[px, py]
        pts[~visible] = np.nan
        lines.append(pts)
    ax.add_collection(LineCollection(lines, colors=[(0, 0, 0, 0.5 * level)], linewidths=0.6, zorder=3))
    from matplotlib.patches import Circle

    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=_lvl(level, "#c3c2b7"), linewidth=1, zorder=3))


# ---------- scenes: each draws local time s (seconds since the scene started) ----------

# ---------- the flag's fill: observations to reanalysis, on each bar's own grid of cells ----------

FILL_START, FILL_LENGTH, CELL_FADE = 0.1, 2.4, 0.1      # cells start lighting 0.1-2.4 s; each fades in over 0.1 s


def bar_corners() -> list[tuple[np.ndarray, ...]]:
    """Each flag bar's corners in logo viewBox units, as (top left, top right, bottom left, bottom right)."""
    out = []
    for p in FLAG_BARS:
        pts = np.array(p.split(), float).reshape(-1, 2)
        order = np.argsort(pts[:, 0])
        left, right = pts[order[:2]], pts[order[2:]]
        tl, bl = left[np.argsort(left[:, 1])]
        tr, br = right[np.argsort(right[:, 1])]
        out.append((tl, tr, bl, br))
    return out


def bar_cells(corners: tuple[np.ndarray, ...], nu: int, nv: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """A bar's nu x nv cells as (quadrilateral, centre), bilinear in the bar's (u along, v across) coordinates:
    each cell follows the bar's slant and edges, and together they tile the bar without gaps."""
    tl, tr, bl, br = corners

    def at(u: float, v: float) -> np.ndarray:
        return (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bl + u * br)

    cells = []
    for j in range(nv):
        for i in range(nu):
            u0, u1, v0, v1 = i / nu, (i + 1) / nu, j / nv, (j + 1) / nv
            quad = np.array([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)])
            cells.append((quad, at((u0 + u1) / 2, (v0 + v1) / 2)))
    return cells


OBS_LENGTH, FILL_FROM = 0.6, 0.3                       # observations 0-0.6 s; the fill from 0.3 s (overlapping)


def _times_from_rate(rate: np.ndarray, t: np.ndarray, n: int) -> np.ndarray:
    """n start times spread so that their density follows `rate` on the grid t: the k-th starts where the
    cumulative rate reaches (k + 0.5) / n."""
    cum = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(t))])
    return np.interp((np.arange(n) + 0.5) / n * cum[-1], cum, t)


def pixel_fill_schedule(seed: int = 11) -> tuple[list[tuple[int, np.ndarray]], np.ndarray, np.ndarray]:
    """The cells of all 20 bars (10 x 4, or 8 x 4 in the narrower outer columns) and when each starts to light.

    Observations (a random subset per bar, its share drawn from 3 % to 35 %, at least one cell) start in random
    order at a rate shaped sin^2 over 0-OBS_LENGTH: their own first step. Every other cell is ranked by its distance
    to the nearest observation in its bar, plus up to 0.6 of a cell width of jitter (distances otherwise come in
    whole cell steps, which lit rings of cells at once), and starts at a rate x (1 - x)^0.9 over FILL_FROM to
    FILL_LENGTH: slow at first, accelerating, then tapering. The two rates overlap, so the cells starting per frame
    dip between the steps but never stop (at 30 fps: up to 17, down to 6, up to 13, then to 1).
    Returns [(bar, quad in logo units)], start times (s from FILL_START), and whether each cell is an observation.
    """
    rng = np.random.default_rng(seed)
    cells, centres, bar_of, observed = [], [], [], []
    for b, corners in enumerate(bar_corners()):
        grid = bar_cells(corners, 10 if b // 5 in (1, 2) else 8, 4)
        share = rng.uniform(0.03, 0.35)
        ob = rng.random(len(grid)) < share
        if not ob.any():
            ob[rng.integers(len(grid))] = True
        for (quad, centre), o in zip(grid, ob):
            cells.append((b, quad))
            centres.append(centre)
            bar_of.append(b)
            observed.append(o)
    centres, bar_of, observed = np.array(centres), np.array(bar_of), np.array(observed)
    reach = np.zeros(len(cells))
    for b in range(len(FLAG_BARS)):
        idx = np.nonzero(bar_of == b)[0]
        obs = idx[observed[idx]]
        reach[idx] = np.linalg.norm(centres[idx, None] - centres[None, obs], axis=-1).min(1)
    cell_width = np.median([np.linalg.norm(q[1] - q[0]) for _, q in cells])
    grid = np.linspace(0, FILL_LENGTH, 24001)
    obs_rate = np.where(grid <= OBS_LENGTH, np.sin(np.pi * grid / OBS_LENGTH) ** 2, 0)
    x = np.clip((grid - FILL_FROM) / (FILL_LENGTH - FILL_FROM), 0, 1)
    fill_rate = x * (1 - x) ** 0.9
    start = np.empty(len(cells))
    obs = np.nonzero(observed)[0]
    start[rng.permutation(obs)] = _times_from_rate(obs_rate, grid, len(obs))
    rest = np.nonzero(~observed)[0]
    order = rest[np.argsort(reach[rest] + rng.uniform(0, 0.6 * cell_width, len(rest)), kind="stable")]
    start[order] = _times_from_rate(fill_rate, grid, len(rest))
    return cells, start, observed


def _flag_cells(ax, x0: float, y0: float, scale: float, t: float) -> None:
    """The flag at (x0, y0) px, `scale` px per logo unit, its cells lit by time t (s since FILL_START). A bar with
    no lit cell is drawn grey and a fully lit bar in its colour, exactly as _flag draws them; a bar in progress is
    grey with its lit cells on top, each edged in its own colour (no seams between cells) and clipped to the bar."""
    from matplotlib.patches import Polygon
    from style import mix

    a = ASSETS
    w = np.clip((t - a.flag_start) / CELL_FADE, 0, 1)
    w = w * w * (3 - 2 * w)
    for k, points in enumerate(a.bars):
        colour = FLAG_ON[k // 5]
        mine = a.flag_bar == k
        done = bool((w[mine] >= 1).all())
        bar = Polygon(np.c_[x0 + points[:, 0] * scale, y0 + points[:, 1] * scale], closed=True,
                      facecolor=colour if done else FLAG_OFF, edgecolor="none", zorder=4)
        ax.add_patch(bar)
        if done or not (w[mine] > 0).any():
            continue
        for i in np.nonzero(mine & (w > 0))[0]:
            quad = a.flag_quads[i]
            c = mix(FLAG_OFF, colour, float(w[i]))
            p = Polygon(np.c_[x0 + quad[:, 0] * scale, y0 + quad[:, 1] * scale], closed=True, facecolor=c,
                        edgecolor=c, linewidth=0.8, zorder=5)
            ax.add_patch(p)
            p.set_clip_path(bar)


FLAG_CENTRE = (448.24, 53.43)                           # middle of the flag's bars, logo viewBox units
FLAG_HEIGHT = 78.34                                     # height of the bars (14.26 to 92.6)
WORD_SPLIT = 392.0                                      # "Major TOM" ends at x = 381, "climate" starts at 403.5


def scene_flag(s: float):
    """The flag fills large and centred, cell by cell (pixel_fill_schedule: observations, then the fill spreading
    from them); moves and scales into the logo (eased 0.8 s); "Major TOM" fades in, then "climate"; then the
    dataset names."""
    from climate_figures import INK_2
    from matplotlib.patches import Rectangle
    from style import ease, phase

    a = ASSETS
    fig, ax = _canvas()
    W, H = L["size"]
    x, y, w = L["logo"]
    final = w / 502.24                                                 # px per logo unit in the final logo
    big = 0.45 * H / FLAG_HEIGHT                                       # bars 45 % of the canvas height
    m = ease((s - 2.6) / 0.8)                                          # 0 centred, 1 in the logo (after the fill)
    scale = big * (final / big) ** m                                   # geometric, so the size changes evenly
    fcx = (1 - m) * W / 2 + m * (x + FLAG_CENTRE[0] * final)
    fcy = (1 - m) * H / 2 + m * (y + FLAG_CENTRE[1] * final)
    _flag_cells(ax, fcx - FLAG_CENTRE[0] * scale, fcy - FLAG_CENTRE[1] * scale, scale, s - FILL_START)
    h = 114.72 * final
    for start, x0, x1 in ((3.4, 0, WORD_SPLIT), (4.2, WORD_SPLIT, 502.24)):   # "Major TOM", then "climate"
        level = phase(s, start, start + 0.4)
        if level > 0:
            im = ax.imshow(a.wordmark, extent=(x, x + w, y + h, y), zorder=3, interpolation="lanczos", alpha=level)
            im.set_clip_path(Rectangle((x + x0 * final, y), (x1 - x0) * final, h, transform=ax.transData))
    level = phase(s, 4.4, 4.8)
    if level > 0:
        cx = W / 2
        ax.text(cx, L["logo_sub"], "Major TOM Climate  ·  Major TOM Core-Weather", fontsize=T_SUB + 3,
                fontweight="bold", ha="center", color=_lvl(level, INK_2))
        ax.text(cx, L["logo_sub"] + 48, "Two open datasets for the Major TOM grid", fontsize=T_SUB, ha="center",
                color=_lvl(level, INK_2))
    return fig


def _ortho(lon, lat, lon0: float, lat0: float):
    """Orthographic projection on the unit sphere centred on (lon0, lat0): x east, y north, and whether each point
    is on the visible side."""
    lam, phi = np.radians(lon), np.radians(lat)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    x = np.cos(phi) * np.sin(lam - l0)
    y = np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(lam - l0)
    visible = np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(lam - l0) > 0
    return x, y, visible


def scene_grid(s: float):
    """Plain globe centred on Loch Lomond, zooming in to 21 rows of real Major TOM cells (majortom.cells), which
    fill column by column, top to bottom, in the loader flag's colours; then cell 625U_30L is picked out."""
    from climate_figures import INK, INK_2, PAGE
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Rectangle
    from style import ease, mix, phase

    a = ASSETS
    fig, ax = _canvas()
    vx, vy, vw, vh = L["grid_view"]
    cx, cy = vx + vw / 2, vy + vh / 2
    lon0, lat0 = a.target.centroid
    r0 = L["grid_r"]                                                   # globe radius at the start, px
    r1 = 0.9 * vh / (21 * 10 / 6378.137)                               # 21 rows of 10 km fill 90 % of the height
    z = ease((s - 0.3) / 2.2)
    S = r0 * (r1 / r0) ** z                                            # px per Earth radius, zooming geometrically
    clip = Rectangle((vx, vy), vw, vh, transform=ax.transData)

    def screen(lon, lat):
        x, y, vis = _ortho(np.asarray(lon), np.asarray(lat), lon0, lat0)
        return cx + S * x, cy - S * y, vis

    sea, land = "#e9eef2", "#f3f2ee"
    disk = Circle((cx, cy), S, facecolor=sea, edgecolor="#c3c2b7", linewidth=1, zorder=1)
    ax.add_patch(disk)
    disk.set_clip_path(clip)
    # world coastlines while the globe is small; the 1:10m land and lakes once the view is inside their box
    half_km = 0.5 * vw / S * 6378.137
    detail = float(np.clip((300 - half_km) / 150, 0, 1))
    lines = []
    for seg in a.coast:
        px, py, vis = screen(seg[:, 0], seg[:, 1])
        pts = np.c_[px, py]
        pts[~vis] = np.nan
        lines.append(pts)
    world = LineCollection(lines, colors=[(0, 0, 0, 0.5 * (1 - detail))], linewidths=0.7, zorder=2)
    ax.add_collection(world)
    world.set_clip_path(clip)
    if detail > 0:
        polys = []
        for ring in a.land:
            px, py, _ = screen(ring[:, 0], ring[:, 1])
            polys.append(np.c_[px, py])
        lakes = [np.c_[screen(r[:, 0], r[:, 1])[:2]] for r in a.lakes]
        for shapes, colour in ((polys, land), (lakes, sea)):
            pc = PolyCollection(shapes, facecolors=[mix(sea, colour, detail)], edgecolors=[(0, 0, 0, 0.55 * detail)],
                                linewidths=0.9, zorder=2)
            ax.add_collection(pc)
            pc.set_clip_path(clip)
    # the cells: outlines appear near the end of the zoom, then they fill column by column, top to bottom
    outline = phase(s, 1.8, 2.5)
    if outline > 0:
        cells = a.cells
        quads = [np.c_[screen(q[:, 0], q[:, 1])[:2]] for q in a.cell_quads]
        n = len(cells)
        lit = int(np.clip((s - 2.5) / 1.6, 0, 1) * n)                 # cells on, in fill order
        settle = phase(s, 4.2, 4.7)                                    # then all but the target fade back
        faces = []
        for k in range(n):
            if a.fill_rank[k] < lit:
                colour = FLAG_ON[a.fill_band[k]]
                if k == a.target_k:
                    faces.append(to_rgba(FLAG_ON[3], 0.85))
                else:
                    faces.append(to_rgba(colour, 0.55 * (1 - settle) + 0.15 * settle))
            else:
                faces.append((0, 0, 0, 0))
        pc = PolyCollection(quads, facecolors=faces, edgecolors=[to_rgba(INK, 0.35 * outline)], linewidths=0.8,
                            zorder=3)
        ax.add_collection(pc)
        pc.set_clip_path(clip)
        lake_lines = LineCollection([np.c_[screen(r[:, 0], r[:, 1])[:2]] for r in a.lakes],
                                    colors=[(0, 0, 0, 0.7)], linewidths=1.2, zorder=4)
        ax.add_collection(lake_lines)
        lake_lines.set_clip_path(clip)
        if settle > 0:
            t = np.c_[screen(a.target_quad[:, 0], a.target_quad[:, 1])[:2]]
            ax.plot(np.r_[t[:, 0], t[:1, 0]], np.r_[t[:, 1], t[:1, 1]], color=_lvl(settle, INK), linewidth=2.5,
                    zorder=5)
            x0, y0 = t[:, 0].max(), t[:, 1].min()                      # leader from the cell's top right corner
            lx, ly = x0 + 120, y0 - 90
            ax.plot([x0, lx], [y0, ly], color=_lvl(settle, INK), linewidth=1.5, zorder=5)
            ax.text(lx + 6, ly, f"{a.target.name}\nLoch Lomond", fontsize=T_BODY, fontweight="bold",
                    color=_lvl(settle, INK), va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=_lvl(settle, PAGE), edgecolor="none"))
    # text
    level = phase(s, 0.2, 0.8)
    m = L["margin"]
    tx, ty = L["grid_text"]
    if L["size"][0] < 1500:
        _head(ax, "Major TOM grid", "5,113,745 cells of 10 × 10 km", level)
        lv = phase(s, 2.7, 3.3)
        ax.text(m, ty, "Rows are 10 km apart; each row has its own number of columns,", fontsize=T_SMALL,
                color=_lvl(lv, INK_2))
        ax.text(m, ty + 28, "so columns shift from row to row.", fontsize=T_SMALL, color=_lvl(lv, INK_2))
        lv = phase(s, 4.3, 4.8)
        ax.text(m, ty + 64, "Cell 625U_30L returns in the Core-Weather example.", fontsize=T_SMALL,
                fontweight="bold", color=_lvl(lv, INK))
    else:
        ax.text(tx, ty - 130, "Major TOM grid", fontsize=T_TITLE + 6, fontweight="bold", color=_lvl(level, INK))
        ax.text(tx, ty - 80, "5,113,745 cells of 10 × 10 km", fontsize=T_SUB, color=_lvl(level, INK_2))
        lv = phase(s, 2.7, 3.3)
        for i, line in enumerate(["Rows are 10 km apart; each row", "has its own number of columns,",
                                  "so columns shift from row to row."]):
            ax.text(tx, ty + i * 30, line, fontsize=T_SMALL, color=_lvl(lv, INK_2))
        lv = phase(s, 4.3, 4.8)
        for i, line in enumerate(["Cell 625U_30L returns in", "the Core-Weather example."]):
            ax.text(tx, ty + 130 + i * 30, line, fontsize=T_SMALL, fontweight="bold", color=_lvl(lv, INK))
    return fig


def _card(number: int, name: str, what: str, facts: str, s: float):
    from climate_figures import INK, INK_2
    from style import phase

    fig, ax = _canvas()
    cx, cy = L["card"]
    w = L["size"][0]
    lit = 10 if number == 1 else 20                                    # the flag half lit for 1 of 2, full for 2
    scale = 1.6
    _flag(ax, cx - 448 * scale, cy - 230, scale, lit, level=phase(s, 0.0, 0.4))
    level = phase(s, 0.1, 0.6)
    ax.text(cx, cy, f"Dataset {number} of 2", fontsize=T_SUB, ha="center", color=_lvl(level, INK_2))
    ax.text(cx, cy + 70, name, fontsize=T_TITLE + 10, fontweight="bold", ha="center", color=_lvl(level, INK))
    for i, line in enumerate(textwrap.wrap(what, 44 if w < 1500 else 70)):
        ax.text(cx, cy + 140 + i * 42, line, fontsize=T_SUB + 3, ha="center", color=_lvl(phase(s, 0.2, 0.6), INK))
    ax.text(cx, cy + 260, facts, fontsize=T_BODY, ha="center", color=_lvl(phase(s, 0.4, 0.7), INK_2))
    return fig


def scene_card1(s: float):
    return _card(1, "Major TOM Climate", "Monthly climate records and 1991–2020 normals for every cell",
                 "1940–2025  ·  6 datasets  ·  reanalysis, satellites, rain gauges", s)


def scene_card2(s: float):
    return _card(2, "Major TOM Core-Weather", "Hourly and daily weather around each Major TOM Core image",
                 "2015–2024  ·  3,704,013 acquisitions  ·  ±24 hours, ±30 days", s)


def _temperature_norm():
    import matplotlib.colors as mc

    return mc.TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


# The six normals: key, label, dataset, column, factor to the shown unit (2 m temperature is the first, in normals.npz).
# Labels of at most 21 characters fit the square's 328 px columns; soil water is the top layer (swvl1, 0-7 cm),
# cloud cover the total (tcc).
SIX = [("t2m", "2 m temperature, ERA5", None, None, None),
       ("precip", "Precipitation, IMERG", "IMERG-Monthly", "imerg_precip_mean", 24.0),          # mm/h -> mm/day
       ("pm25", "Surface PM2.5, CAMS", "CAMS-Monthly", "cams_pm2p5_mean", 1e9),                # kg m-3 -> ug m-3
       ("snow", "Snow cover, ERA5-Land", "ERA5Land-Monthly", "era5land_snowc_mean", 1.0),       # %
       ("soil", "Soil water, ERA5-Land", "ERA5Land-Monthly", "era5land_swvl1_mean", 1.0),   # m3 m-3
       ("cloud", "Cloud cover, ERA5", "ERA5-Monthly", "era5_tcc_mean", 100.0)]            # 0-1 -> %


def _six_scales() -> dict:
    """Colour scale per normal: temperature diverging at 0 °C; the others one hue, light to dark (precipitation and
    PM2.5 the README figures' validated ramps, on log scales: both span two orders of magnitude), except cloud
    cover, dark to light, so cloudy regions look white as clouds do."""
    import matplotlib.colors as mc
    from climate_figures import COLD_HOT, DRY_WET, HAZE

    ramp = lambda name, lo, hi: mc.LinearSegmentedColormap.from_list(name, [lo, hi])   # noqa: E731
    return {"t2m": (COLD_HOT, _temperature_norm()),
            "precip": (DRY_WET, mc.LogNorm(0.1, 15)),
            "pm25": (HAZE, mc.LogNorm(1, 100)),
            "snow": (ramp("snow", "#eef2f7", "#2350a0"), mc.Normalize(0, 100)),
            "soil": (ramp("soil", "#edf4f2", "#0f5e58"), mc.Normalize(0.02, 0.5)),
            "cloud": (ramp("cloud", "#36435a", "#eaeef4"), mc.Normalize(0, 100))}   # more cloud lighter; one hue


def six() -> None:
    """The five added normals (blocks of 4 x 4 cells, north up, as normals.npz) and the variable count of the
    release: every series/static column except the keys and the `_valid` share, per dataset."""
    import glob

    import pyarrow.parquet as pq

    out = {}
    for key, _, dataset, column, factor in SIX[1:]:
        a = np.full((12, 501, 1002), np.nan, np.float32)
        for m in range(1, 13):
            t = pq.read_table(RELEASE / dataset / "climatology_overviews" / f"{m:02d}.parquet",
                              columns=["block_row", "block_col", column]).to_pandas()
            a[m - 1, 500 - t.block_row.to_numpy(), t.block_col.to_numpy()] = t[column].to_numpy() * factor
        out[key] = a
        print(key, np.nanpercentile(a, [1, 50, 99]), flush=True)
    np.savez_compressed(DATA / "six.npz", **out)
    keys = {"grid_cell", "grid_row_u", "grid_col_r", "year", "month"}
    counts = {}
    for d in ("CPC-Monthly", "IMERG-Monthly", "CAMS-Monthly", "ERA5Land-Monthly", "ERA5-Monthly"):
        names = pq.read_schema(sorted(glob.glob(str(RELEASE / d / "series" / "year=*" / "*.parquet")))[-1]).names
        counts[d] = sum(1 for c in names if c not in keys and not c.endswith("_valid"))
    counts["ERA5-Static"] = sum(1 for c in pq.read_schema(RELEASE / "ERA5-Static" / "static.parquet").names
                                if c not in keys)
    (DATA / "variables.json").write_text(json.dumps(counts, indent=1))
    print("variables", counts, sum(counts.values()), flush=True)


def scene_normals(s: float):
    """One globe of 2 m temperature normals; after 1.8 s it shrinks into the first of six globes and five more
    normals fade in; all cycle the months together (0.3 s a month), then the release's variable count counts up."""
    from climate_figures import COLD_HOT, INK, INK_2, colour_bar
    from matplotlib.colors import to_rgb
    from style import ease, phase

    a = ASSETS
    fig, ax = _canvas()
    W, H = L["size"]
    wide = W >= 1500
    c = max(0.0, s - 0.7) / 0.3                                        # months since January, 0.3 s a month
    m0 = int(np.floor(c)) % 12
    frac = float(np.clip((c - np.floor(c) - 0.4) / 0.6, 0, 1))        # cross-fade over the last 60 % of a month
    m1 = (m0 + 1) % 12
    empty = np.array(to_rgb("#cfcdc6"))
    lon0 = -10 - 5.0 * s

    def rgb_for(key):
        cmap, norm = a.scales[key]
        data = a.six[key]

        def rgb_of(br, bc, row, col):
            def one(m):
                v = data[m][br, bc]
                out = cmap(norm(np.clip(np.nan_to_num(v, nan=norm.vmin), norm.vmin, norm.vmax)))[..., :3]
                out[~np.isfinite(v)] = empty
                return out
            return one(m0) if frac == 0 else (1 - frac) * one(m0) + frac * one(m1)
        return rgb_of

    # header: the single globe's, then the six's
    shrink = ease((s - 1.8) / 0.7)
    one, many = 1 - phase(s, 1.8, 2.2), phase(s, 2.2, 2.6)
    if one > 0:
        _head(ax, "2 m air temperature, ERA5", "Monthly normal 1991–2020, Major TOM Climate (ERA5-Monthly)", one)
    if many > 0:
        _head(ax, "Monthly normals 1991–2020", "Six of the variables in Major TOM Climate", many)
    # globes: the first moves from the big position into its slot; the other five fade in
    bx, by, br_ = L["normals_globe"]
    slots = L["normals_slots"]
    for k, (key, label, *_rest) in enumerate(SIX):
        sx, sy, sr = slots[k]
        if k == 0:
            centre = (bx + (sx - bx) * shrink, by + (sy - by) * shrink, br_ * (sr / br_) ** shrink)
            level = 1.0
        else:
            level = phase(s, 2.3 + 0.08 * k, 2.8 + 0.08 * k)
            if level <= 0:
                continue
            centre = (sx, sy, sr)
        _globe(ax, centre, rgb_for(key), lon0=lon0, level=level)
        lv = phase(s, 2.3 + 0.08 * k, 2.8 + 0.08 * k)
        if lv > 0:
            ax.text(sx, sy - sr - 16, label, fontsize=T_SMALL, fontweight="bold", ha="center", color=_lvl(lv, INK))
    # month, shared
    month = MONTHS[m1 if frac > 0.5 else m0]
    x, y = L["normals_month"]
    if not wide:                                                       # square: below the big globe, then the grid
        y = y + (L["normals_month_six"] - y) * shrink
    ax.text(x, y, month, fontsize=T_TITLE + 6, fontweight="bold", ha="left" if wide else "center", color=INK)
    bar = 1 - phase(s, 1.6, 2.0)                                      # the temperature scale goes with the big globe
    if bar > 0:
        cbx, cby, cbw = L["normals_bar"]
        ticks = [-40, -20, 0, 20, 40]
        colour_bar(fig, [cbx / W, 1 - (cby + 16) / H, cbw / W, 16 / H], COLD_HOT, _temperature_norm(), ticks,
                   [f"{v:+d}".replace("+0", "0") for v in ticks], "°C", size=T_SMALL)
        if bar < 1:                                                   # fade: cover with the page colour
            from climate_figures import PAGE
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((cbx - 30, cby - 4), cbw + 60, 70, facecolor=PAGE, alpha=1 - bar, zorder=10))
    if wide and one > 0:
        ax.text(x, y + 50, "Normals: the mean of each calendar", fontsize=T_BODY, color=_lvl(one, INK_2))
        ax.text(x, y + 82, "month over 30 years, per cell", fontsize=T_BODY, color=_lvl(one, INK_2))
    # the total, counting up
    lv = phase(s, 3.6, 3.9)
    if lv > 0:
        total = sum(a.variables.values())
        n = int(round(ease((s - 3.6) / 0.6) * total))
        tx, ty = L["normals_total"]
        ax.text(tx, ty, f"{n} variables in {len(a.variables)} datasets", fontsize=T_BODY + 4, fontweight="bold",
                ha="left" if wide else "center", color=_lvl(lv, INK))
    return fig


def _chart_style(p, title: str | None = None) -> None:
    from climate_figures import INK, INK_2, PAGE

    p.set_facecolor(PAGE)
    for side in ("top", "right", "left"):
        p.spines[side].set_visible(False)
    p.spines["bottom"].set_color("#c3c2b7")
    p.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    p.set_axisbelow(True)
    p.tick_params(colors=INK_2, labelsize=T_SMALL, length=0, pad=8)
    if title:
        p.set_title(title, fontsize=T_BODY, color=INK, loc="left", fontweight="bold", pad=14)


def scene_series(s: float):
    import pandas as pd
    from climate_figures import INK, INK_2, PAGE, SERIES

    a = ASSETS
    fig, ax = _canvas()
    t = a.series
    first, last = int(t.year.min()), int(t.year.max())
    _head(ax, "2 m air temperature, ERA5", f"{PLACE} (cell {CELL}), every month {first}–{last}")
    v = t.era5_t2m.to_numpy()
    mean12 = pd.Series(v).rolling(12, center=True).mean().to_numpy()
    n = len(v)
    shown = float(np.clip((s - 0.4) / 3.2, 0, 1)) * (n - 1)           # months drawn so far
    k = int(shown)
    p = _axes(fig, L["series_chart"])
    _chart_style(p, "°C")
    xs = t.year.to_numpy() + (t.month.to_numpy() - 0.5) / 12
    monthly, annual = mix_page(SERIES[0], 0.45), SERIES[0]
    p.plot(xs[:k + 1], v[:k + 1], color=monthly, linewidth=1.2)
    ok = np.isfinite(mean12[:k + 1])
    p.plot(xs[:k + 1][ok], mean12[:k + 1][ok], color=annual, linewidth=3)
    p.plot(xs[k], v[k], "o", color=annual, markersize=9, markeredgecolor=PAGE, markeredgewidth=2, zorder=5)
    p.set_xlim(first - 1, last + 9 if L["size"][0] < 1500 else last + 6)
    lo, hi = np.nanmin(v), np.nanmax(v)
    p.set_ylim(np.floor(lo) - 1, np.ceil(hi) + 1)
    p.set_xticks(range(1940, last + 1, 20))
    if k >= n - 1:                                                     # direct labels once complete
        p.text(last + 1.2, mean12[np.isfinite(mean12)][-1], "12-month\nmean", fontsize=T_SMALL, color=INK,
               va="center")
        p.text(last + 1.2, np.nanmax(v[-36:]), "monthly", fontsize=T_SMALL, color=INK_2, va="center")
    cx, cy, cw, _ = L["series_chart"]
    ax.text(cx + cw, cy - 22, str(int(t.year[k])), fontsize=T_TITLE, fontweight="bold", ha="right", color=INK_2,
            va="baseline")
    ax.text(L["margin"], L["series_note"], f"{n:,} months of one cell, read from one by_cell/ Parquet file",
            fontsize=T_SMALL, color=INK_2)
    return fig


def mix_page(colour: str, f: float) -> str:
    """`colour` lightened towards the page by fraction f."""
    from climate_figures import PAGE
    from style import mix

    return mix(colour, PAGE, f)


def scene_weather(s: float):
    from climate_figures import INK, INK_2, PAGE
    from core_weather_figures import COLOUR
    from style import phase

    a = ASSETS
    fig, ax = _canvas()
    _head(ax, f"{PLACE}, {a.sample['acquired'][:10]}", "One Core image and the weather around it")
    x, y, size = L["weather_image"]
    _image(ax, a.thumb, x, y, size, alpha=phase(s, 0.0, 0.6))
    ix, iy, step = L["weather_info"]
    wide = L["size"][0] >= 1500
    info = [("Image", "Sentinel-2 L2A (Core-S2L2A)"), ("Acquired", f"{a.sample['acquired']} UTC"),
            ("Cell", f"{a.sample['cell']}, 10 × 10 km"), ("Cloud cover", f"{a.sample['cloud_cover']:.1f} % (Sentinel-2)")]
    for i, (k, v) in enumerate(info):
        lv = phase(s, 0.4 + 0.15 * i, 0.9 + 0.15 * i)
        if wide:
            cx, cy = ix + (i % 2) * 380, iy + (i // 2) * 2 * step
        else:
            cx, cy = ix, iy + i * step * 1.25
        ax.text(cx, cy, k, fontsize=T_SMALL, color=_lvl(lv, INK_2), va="top")
        ax.text(cx, cy + 28, v, fontsize=T_BODY if not wide else T_SMALL, color=_lvl(lv, INK), va="top")
    h = a.hourly
    until = -24 + float(np.clip((s - 1.4) / 4.0, 0, 1)) * 48          # hours drawn so far
    shown = h[h.index <= until]
    t_ax = _axes(fig, L["weather_t"])
    _chart_style(t_ax, "Hourly 2 m temperature, °C")
    for col, key in (("era5_t2m", "ERA5"), ("era5land_t2m", "ERA5-Land")):
        t_ax.plot(shown.index, shown[col] - 273.15, color=COLOUR[key], linewidth=2.4)
    temps = np.r_[h.era5_t2m, h.era5land_t2m] - 273.15
    t_ax.set_ylim(np.floor(np.nanmin(temps)) - 1, np.ceil(np.nanmax(temps)) + 1)
    c_ax = _axes(fig, L["weather_c"])
    _chart_style(c_ax, "Hourly total cloud cover, ERA5, %")
    c_ax.fill_between(shown.index, shown.era5_tcc * 100, color=mix_page(COLOUR["ERA5"], 0.55), linewidth=0, step="mid")
    c_ax.step(shown.index, shown.era5_tcc * 100, color=COLOUR["ERA5"], linewidth=2, where="mid")
    c_ax.set_ylim(0, 105)
    c_ax.set_yticks([0, 50, 100])
    for p in (t_ax, c_ax):
        p.set_xlim(-24.5, 24.5)
        p.set_xticks([-24, -12, 0, 12, 24], ["−24 h", "−12 h", "image", "+12 h", "+24 h"])
        p.axvline(0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    if until >= 24:                                                    # direct labels once the lines are complete
        end_e, end_l = h.era5_t2m.iloc[-1] - 273.15, h.era5land_t2m.iloc[-1] - 273.15
        gap = 0.2 * np.ptp(t_ax.get_ylim())
        hi_, lo_ = (end_e, end_l) if end_e >= end_l else (end_l, end_e)
        ys = {"ERA5": end_e, "ERA5-Land": end_l}
        if abs(end_e - end_l) < gap:
            ys = {("ERA5" if end_e >= end_l else "ERA5-Land"): hi_ + gap / 2,
                  ("ERA5-Land" if end_e >= end_l else "ERA5"): lo_ - gap / 2}
        for key, yv in ys.items():
            t_ax.text(25.3, yv, key, fontsize=T_SMALL, color=INK_2, va="center")
    lv = phase(s, 4.7, 5.2)
    if lv > 0:
        ax.text(L["margin"], L["weather_note"], "No precipitation in these 48 hours (ERA5, IMERG).",
                fontsize=T_SMALL, color=_lvl(lv, INK_2))
    return fig


# Emojis (Noto Color Emoji, drawn to PNG by emojis()): one per kind of data, and one per main quantity
EMOJI = {"reanalysis": "\U0001F9EE", "satellite": "\U0001F6F0", "gauges": "\U0001F327", "temperature": "\U0001F321",
         "precipitation": "\U0001F327", "wind": "\U0001F4A8", "snow": "❄", "air": "\U0001F3ED", "parquet": "\U0001F4E6", "https": "\U0001F310",
         "python": "\U0001F40D",
         "soil water": "\U0001F4A7"}

# group: (emoji, what the kind of data is, tiles); tile: name, producer, grid and record, quantities, the datasets
# that use it (with the variable count where it is large)
SOURCES = {
    "Reanalysis": ("reanalysis", "models fitted to observations", [
        ("ERA5", "ECMWF, Copernicus", "0.25°  ·  1940–2025",
         [("temperature", "temperature"), ("precipitation", "precipitation"), ("wind", "wind"), ("snow", "snow"),
          ("soil water", "soil water")], "201 variables · Climate, Core-Weather"),
        ("ERA5-Land", "ECMWF, Copernicus", "0.1°  ·  1950–2025",
         [("temperature", "temperature"), ("soil water", "soil water"), ("snow", "snow")],
         "49 variables · Climate, Core-Weather"),
        ("CAMS EAC4", "ECMWF, Copernicus", "0.75°  ·  2003–2025",
         [("air", "aerosols, PM2.5, trace gases")], "Climate, Core-Weather")]),
    "Satellite": ("satellite", "observations from orbit", [
        ("IMERG V07", "NASA GPM", "0.1°  ·  1998–2024", [("precipitation", "precipitation")],
         "Climate, Core-Weather"),
        ("Sentinel-5P", "ESA, Copernicus", "5.5 × 3.5 km  ·  from 2018",
         [("air", "trace gases, once a day")], "Core-Weather")]),
    "Rain gauges": ("gauges", "ground measurements", [
        ("CPC Global Unified", "NOAA", "0.5°  ·  1979–2025",
         [("precipitation", "precipitation"), ("temperature", "temperature")], "Climate, Core-Weather")]),
}
CHAR_PX = 1.39 * 0.6                                    # IBM Plex Mono advance, px per point of size


def _emoji(ax, key: str, x: float, baseline: float, size: float, level: float = 1.0) -> float:
    """An emoji at (x, baseline) px, scaled to text of `size` pt; returns its width plus a gap, px."""
    img = ASSETS.emoji[key]
    h = size * 1.39 * 1.25
    w = h * img.shape[1] / img.shape[0]
    rgba = img.astype(np.float32) / 255
    rgba[..., 3] *= level
    ax.imshow(rgba, extent=(x, x + w, baseline + 0.2 * h, baseline - 0.8 * h), interpolation="lanczos", zorder=4)
    return w + 0.35 * size * 1.39


def _quantities(ax, items, x: float, y: float, width: float, size: float, colour: str, level: float,
                step: float = 28) -> None:
    """(emoji key, label) pairs laid out left to right from (x, y baseline), wrapping within `width` px."""
    cx, cy = x, y
    for key, label in items:
        need = size * 1.39 * 1.25 * 1.4 + len(label) * size * CHAR_PX
        if cx > x and cx + need > x + width:
            cx, cy = x, cy + step
        cx += _emoji(ax, key, cx, cy, size, level)
        ax.text(cx, cy, label, fontsize=size, color=colour, va="baseline")
        cx += len(label) * size * CHAR_PX + size * 1.39


def scene_sources(s: float):
    from climate_figures import INK, INK_2, INK_MUTED
    from matplotlib.patches import FancyBboxPatch
    from style import ease

    fig, ax = _canvas()
    _head(ax, "Sources", "Each main quantity from more than one kind of data")
    bx, by, bw, bh = L["sources_box"]
    columns = L["sources_cols"]
    gap = 24
    cw = (bw - gap * (len(columns) - 1)) / len(columns)
    th = L["tile_h"]
    k = 0
    for c, groups in enumerate(columns):
        x = bx + c * (cw + gap)
        y = by
        for group in groups:
            icon, about, tiles = SOURCES[group]
            lv = ease((s - 0.3 - 0.3 * k) / 0.5)
            gx = x + _emoji(ax, icon, x, y + 24, T_BODY + 2, lv)
            ax.text(gx, y + 24, group, fontsize=T_BODY + 2, fontweight="bold", color=_lvl(lv, INK), va="baseline")
            ax.text(x, y + 54, about, fontsize=T_SMALL, color=_lvl(lv, INK_2), va="baseline")
            y += 72
            for name, producer, grid, items, used in tiles:
                u = ease((s - 0.3 - 0.3 * k) / 0.5)
                k += 1
                dy = 18 * (1 - u)
                ax.add_patch(FancyBboxPatch((x, y + dy), cw, th - 14, boxstyle="round,pad=0,rounding_size=10",
                                            facecolor=_lvl(u, "#f1f0ec"), edgecolor=_lvl(u, "#dddbd5"), zorder=1))
                ty = y + dy + 34
                ax.text(x + 18, ty, name, fontsize=T_BODY + 2, fontweight="bold", color=_lvl(u, INK), va="baseline")
                ax.text(x + cw - 18, ty, producer, fontsize=T_SMALL, color=_lvl(u, INK_2), ha="right", va="baseline")
                ax.text(x + 18, ty + 34, grid, fontsize=T_SMALL, color=_lvl(u, INK), va="baseline")
                _quantities(ax, items, x + 18, ty + 72, cw - 36, T_SMALL, _lvl(u, INK_2), u)
                ax.text(x + 18, ty + 136, used, fontsize=T_SMALL - 2, color=_lvl(u, INK_MUTED),
                        va="baseline")
                y += th
            y += 10
    lv = ease((s - 0.3 - 0.3 * k) / 0.5)
    note = "Precipitation from gauges (CPC), satellites (IMERG) and reanalysis (ERA5, ERA5-Land): compare or choose " \
           "per region."
    lines = textwrap.wrap(note, 140 if L["size"][0] >= 1500 else 70)
    for j, line in enumerate(lines):
        ax.text(L["margin"], L["sources_note"] - 28 * (len(lines) - 1 - j), line, fontsize=T_SMALL,
                color=_lvl(lv, INK_2))
    return fig


CODE = [
    "import fsspec",
    "import pandas as pd",
    "",
    'url = "https://data.source.coop/major-tom/climate/CPC-Monthly"',
    "history = pd.read_parquet(",
    f'    f"{{url}}/by_cell/band={BAND}.parquet", filesystem=fsspec.filesystem("https"),',
    '    columns=["year", "month", "cpc_precip"],',
    f'    filters=[("grid_row_u", "=", {ROW}), ("grid_col_r", "=", {COL})])',
]


def scene_access(s: float):
    from climate_figures import INK, INK_2
    from core_weather_figures import COLOUR
    from matplotlib.patches import FancyBboxPatch

    a = ASSETS
    acc = a.access
    fig, ax = _canvas()
    _head(ax, "Reading one cell", "47 years of one cell, read live from source.coop")
    x, y, w, h = L["code"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", facecolor="#f1f0ec",
                                edgecolor="#dddbd5", zorder=1))
    total = sum(len(line) for line in CODE)
    typed = int(np.clip((s - 0.2) / 2.2, 0, 1) * total)                # characters typed so far
    code_size = 13 if L["size"][0] < 1500 else 13.5
    for i, line in enumerate(CODE):
        shown = line[:max(0, typed)]
        typed -= len(line)
        ax.text(x + 22, y + 40 + i * 33, shown, fontsize=code_size, color=INK, va="baseline", family="IBM Plex Mono")
    # the read: the chart fills and the counters run for the measured time (median of three reads)
    start = 2.6
    u = float(np.clip((s - start) / acc["seconds"], 0, 1))
    p = _axes(fig, L["access_chart"])
    hist = a.history
    _chart_style(p, "Monthly precipitation, CPC rain gauges, mm")
    xs = hist.year.to_numpy() + (hist.month.to_numpy() - 0.5) / 12
    v = hist.cpc_precip.to_numpy()
    k = int(u * (len(v) - 1))
    if s >= start:
        p.plot(xs[:k + 1], v[:k + 1], color=COLOUR["CPC gauges"], linewidth=1.2)
    p.set_xlim(xs[0] - 1, xs[-1] + 1)
    p.set_ylim(0, np.nanmax(v) * 1.08)
    p.set_xticks(range(1980, 2030, 10))
    cx, cy = L["access_counts"]
    if s >= start:
        counts = [(f"{u * acc['seconds']:.2f} s", "median of 3 runs"), (f"{round(u * acc['requests'])}", "HTTP requests"),
                  (f"{u * acc['bytes'] / 1e6:.1f} MB", f"of a {acc['file_bytes'] / 1e6:,.0f} MB file"),
                  (f"{round(u * acc['rows'])}", "months")]
        wide = L["size"][0] >= 1500                                    # wide: 2 x 2 under the code; square: a row
        cw = 450 if wide else (L["size"][0] - 2 * L["margin"]) / 4
        for i, (value, label) in enumerate(counts):
            px, py = (cx + (i % 2) * cw, cy + (i // 2) * 130) if wide else (cx + i * cw, cy)
            ax.text(px, py, value, fontsize=T_TITLE + 2, fontweight="bold", color=INK, va="baseline")
            ax.text(px, py + 38, label, fontsize=T_SMALL, color=INK_2, va="baseline")
    return fig


def scene_size(s: float):
    """The two datasets' sizes, counting up (the Major TOM Climate icon beside its number; Core-Weather has no
    mark of its own), then three lines on how they are read, each with an emoji."""
    from climate_figures import INK, INK_2
    from style import ease, phase

    acc = ASSETS.access
    fig, ax = _canvas()
    _head(ax, "Size and access", "Both datasets on source.coop", phase(s, 0.0, 0.4))
    climate_tb = sum(acc["climate_gb"].values()) / 1000
    blocks = [("Major TOM Climate", f"{{:.2f}} TB", climate_tb, "6 datasets, 1940–2025"),
              ("Major TOM Core-Weather", f"{{:.1f}} GB", acc["weather_gb"], "3,704,013 acquisitions, 2015–2024")]
    for i, (name, fmt, value, what) in enumerate(blocks):
        bx, by = L["size_blocks"][i]
        lv = phase(s, 0.3 + 0.5 * i, 0.8 + 0.5 * i)
        u = ease((s - 0.3 - 0.5 * i) / 1.4)                           # count up over 1.4 s
        ax.text(bx, by, name, fontsize=T_BODY + 2, fontweight="bold", color=_lvl(lv, INK), va="baseline")
        ax.text(bx, by + 96, fmt.format(u * value), fontsize=T_TITLE + 34, fontweight="bold", color=_lvl(lv, INK),
                va="baseline")
        ax.text(bx, by + 142, what, fontsize=T_BODY, color=_lvl(lv, INK_2), va="baseline")
        if i == 0:
            ix, iy, ih = L["size_icon"]
            _image(ax, ASSETS.climate_icon, ix, iy, ih, alpha=lv)
    fx, fy = L["size_formats"]
    for j, (key, line) in enumerate([("parquet", "Parquet"), ("https", "HTTPS range reads, no account needed"),
                                     ("python", "pandas + fsspec in Python, hyparquet in the browser")]):
        lv = phase(s, 1.8 + 0.25 * j, 2.3 + 0.25 * j)
        x = fx + _emoji(ax, key, fx, fy + j * 58, T_SUB + 1, lv)
        ax.text(x, fy + j * 58, line, fontsize=T_SUB + 1, color=_lvl(lv, INK), va="baseline")
    return fig


def scene_end(s: float):
    from climate_figures import INK, INK_2
    from style import phase

    a = ASSETS
    fig, ax = _canvas()
    x, y, w = L["end_logo"]
    _logo(ax, x, y, w)
    cx, ly, step = L["end_lines"]
    lines = [("Major TOM Climate", "source.coop/major-tom/climate"),
             ("Major TOM Core-Weather", "source.coop/major-tom/core-weather")]
    for i, (name, url) in enumerate(lines):
        level = phase(s, 0.2 + 0.3 * i, 0.7 + 0.3 * i)
        ax.text(cx, ly + i * step, name, fontsize=T_TITLE - 4, fontweight="bold", ha="center", color=_lvl(level, INK))
        ax.text(cx, ly + i * step + 42, url, fontsize=T_BODY + 1, ha="center", color=_lvl(level, INK_2))
    level = phase(s, 0.4, 0.8)
    if level > 0:
        m = L["margin"]
        ax.plot([m, L["size"][0] - m], [L["end_rule"], L["end_rule"]], color=_lvl(level, "#e1e0d9"), linewidth=1)
        ex, ey = L["end_logos"]
        _image(ax, a.earth_compress, ex, ey, 76, alpha=level)
        mark_h = 60
        mx = ex + 780 - 330 if L["size"][0] < 1500 else ex + 800 - 330
        _image(ax, a.mark, mx, ey + 8, mark_h, alpha=level)
        ax.text(mx + mark_h * a.mark.shape[1] / a.mark.shape[0] + 12, ey + 38, "asterisk labs",
                fontsize=mark_h * 0.7 * 0.72, fontfamily="League Spartan", fontweight=500, color=_lvl(level, INK),
                va="center")
        credit = ["Developed by Asterisk Labs as part of Earth Compress, an ARIA-funded",
                  "project jointly led with the National Oceanography Centre."]
        ccx, ccy = L["end_credit"]
        for i, line in enumerate(credit):
            ax.text(ccx, ccy + i * 34, line, fontsize=T_SMALL, ha="center", color=_lvl(level, INK_2))
        foot = ("Contains modified Copernicus Climate Change Service information 2026 and modified Copernicus "
                "Sentinel data 2022")
        for i, line in enumerate(textwrap.wrap(foot, 80 if L["size"][0] < 1500 else 140)):
            ax.text(L["size"][0] / 2, L["end_foot"] + i * 24, line, fontsize=T_FOOT, ha="center",
                    color=_lvl(level, INK_2))
    return fig


DRAW = {"flag": scene_flag, "grid": scene_grid, "card1": scene_card1, "normals": scene_normals,
        "series": scene_series, "card2": scene_card2, "weather": scene_weather, "sources": scene_sources,
        "access": scene_access, "size": scene_size, "end": scene_end}


def frame(job: tuple[int, str]) -> Path:
    """Draw frame i (time i / FPS) of a layout, with the fade through the page colour at scene edges, and save it."""
    import matplotlib.pyplot as plt
    from climate_figures import PAGE

    global ASSETS
    i, layout = job
    if ASSETS is None:
        _setup()
        ASSETS = Assets()
    L.clear()
    L.update(LAYOUTS[layout])
    t = i / FPS
    name, (start, end) = next((k, v) for k, v in SCENES.items() if v[0] <= t < v[1] or k == "end")
    s = t - start
    fig = DRAW[name](s)
    veil = 1 - min(1.0, s / FADE, (end - t) / FADE)
    if name == "flag":
        veil = 1 - min(1.0, (end - t) / FADE)                         # the first frame starts on the flag
    if name == "end":
        veil = 1 - min(1.0, s / FADE)                                 # the last frame holds, no fade out
    if veil > 0:
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=PAGE, alpha=veil,
                                         zorder=100))
    out = WORK / f"frames-{layout}" / f"{i:05d}.png"
    fig.savefig(out, dpi=100, facecolor=PAGE)
    plt.close(fig)
    return out


KEY_FRAMES = {"flag-fill": 1.2, "flag-centre": 2.55, "flag": 7.1, "grid-zoom": 9.1, "grid": 14.8, "card1": 17.6,
              "normals": 20.6, "normals-six": 26.2, "series": 30.7, "card2": 34.0, "weather": 42.7, "sources": 49.7, "access": 56.8,
              "size": 62.3, "end": 65.8}


def render(layout: str | None = None, only: str | None = None) -> None:
    """All frames of one layout (or both) in parallel, then the MP4 and the key frames as PNG."""
    layouts = [layout] if layout else list(LAYOUTS)
    n = int(SCENES["end"][1] * FPS)
    for name in layouts:
        frames = WORK / f"frames-{name}"
        if only is None and frames.exists():
            shutil.move(str(frames), f"/RUBBISH/intro-v2-frames-{name}-{os.getpid()}")
        frames.mkdir(parents=True, exist_ok=True)
        todo = [(i, name) for i in (range(n) if only is None else [int(only)])]
        with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
            list(pool.map(frame, todo, chunksize=4))
        if only is not None:
            continue
        out = WORK / f"major-tom-climate-weather-v2-{name}.mp4"
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                        str(frames / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset",
                        "slow", "-movflags", "+faststart", str(out)], check=True)
        SHOTS.mkdir(exist_ok=True)
        shutil.copy(out, SHOTS / out.name)
        for scene, t in KEY_FRAMES.items():
            shutil.copy(frames / f"{int(t * FPS):05d}.png", SHOTS / f"intro2-{name}-{scene}.png")
        print(out, f"{out.stat().st_size / 1e6:.1f} MB, {n} frames", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    {"logos": logos, "shapes": shapes, "emojis": emojis, "six": six, "prep": prep, "render": lambda: render(*sys.argv[2:4])}[what]()
