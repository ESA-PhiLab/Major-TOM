"""Figures for the Core-Weather README: the weather around one acquisition at each of the three places, and a
global map of 2 m temperature at acquisition time.

    places: Core-S2L2A metadata (one sample per cell) -> that acquisition's Core-Weather rows (merged/, local)
      + the thumbnails of the cell's Core-S2L2A, Core-S1RTC and Core-DEM samples (one row group each, Hugging Face)
      -> one figure per place: thumbnails above; hourly (±24 h) and daily (±30 d) temperature and precipitation
      -> <out>/figures/place-<name>.png
    map: hourly/year=*/*.parquet, rows at step_h = 0 (the hour nearest each acquisition), era5_t2m
      -> mean per cell over its acquisitions (all three datasets, 2016-2024: seasons and times of day mixed)
      -> mean over blocks of 12 x 12 cells (about 120 km; 12 divides the 2,004 x 4,008 array), so single
         acquisitions do not show as noise
      -> the sinusoidal map of climate_figures.py, its temperature scale -> <out>/figures/temperature-at-acquisition.png
    Runs in the miko-torch environment (Basemap), as climate_figures.py.

    python docs/evidence/core_weather_figures.py /data/Major-TOM /data/Major-TOM/Core-Weather/merged [places|map]
"""
from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from climate_figures import (COLD_HOT, INK, INK_2, PAGE, PLACES, ROWS, COLS, SERIES, WIDTH_PX, SCALE,  # noqa: E402
                             grid_image, map_figure, use_major_tom_font)

# one colour per source in every chart: the three validated categorical slots, ERA5-Land in the secondary ink
COLOUR = {"ERA5": SERIES[0], "IMERG": SERIES[1], "CPC gauges": SERIES[2], "ERA5-Land": INK_2}
DATASETS = [("Core-S2L2A", "Sentinel-2 L2A"), ("Core-S1RTC", "Sentinel-1 RTC"), ("Core-DEM", "Copernicus DEM")]


def cell_name(row: int, col: int) -> str:
    return f"{abs(row)}{'U' if row >= 0 else 'D'}_{abs(col)}{'R' if col >= 0 else 'L'}"


def thumbnail(url: str, row_group: int) -> Image.Image:
    """The stored thumbnail of one Core sample: its image file's row group, thumbnail column only."""
    with fsspec.open(url, "rb", block_size=2**20) as f:
        t = pq.ParquetFile(f).read_row_group(row_group, columns=["thumbnail"])
    return Image.open(io.BytesIO(t.column("thumbnail")[0].as_py()))


def stretch(img: Image.Image) -> Image.Image:
    """Linear contrast stretch of a thumbnail, the same for all three channels (keeps the colour balance): the
    0.5th-99.5th percentile of all values to 0-255 (stored thumbnails are flat, e.g. 84-118 of 255 at 625U_30L)."""
    a = np.asarray(img.convert("RGB"), np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    return Image.fromarray(np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))


def weather_rows(root: Path, table: str, cell: str, acquired: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    path = root / table / f"year={acquired:%Y}" / f"{acquired:%Y-%m-%d}.parquet"
    step = "step_h" if table == "hourly" else "step_d"
    t = pd.read_parquet(path, columns=[step, *columns], filters=[("grid_cell", "=", cell), ("acquired", "=", acquired)])
    return t.sort_values(step).set_index(step)


def _style(ax, title: str, unit: str, xlabel: str) -> None:
    ax.set_facecolor(PAGE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.axvline(0, color=INK_2, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_title(f"{title}, {unit}", fontsize=9, color=INK, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8, color=INK_2)


def place_figure(core: Path, root: Path, name: str, row: int, col: int, out: Path) -> None:
    cell = cell_name(row, col)
    s2 = pd.read_parquet(core / "Core-S2L2A" / "metadata.parquet", filters=[("grid_cell", "=", cell)]).iloc[0]
    acquired = pd.Timestamp(s2.timestamp)
    hourly = weather_rows(root, "hourly", cell, acquired, ["era5_t2m", "era5land_t2m", "era5_tp", "imerg_precip"])
    daily = weather_rows(root, "daily", cell, acquired, ["era5_t2m_mean", "era5_t2m_min", "era5_t2m_max",
                                                         "era5_tp_sum", "imerg_precip_sum", "cpc_precip"])
    fig = plt.figure(figsize=(WIDTH_PX / 100, 7.6), dpi=100, facecolor=PAGE)
    fig.text(0.02, 0.975, f"{name}, Sentinel-2 acquisition {acquired:%Y-%m-%d %H:%M} UTC",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.94, f"Core-Weather, cell {cell} · cell means; dashed line: the acquisition (step 0)",
             fontsize=9, color=INK_2, va="top")
    for i, (dataset, label) in enumerate(DATASETS):
        ax = fig.add_axes([0.02 + i * 0.33, 0.56, 0.3, 0.31])
        ax.set_axis_off()
        m = pd.read_parquet(core / dataset / "metadata.parquet", filters=[("grid_cell", "=", cell)])
        if m.empty:
            ax.text(0.5, 0.5, f"no {label} sample\nin this cell", ha="center", va="center", fontsize=9, color=INK_2)
        else:
            s = m.iloc[0]
            ax.imshow(thumbnail(s.parquet_url, int(s.parquet_row)))
            when = f", {pd.Timestamp(s.timestamp):%Y-%m-%d}" if "timestamp" in m else ""
            ax.set_title(f"{label}{when}", fontsize=9, color=INK, loc="left")
    axes = [fig.add_axes([0.07 + j * 0.49, 0.32 - i * 0.26, 0.42, 0.18]) for i in range(2) for j in range(2)]
    a = axes[0]
    a.plot(hourly.index, hourly.era5_t2m - 273.15, color=COLOUR["ERA5"], linewidth=1.8, label="ERA5")
    a.plot(hourly.index, hourly.era5land_t2m - 273.15, color=COLOUR["ERA5-Land"], linewidth=1.8, label="ERA5-Land")
    _style(a, "Hourly 2 m temperature", "°C", "hours from the acquisition")
    a.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    a = axes[1]
    a.bar(hourly.index - 0.2, hourly.era5_tp * 1000, width=0.4, color=COLOUR["ERA5"], label="ERA5")
    a.bar(hourly.index + 0.2, hourly.imerg_precip, width=0.4, color=COLOUR["IMERG"], label="IMERG")
    _style(a, "Hourly precipitation", "mm", "hours from the acquisition")
    a.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    a = axes[2]
    a.fill_between(daily.index, daily.era5_t2m_min - 273.15, daily.era5_t2m_max - 273.15, color=SERIES[0], alpha=0.18,
                   linewidth=0, label="min to max")
    a.plot(daily.index, daily.era5_t2m_mean - 273.15, color=SERIES[0], linewidth=1.8, label="mean")
    _style(a, "Daily 2 m temperature (ERA5)", "°C", "days from the acquisition")
    a.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    a = axes[3]
    for k, (column, label, scale) in enumerate([("era5_tp_sum", "ERA5", 1000), ("imerg_precip_sum", "IMERG", 1),
                                                ("cpc_precip", "CPC gauges", 1)]):
        a.bar(daily.index + (k - 1) * 0.27, daily[column] * scale, width=0.27, color=COLOUR[label], label=label)
    _style(a, "Daily precipitation", "mm", "days from the acquisition")
    a.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100 * SCALE, facecolor=PAGE)
    plt.close(fig)


def places(core: Path, root: Path) -> None:
    for name, (row, col) in PLACES.items():
        slug = name.split(" (")[0].lower().replace(" ", "-")
        place_figure(core, root, name, row, col, root / "figures" / f"place-{slug}.png")
        print(name, "done", flush=True)


def _step0(path: Path) -> pd.DataFrame:
    """One day file's per-cell sum and count of era5_t2m at step 0."""
    t = pd.read_parquet(path, columns=["grid_cell", "era5_t2m"], filters=[("step_h", "=", 0)])
    return t.groupby("grid_cell").era5_t2m.agg(["sum", "count"])


def _coarse(image: np.ndarray, k: int = 12) -> np.ndarray:
    """Mean over k x k blocks of cells, put back on every cell of the block that has a value."""
    ok = np.isfinite(image)
    h, w = ROWS // k, COLS // k
    total = np.where(ok, image, 0).reshape(h, k, w, k).sum((1, 3))
    n = ok.reshape(h, k, w, k).sum((1, 3))
    mean = np.repeat(np.repeat(total / np.maximum(n, 1), k, 0), k, 1)
    return np.where(ok, mean, np.nan).astype(np.float32)


def global_map(root: Path) -> None:
    files = sorted((root / "hourly").glob("year=*/*.parquet"))
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        parts = list(pool.map(_step0, files, chunksize=8))
    t = pd.concat(parts).groupby(level=0).sum()
    names = t.index.to_series().str.extract(r"(\d+)([UD])_(\d+)([LR])")
    rows = names[0].astype(int).where(names[1] == "U", -names[0].astype(int)).to_numpy()
    cols = names[2].astype(int).where(names[3] == "R", -names[2].astype(int)).to_numpy()
    values = (t["sum"] / t["count"]).to_numpy() - 273.15
    print(f"{len(files):,} files, {t['count'].sum():,} acquisitions, {len(t):,} cells", flush=True)
    ticks = [-40, -20, -10, 0, 10, 20, 30, 40]
    map_figure([(_coarse(grid_image(rows, cols, values)), None)], COLD_HOT, TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40),
               ticks, [f"{v:+d}".replace("+0", "0") for v in ticks],
               "2 m temperature at acquisition time, ERA5, Core-Weather",
               "Hour nearest each acquisition (Core-S2L2A, S2L1C, S1RTC), 2016–2024 · mean per 12 × 12 cells",
               "°C (grey: 0 °C, freezing)",
               root / "figures" / "temperature-at-acquisition.png", empty_label="no acquisition")


if __name__ == "__main__":
    use_major_tom_font()
    core, root = Path(sys.argv[1]), Path(sys.argv[2])
    for what in sys.argv[3:] or ["places", "map"]:
        {"places": places, "map": lambda c, r: global_map(r)}[what](core, root)
