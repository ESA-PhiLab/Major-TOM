"""Figures for major-tom/climate: each dataset's normals on the Major TOM sinusoidal map (the Core coverage style).

    climatology/band=NN.parquet (grid_row_u, grid_col_r, month, <variable>_mean)
      -> one value per cell (e.g. annual total = sum of the 12 monthly normals)
      -> the 2004 x 4008 (row, column) array: Major TOM columns are equal-width in km, so this array is a
         sinusoidal map with one cell per pixel (as majortom/extras/coverage_vis.py draws coverage)
      -> block means down to the README width (900 px; source.coop shows images at full size), continuous colours:
         cells without data dark (#242424, as the coverage maps), outside the Earth the page colour
      -> Basemap's sinusoidal coastlines on top -> <release>/<dataset>/figures/<name>.png
    Colour scales checked with the dataviz palette validator against the dark map surface (lightness order, contrast).
    Text in IBM Plex Mono (the Major TOM wordmark). Runs in the miko-torch environment (Basemap).

    python docs/evidence/climate_figures.py /data/Major-TOM/climate-release cpc-precip
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
import pyarrow.parquet as pq                                             # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm, TwoSlopeNorm, to_rgba  # noqa: E402

ROWS, COLS = 2004, 4008                  # grid rows -1002..1001, columns -2004..2003 at 10 km
R_KM, D_KM = 6378.137, 10.0
WIDTH_PX = 900                           # shown width in the README (width="100%")
SCALE = 2                                # saved at 2x (1,800 px) so maps and text stay sharp on dense screens
PAGE, MAP_EMPTY = "#fcfcfb", "#242424"   # chart surface; cells without data, as in the Core coverage maps
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
COAST = (0, 0, 0, 0.55)
# precipitation, dry -> wet: sand, green, blue; lightness falls steadily (reads in greyscale and for colour-blind
# readers) and the wet end clears the dark map surface 3.2:1 (dataviz validator, --ordinal; multi-hue by choice)
DRY_WET = LinearSegmentedColormap.from_list("dry_wet", ["#f2e3b6", "#cfdc8f", "#8fc77d", "#47ad93", "#2f8fc9",
                                                         "#4a6ad6"])
FONTS = Path.home() / ".cache" / "majortom" / "fonts"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-{}.ttf"   # SIL OFL 1.1


def use_major_tom_font() -> None:
    """IBM Plex Mono, the Major TOM wordmark's typeface: fetched once into ~/.cache/majortom/fonts, then used for
    every text in the figures. No fallback: a figure in another font is an error, not a silent substitute."""
    import urllib.request

    from matplotlib import font_manager

    FONTS.mkdir(parents=True, exist_ok=True)
    for weight in ("Regular", "Bold"):
        path = FONTS / f"IBMPlexMono-{weight}.ttf"
        if not path.exists():
            urllib.request.urlretrieve(FONT_URL.format(weight), path)
        font_manager.fontManager.addfont(str(path))
    if "IBM Plex Mono" not in {f.name for f in font_manager.fontManager.ttflist}:
        raise RuntimeError(f"IBM Plex Mono not registered from {FONTS}")
    matplotlib.rcParams["font.family"] = "IBM Plex Mono"


def inside_earth() -> np.ndarray:
    """(ROWS, COLS) mask of grid positions that are cells (each row has its own number of columns)."""
    rows = np.arange(-ROWS // 2, ROWS // 2)
    lat = rows * 180 / ROWS
    n = np.maximum(np.ceil(2 * np.pi * R_KM * np.cos(np.radians(lat)) / D_KM), 1).astype(int)
    cols = np.arange(-COLS // 2, COLS // 2)
    mask = (cols[None, :] >= -(n[:, None] // 2)) & (cols[None, :] <= n[:, None] - 1 - n[:, None] // 2)
    return mask[::-1]                                                    # north up


def grid_image(rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Values on the (row, column) array, north up; NaN where there is no cell or no value."""
    image = np.full((ROWS, COLS), np.nan, np.float32)
    image[ROWS - (rows + ROWS // 2) - 1, cols + COLS // 2] = values
    return image


def _blocks(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Sums over an h x w grid of near-equal blocks."""
    ry = np.linspace(0, a.shape[0], h + 1).astype(int)[:-1]
    rx = np.linspace(0, a.shape[1], w + 1).astype(int)[:-1]
    return np.add.reduceat(np.add.reduceat(a, ry, axis=0), rx, axis=1)


def coloured(image: np.ndarray, cmap, norm, w: int) -> np.ndarray:
    """RGBA picture w pixels wide: block mean of the cells' values -> cmap(norm(mean)); blocks mostly without
    data dark, blocks mostly outside the Earth transparent."""
    h = w // 2
    inside = inside_earth().astype(np.float64)
    ok = np.isfinite(image)
    total = _blocks(np.ones_like(inside), h, w)
    share_in = _blocks(inside, h, w) / total
    n_ok = _blocks(ok.astype(np.float64), h, w)
    mean = _blocks(np.where(ok, image, 0).astype(np.float64), h, w) / np.maximum(n_ok, 1)
    has = n_ok / np.maximum(_blocks(inside, h, w), 1) >= 0.5
    rgba = np.zeros((h, w, 4))
    rgba[share_in >= 0.5] = to_rgba(MAP_EMPTY)
    pick = (share_in >= 0.5) & has
    rgba[pick] = cmap(norm(np.clip(mean[pick], norm.vmin, norm.vmax)))
    return rgba


def draw_map(fig, rect: list[float], image: np.ndarray, cmap, norm, name: str | None = None, size: int = 11) -> None:
    """One sinusoidal map in `rect` (figure fractions): cells rasterised at the axes' saved width, coastlines on top."""
    from mpl_toolkits.basemap import Basemap

    ax = fig.add_axes(rect)
    m = Basemap(projection="sinu", lon_0=0, resolution="l", ax=ax)
    width = int(fig.get_figwidth() * rect[2] * 100 * SCALE)
    picture = coloured(image, cmap, norm, width)
    ax.imshow(picture, extent=(m.llcrnrx, m.urcrnrx, m.llcrnry, m.urcrnry), interpolation="nearest", zorder=1)
    m.drawcoastlines(color=COAST, linewidth=0.3, zorder=2)
    ax.set_axis_off()
    if name:
        ax.text(0.02, 0.97, name, transform=ax.transAxes, fontsize=size, fontweight="bold", color=INK, va="top")


def colour_bar(fig, rect: list[float], cmap, norm, ticks, labels, unit: str, size: int = 8) -> None:
    cax = fig.add_axes(rect)
    bar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal", ticks=ticks)
    bar.set_ticklabels(labels)
    bar.minorticks_off()
    bar.outline.set_visible(False)
    cax.tick_params(labelsize=size, colors=INK_2, length=0, pad=3)
    cax.set_xlabel(unit, fontsize=size, color=INK_2, labelpad=2)


def map_figure(panels: list[tuple[np.ndarray, str | None]], cmap, norm, ticks: list[float], labels: list[str],
               title: str, subtitle: str, unit: str, out: Path, empty_label: str | None = None) -> None:
    """One or more sinusoidal maps stacked, sharing one colour scale; header above, colour bar below."""
    head, panel, foot = 0.8, 4.45, 0.75                                  # inches
    height = head + panel * len(panels) + foot
    fig = plt.figure(figsize=(WIDTH_PX / 100, height), dpi=100, facecolor=PAGE)
    fig.text(0.02, 1 - 0.25 / height, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.55 / height, subtitle, fontsize=9, color=INK_2, va="top")
    for i, (image, name) in enumerate(panels):
        bottom = (foot + panel * (len(panels) - 1 - i)) / height
        draw_map(fig, [0.01, bottom, 0.98, panel / height], image, cmap, norm, name)
    colour_bar(fig, [0.25, 0.42 / height, 0.46, 0.16 / height], cmap, norm, ticks, labels, unit)
    if empty_label:
        fig.patches.append(plt.Rectangle((0.74, 0.42 / height), 0.016, 0.16 / height, transform=fig.transFigure,
                                         facecolor=MAP_EMPTY, edgecolor="none"))
        fig.text(0.762, 0.50 / height, empty_label, fontsize=8, color=INK_2, va="center")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100 * SCALE, facecolor=PAGE)
    plt.close(fig)


HOURS_IN = {m: 24 * d for m, d in zip(range(1, 13), [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])}
PLACES = {"Loch Lomond": (624, -29), "Dolomites (Corvara)": (518, 90), "Amazon (Manaus)": (-35, -667)}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3: validated all-pairs, light surface


def normals(release: Path, dataset: str, columns: list[str]) -> pd.DataFrame:
    """Every cell's 12 monthly normals of `columns` (grid_row_u, grid_col_r, month, ...)."""
    return pd.concat(pq.read_table(f, columns=["grid_row_u", "grid_col_r", "month", *columns]).to_pandas()
                     for f in sorted((release / dataset / "climatology").glob("band=*.parquet")))


def annual(release: Path, dataset: str, column: str, per_month=None, how: str = "sum") -> pd.DataFrame:
    """Per cell: the sum (or mean) over the 12 monthly normals of `column` (NaN unless all 12 months have one);
    `per_month` scales each month first (e.g. a rate in mm/h times the hours in the month)."""
    t = normals(release, dataset, [column])
    if per_month is not None:
        t[column] = t[column] * t["month"].map(per_month)
    g = t.groupby(["grid_row_u", "grid_col_r"])[column]
    out = g.sum(min_count=12) if how == "sum" else g.mean().where(g.count() == 12)
    return out.reset_index()


def _spec(*args, **kwargs) -> tuple:
    """A map figure's description: map_figure's arguments, drawn alone (FIGURES) or as an overview tile."""
    return args, kwargs


def _image(t: pd.DataFrame, column: str) -> np.ndarray:
    return grid_image(t.grid_row_u.to_numpy(), t.grid_col_r.to_numpy(), t[column].to_numpy())


def precipitation_scale():
    """Dry sand -> green -> wet blue, log 50-4,000 mm a year (shared by every precipitation map)."""
    norm = LogNorm(vmin=50, vmax=4000)
    ticks = [50, 100, 250, 500, 1000, 2000, 4000]
    return DRY_WET, norm, ticks, [f"{t:,}" for t in ticks]


def cpc_precip(release: Path) -> tuple:
    a = annual(release, "CPC-Monthly", "cpc_precip_mean")
    cmap, norm, ticks, labels = precipitation_scale()
    return _spec([(_image(a, "cpc_precip_mean"), None)], cmap, norm, ticks, labels,
               "Annual precipitation, CPC (rain gauges), 1991–2020",
               "NOAA CPC Global Unified, normal 1991–2020 · CPC-Monthly · 1,657,095 land cells",
               "mm per year (log scale)", release / "CPC-Monthly" / "figures" / "annual-precipitation.png",
               empty_label="no data (sea)")


def imerg_precip(release: Path) -> tuple:
    a = annual(release, "IMERG-Monthly", "imerg_precip_mean", per_month=HOURS_IN)
    cmap, norm, ticks, labels = precipitation_scale()
    return _spec([(_image(a, "imerg_precip_mean"), None)], cmap, norm, ticks, labels,
               "Annual precipitation, IMERG (satellites), 1998–2024",
               "NASA GPM IMERG Final V07, normal 1998–2024 · IMERG-Monthly · 5,109,586 cells, land and sea",
               "mm per year (log scale; same scale as CPC-Monthly)",
               release / "IMERG-Monthly" / "figures" / "annual-precipitation.png")


# temperature, cold -> 0 °C -> hot: blue arm, neutral grey midpoint, red arm (each arm validated: lightness order,
# step gaps, contrast with the dark map surface)
COLD_HOT = LinearSegmentedColormap.from_list("cold_hot", ["#2c66c8", "#5b8ee0", "#9dbbea", "#dddbd5", "#eeaa9f",
                                                           "#e0675a", "#c8372f"])


def cpc_temperature(release: Path) -> tuple:
    t = normals(release, "CPC-Monthly", ["cpc_tmax_mean"])
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    ticks = [-40, -20, -10, 0, 10, 20, 30, 40]
    panels = [(_image(t[t.month == m], "cpc_tmax_mean"), name) for m, name in ((1, "January"), (7, "July"))]
    return _spec(panels, COLD_HOT, norm, ticks, [f"{v:+d}".replace("+0", "0") for v in ticks],
               "Daily maximum temperature, CPC, 1991–2020",
               "NOAA CPC Global Unified, normal 1991–2020 · CPC-Monthly · mean of the month's daily maxima",
               "°C (grey: 0 °C, freezing)", release / "CPC-Monthly" / "figures" / "maximum-temperature.png",
               empty_label="no data (sea)")


# haze, clean -> heavy: cream to burnt orange (validated: one hue, lightness order, contrast with the dark map)
HAZE = LinearSegmentedColormap.from_list("haze", ["#f6e6c8", "#f2c784", "#eba04f", "#d9742c", "#b9521f"])


def cams_aerosol(release: Path) -> tuple:
    a = annual(release, "CAMS-Monthly", "cams_aod550_mean", how="mean")
    ticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1]
    return _spec([(_image(a, "cams_aod550_mean"), None)], HAZE, LogNorm(vmin=0.03, vmax=1), ticks,
               [f"{t:g}" for t in ticks], "Aerosol optical depth at 550 nm, CAMS, 2003–2025",
               "Copernicus CAMS reanalysis EAC4, mean 2003–2025 · CAMS-Monthly · total aerosol at 550 nm",
               "optical depth, no unit (log scale)",
               release / "CAMS-Monthly" / "figures" / "aerosol-optical-depth.png")


def cams_pm25(release: Path) -> tuple:
    a = annual(release, "CAMS-Monthly", "cams_pm2p5_mean", how="mean")
    a["cams_pm2p5_mean"] *= 1e9                                          # kg m-3 -> µg m-3, for the labels only
    ticks = [2, 5, 10, 25, 50, 100, 150]
    return _spec([(_image(a, "cams_pm2p5_mean"), None)], HAZE, LogNorm(vmin=2, vmax=150), ticks,
               [f"{t:g}" for t in ticks], "Surface PM2.5, CAMS, 2003–2025",
               "Copernicus CAMS reanalysis EAC4, mean 2003–2025 · CAMS-Monthly · stored in kg/m³",
               "µg/m³ (log scale; WHO annual guideline 5)",
               release / "CAMS-Monthly" / "figures" / "pm25.png")


# height, low -> high: dark green, olive, tan, snow (validated: lightness order, step gaps, contrast)
TERRAIN = LinearSegmentedColormap.from_list("terrain", ["#3f7f4f", "#7fa363", "#b6b679", "#d6c9a2", "#ebe3d4",
                                                         "#f9f8f5"])


def era5_height(release: Path) -> tuple:
    s = pq.read_table(release / "ERA5-Static" / "static.parquet",
                      columns=["grid_row_u", "grid_col_r", "era5_z", "era5_lsm"]).to_pandas()
    s["height"] = np.where(s.era5_lsm >= 0.5, s.era5_z / 9.80665, np.nan)    # sea left dark
    ticks = [0, 250, 500, 1000, 2000, 3000, 5000]
    return _spec([(_image(s, "height"), None)], TERRAIN, PowerNorm(0.5, vmin=0, vmax=5000), ticks,
               [f"{t:,}" for t in ticks], "Surface height, ERA5 model",
               "ECMWF ERA5, geopotential at the surface ÷ g · ERA5-Static · the model's smoothed terrain at 0.25°",
               "m above sea level (square-root scale)", release / "ERA5-Static" / "figures" / "surface-height.png",
               empty_label="sea")


def era5_precip(release: Path) -> tuple:
    """ERA5's monthly means of precipitation are per day (m day⁻¹): x days in the month x 1000 -> mm."""
    a = annual(release, "ERA5-Monthly", "era5_tp_mean", per_month={m: h / 24 * 1000 for m, h in HOURS_IN.items()})
    cmap, norm, ticks, labels = precipitation_scale()
    return _spec([(_image(a, "era5_tp_mean"), None)], cmap, norm, ticks, labels,
               "Annual precipitation, ERA5 (reanalysis), 1991–2020",
               "ECMWF ERA5, normal 1991–2020 · ERA5-Monthly · 5,113,745 cells, land and sea",
               "mm per year (log scale)", release / "ERA5-Monthly" / "figures" / "annual-precipitation.png")


def era5_temperature(release: Path) -> tuple:
    t = normals(release, "ERA5-Monthly", ["era5_t2m_mean"])
    t["era5_t2m_mean"] -= 273.15                                          # K -> °C, for the scale shared with CPC
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    ticks = [-40, -20, -10, 0, 10, 20, 30, 40]
    panels = [(_image(t[t.month == m], "era5_t2m_mean"), name) for m, name in ((1, "January"), (7, "July"))]
    return _spec(panels, COLD_HOT, norm, ticks, [f"{v:+d}".replace("+0", "0") for v in ticks],
               "2 m temperature, ERA5, 1991–2020",
               "ECMWF ERA5, normal 1991–2020 · ERA5-Monthly · monthly mean of the 2 m air temperature",
               "°C (grey: 0 °C, freezing)", release / "ERA5-Monthly" / "figures" / "temperature.png")


def _save_vector(fig, out: Path) -> None:
    """Line charts as SVG: sharp at any display size; text as outlines, so no font is needed to view them."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(out, format="svg", facecolor=PAGE)


def _cell_normals(release: Path, dataset: str, columns: list[str], row: int, col: int) -> pd.DataFrame:
    band = (row + 1002) * 20 // 2004
    t = pq.read_table(release / dataset / "climatology" / f"band={band:02d}.parquet",
                      columns=["month", *columns], filters=[("grid_row_u", "=", row), ("grid_col_r", "=", col)])
    return t.to_pandas().sort_values("month")


def _axes_style(ax, ylabel: str | None) -> None:
    ax.set_facecolor(PAGE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)


def places(release: Path) -> None:
    """Monthly precipitation (CPC gauges, IMERG satellites, ERA5 reanalysis) and temperature at the three places."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_PX / 100, 6.4), dpi=100, facecolor=PAGE, sharex=True, sharey="row",
                             gridspec_kw=dict(left=0.09, right=0.95, top=0.74, bottom=0.07, hspace=0.42, wspace=0.12))
    fig.text(0.02, 0.965, "Monthly precipitation and temperature: Loch Lomond, Corvara, Manaus", fontsize=13, fontweight="bold",
             color=INK, va="top")
    fig.text(0.02, 0.915, "Monthly normals of precipitation: CPC rain gauges 1991–2020, IMERG satellites 1998–2024, "
             "ERA5 1991–2020", fontsize=9, color=INK_2, va="top")
    for j, (name, (row, col)) in enumerate(PLACES.items()):
        cpc = _cell_normals(release, "CPC-Monthly", ["cpc_precip_mean", "cpc_tmax_mean", "cpc_tmin_mean"], row, col)
        imerg = _cell_normals(release, "IMERG-Monthly", ["imerg_precip_mean"], row, col)
        rain = imerg.imerg_precip_mean.to_numpy() * imerg.month.map(HOURS_IN).to_numpy()
        era5 = _cell_normals(release, "ERA5-Monthly", ["era5_tp_mean"], row, col)                 # m per day
        era5_rain = era5.era5_tp_mean.to_numpy() * era5.month.map(HOURS_IN).to_numpy() / 24 * 1000
        top, bottom = axes[0, j], axes[1, j]
        top.plot(cpc.month, cpc.cpc_precip_mean, color=SERIES[0], linewidth=2, label="CPC (gauges)")
        top.plot(imerg.month, rain, color=SERIES[1], linewidth=2, label="IMERG (satellites)")
        top.plot(era5.month, era5_rain, color=SERIES[2], linewidth=2, label="ERA5 (reanalysis)")
        code = f"{abs(row)}{'U' if row >= 0 else 'D'}_{abs(col)}{'R' if col >= 0 else 'L'}"
        top.set_title(f"{name}\n{code}", fontsize=10, color=INK, loc="left", fontweight="bold")
        bottom.fill_between(cpc.month, cpc.cpc_tmin_mean, cpc.cpc_tmax_mean, color=SERIES[0], alpha=0.18,
                            linewidth=0)
        bottom.plot(cpc.month, cpc.cpc_tmax_mean, color=SERIES[0], linewidth=2)
        bottom.plot(cpc.month, cpc.cpc_tmin_mean, color=SERIES[0], linewidth=2, linestyle=(0, (3, 2)))
        _axes_style(top, "precipitation, mm per month" if j == 0 else None)
        _axes_style(bottom, "temperature, °C" if j == 0 else None)
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 2].text(12.2, axes[1, 2].lines[0].get_ydata()[-1], "max", fontsize=8, color=INK_2, va="center")
    axes[1, 2].text(12.2, axes[1, 2].lines[1].get_ydata()[-1], "min", fontsize=8, color=INK_2, va="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=3, frameon=False, fontsize=8,
               labelcolor=INK_2, handlelength=1.6)
    y = axes[1, 0].get_position().y1
    fig.text(0.09, y + 0.025, "CPC daily maximum (solid) and minimum (dashed) temperature, mean of the month, 1991–2020",
             fontsize=8, color=INK_2)
    _save_vector(fig, release / "figures" / "places.svg")
    plt.close(fig)


def air_quality_seasons(release: Path) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_PX / 100, 3.8), dpi=100, facecolor=PAGE,
                           gridspec_kw=dict(left=0.08, right=0.86, top=0.74, bottom=0.12))
    fig.text(0.02, 0.95, "Monthly aerosol optical depth: Loch Lomond, Corvara, Manaus", fontsize=13, fontweight="bold", color=INK,
             va="top")
    fig.text(0.02, 0.87, "CAMS reanalysis EAC4, monthly normals 2003–2025 · total aerosol optical depth at 550 nm",
             fontsize=9, color=INK_2, va="top")
    for colour, (name, (row, col)) in zip(SERIES, PLACES.items()):
        t = _cell_normals(release, "CAMS-Monthly", ["cams_aod550_mean"], row, col)
        ax.plot(t.month, t.cams_aod550_mean, color=colour, linewidth=2, label=name)
        ax.text(12.25, t.cams_aod550_mean.iloc[-1], name, fontsize=8, color=INK, va="center")
    _axes_style(ax, "optical depth")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    _save_vector(fig, release / "figures" / "air-quality-seasons.svg")
    plt.close(fig)


def overview(release: Path) -> None:
    """Eight maps, one or two per dataset, for the top-level README."""
    tiles = [(cpc_precip, 0, "Annual precipitation (CPC-Monthly)", "", "mm per year"),
             (imerg_precip, 0, "Annual precipitation (IMERG-Monthly)", "", "mm per year"),
             (era5_precip, 0, "Annual precipitation (ERA5-Monthly)", "", "mm per year"),
             (era5_temperature, 1, "July 2 m temperature (ERA5-Monthly)", "", "°C"),
             (cpc_temperature, 1, "July daily maximum temperature (CPC-Monthly)", "", "°C"),
             (era5_height, 0, "Surface height (ERA5-Static)", "", "m"),
             (cams_aerosol, 0, "Aerosol optical depth at 550 nm (CAMS-Monthly)", "", "no unit"),
             (cams_pm25, 0, "Surface PM2.5 (CAMS-Monthly)", "", "µg/m³")]
    head, map_h, title_h, bar_h, gap = 0.95, 2.2, 0.32, 0.62, 0.15                      # inches
    row_h = title_h + map_h + bar_h + gap
    height = head + row_h * -(-len(tiles) // 2)
    width = WIDTH_PX / 100
    fig = plt.figure(figsize=(width, height), dpi=100, facecolor=PAGE)
    fig.text(0.02, 1 - 0.25 / height, "Major TOM Climate", fontsize=13, fontweight="bold", color=INK,
             va="top")
    fig.text(0.02, 1 - 0.58 / height, "Normals per Major TOM cell: CPC and ERA5 1991–2020, IMERG 1998–2024, CAMS 2003–2025. Sinusoidal projection.",
             fontsize=9, color=INK_2, va="top")
    for i, (spec, panel, name, dataset, unit) in enumerate(tiles):
        (panels, cmap, norm, ticks, labels, *_), _ = spec(release)
        row, col = divmod(i, 2)
        left = 0.01 + col * 0.5
        top = height - head - row * row_h
        fig.text(left + 0.01, (top - 0.05) / height, name, fontsize=9, fontweight="bold", color=INK, va="top")
        draw_map(fig, [left, (top - title_h - map_h) / height, 0.48, map_h / height], panels[panel][0], cmap, norm)
        colour_bar(fig, [left + 0.06, (top - title_h - map_h - 0.22) / height, 0.36, 0.1 / height], cmap, norm,
                   ticks, labels, unit, size=7)
    out = release / "figures" / "overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100 * SCALE, facecolor=PAGE)
    plt.close(fig)


def _draw(spec):
    return lambda release: (lambda a, k: map_figure(*a, **k))(*spec(release))


FIGURES = {"cpc-precip": _draw(cpc_precip), "imerg-precip": _draw(imerg_precip),
           "cpc-temperature": _draw(cpc_temperature), "cams-aerosol": _draw(cams_aerosol),
           "cams-pm25": _draw(cams_pm25), "era5-height": _draw(era5_height), "era5-precip": _draw(era5_precip),
           "era5-temperature": _draw(era5_temperature), "overview": overview,
           "places": places, "air-quality": air_quality_seasons}

if __name__ == "__main__":
    use_major_tom_font()
    release = Path(sys.argv[1])
    for name in sys.argv[2:] or FIGURES:
        FIGURES[name](release)
        print(name, "done", flush=True)
