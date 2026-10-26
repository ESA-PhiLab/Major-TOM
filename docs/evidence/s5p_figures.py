"""Figures of Core-Weather's Sentinel-5P table (docs/figures/s5p_*.png).

    <root>/daily/year=YYYY/<acquisition date>.parquet   one row per acquisition and daily step (-30 .. +30)
      -> A. maps:   step 0 of every acquisition in one year, median per 0.5° pixel, one panel per variable
      -> B. series: all 61 steps of one acquisition at each of three places (PLACES: a city, a dust source, a clean
                    reference), the acquisition at the cell nearest the place with the most valid NO2 and aerosol
      -> C. valid:  share of rows with a valid value, per variable, over the Sentinel-5P record (from 2018-06-28)

    python docs/evidence/s5p_figures.py /data/Major-TOM/Core-Weather/s5p docs/figures 2022     (Slurm CPU job)

Colours: the dataviz reference palette (sequential blue ramp; categorical slots 1-3, validated there).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import Cell                                          # noqa: E402

VARS = {"no2": ("Tropospheric NO₂", "mol m⁻²", True), "co": ("CO", "mol m⁻²", False),
        "o3": ("O₃", "mol m⁻²", False), "ch4": ("CH₄", "ppb", False), "so2": ("SO₂", "mol m⁻²", False),
        "hcho": ("Tropospheric HCHO", "mol m⁻²", False), "aer_ai": ("Absorbing aerosol index", "", False)}
FIRST_DAY = pd.Timestamp("2018-06-28")
PLACES = {"Delhi (city)": (77.2, 28.6), "Bodélé Depression, Chad (dust)": (17.8, 16.9),
          "Central Australia (clean)": (133.9, -23.7)}
PIXEL = 0.5                                                              # map pixel, degrees
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})


def lonlat(names: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Cell centroids of grid cell names such as 495D_1175R (row 495 down, column 1175 right)."""
    unique = names.unique()
    xy = {}
    for name in unique:
        r, ud, c, lr = re.fullmatch(r"(\d+)([UD])_(\d+)([LR])", name).groups()
        xy[name] = Cell(int(r) * (1 if ud == "U" else -1), int(c) * (1 if lr == "R" else -1), 10.0).centroid
    pts = np.array([xy[n] for n in names])
    return pts[:, 0], pts[:, 1]


def raster(lon: np.ndarray, lat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Median of x per PIXEL-degree pixel, north up; NaN where no acquisition falls."""
    col = ((lon + 180) / PIXEL).astype(int).clip(0, int(360 / PIXEL) - 1)
    row = ((90 - lat) / PIXEL).astype(int).clip(0, int(180 / PIXEL) - 1)
    grid = np.full((int(180 / PIXEL), int(360 / PIXEL)), np.nan, np.float32)
    med = pd.Series(x).groupby(row * 100_000 + col).median()
    grid[med.index // 100_000, med.index % 100_000] = med.to_numpy()
    return grid


def maps(day0: pd.DataFrame, year: int, out: Path) -> None:
    """A: one panel per variable; values where the cell is valid over at least half its area, median per pixel."""
    lon, lat = day0.lon.to_numpy(), day0.lat.to_numpy()
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), constrained_layout=True)
    for ax, (v, (label, unit, log)) in zip(axes.flat, VARS.items()):
        ok = (day0[f"s5p_{v}_valid"] >= 0.5).to_numpy() & np.isfinite(day0[f"s5p_{v}"].to_numpy())
        x = day0[f"s5p_{v}"].to_numpy()[ok]
        lo, hi = np.percentile(x, [2, 98]) if len(x) else (0, 1)
        norm = LogNorm(max(lo, hi / 100, 1e-9), hi) if log else Normalize(lo, hi)
        dots = ax.imshow(raster(lon[ok], lat[ok], np.clip(x, norm.vmin, norm.vmax)), cmap=BLUES, norm=norm,
                         extent=(-180, 180, -90, 90), interpolation="nearest")
        ax.set(xlim=(-180, 180), ylim=(-60, 85), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{label}  ({ok.sum():,} acquisitions)", loc="left")
        fig.colorbar(dots, ax=ax, shrink=0.8, pad=0.01, label=f"{unit}, 2nd–98th percentile" if unit else
                     "2nd–98th percentile")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.5, f"Sentinel-5P at each Core acquisition, {year}\n\nStep 0: the UTC day of the overpass\n"
                       "on the acquisition's local day. Median of the\nacquisitions in each 0.5° pixel; cells valid\n"
                       "over less than half their area are left out.", va="center", color=MUTED)
    fig.savefig(out / f"s5p_maps_{year}.png", dpi=150)
    plt.close(fig)


def series(rows: pd.DataFrame, picks: dict[str, tuple[str, pd.Timestamp]], out: Path) -> None:
    """B: the 61 daily steps of three acquisitions; gaps where the value is invalid over half the cell or more."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), constrained_layout=True, sharex=True)
    for ax, (v, (label, unit, _)) in zip(axes.flat, VARS.items()):
        for colour, (name, (cell, when)) in zip(SERIES, picks.items()):
            r = rows[(rows.grid_cell == cell) & (rows.acquired == when)].sort_values("step_d")
            y = r[f"s5p_{v}"].where(r[f"s5p_{v}_valid"] >= 0.5)
            ax.plot(r.step_d, y, color=colour, lw=2, label=name)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(label, loc="left")
        ax.set_ylabel(unit)
    for ax in axes[-1]:
        ax.set_xlabel("days from the acquisition")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center left", frameon=False, title="Acquisition")
    fig.savefig(out / "s5p_series.png", dpi=150)
    plt.close(fig)


def valid(root: Path, out: Path) -> pd.Series:
    """C: mean valid share per variable over all rows from the first Sentinel-5P day; read file by file."""
    cols = [f"s5p_{v}_valid" for v in VARS]
    total, n = np.zeros(len(cols)), 0
    for f in sorted((root / "daily").rglob("*.parquet")):
        t = ds.dataset(f).to_table(columns=cols, filter=ds.field("s5p_start") >= FIRST_DAY).to_pandas()
        total += t.fillna(0).sum().to_numpy()
        n += len(t)
    share = pd.Series(total / n, index=[VARS[v][0] for v in VARS]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.barh(share.index, share.values, color=SERIES[0], height=0.6)
    for y, s in enumerate(share.values):
        ax.text(s + 0.01, y, f"{s:.0%}", va="center", color=INK)
    ax.set(xlim=(0, 1.1), xticks=[0, 0.25, 0.5, 0.75, 1], xticklabels=["0", "25%", "50%", "75%", "100%"])
    ax.set_title(f"Sentinel-5P: mean share of each cell with a valid value\n{n:,} daily rows from 2018-06-28",
                 loc="left")
    fig.savefig(out / "s5p_valid.png", dpi=150)
    plt.close(fig)
    return share


def main(root: str, out: str, year: str) -> None:
    root, out, year = Path(root), Path(out), int(year)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["grid_cell", "acquired", "step_d", *[f"s5p_{v}{s}" for v in VARS for s in ("", "_valid")]]
    day0 = ds.dataset(root / "daily" / f"year={year}").to_table(columns=cols, filter=ds.field("step_d") == 0) \
        .to_pandas()
    print(f"{year}: {len(day0):,} acquisitions at step 0", flush=True)
    day0["lon"], day0["lat"] = lonlat(day0.grid_cell)
    maps(day0, year, out)
    picks = {}
    for name, (x, y) in PLACES.items():
        distance = (day0.lon - x).abs() + (day0.lat - y).abs()
        near = day0[distance == distance.min()]
        row = near.loc[(near.s5p_no2_valid + near.s5p_aer_ai_valid).idxmax()]
        picks[f"{name}: {row.grid_cell}, {row.acquired:%Y-%m-%d}"] = (row.grid_cell, row.acquired)
    files = {root / "daily" / f"year={year}" / f"{w:%Y-%m-%d}.parquet" for _, w in picks.values()}
    rows = pd.concat(ds.dataset(f).to_table(columns=cols).to_pandas() for f in files)
    series(rows, picks, out)
    print("series:", list(picks), flush=True)
    print(valid(root, out).to_string(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
