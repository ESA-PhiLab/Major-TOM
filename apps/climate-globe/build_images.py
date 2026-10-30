"""Images for the climate globe: one longitude/latitude picture per dataset, variable and month.

    climatology/band=NN.parquet (grid_row_u, grid_col_r, month, <variable>_mean)       (the published release)
      -> per month, every cell's normal on the 2004 x 4008 (row, column) array     (docs/evidence/climate_figures.py)
      -> each output pixel takes the value of the Major TOM cell its centre falls in (no interpolation: every
         pixel shows one cell's real value, as in the README maps)
      -> colour scale as in the README figures (log scale for precipitation); no cell or no value: transparent
      -> images/<dataset>/<variable>/MM.png (WIDTH x WIDTH/2, plate carrée: what a web globe drapes over the sphere)
         and images/images.json, which lists them with their scale for the page
    Runs in the miko-torch environment (shares the figure script's imports):

    sbatch ~/slurm-templates/cpu.sbatch python apps/climate-globe/build_images.py /data/Major-TOM/climate-release
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "evidence"))
from climate_figures import COLS, D_KM, DRY_WET, R_KM, ROWS, grid_image, normals  # noqa: E402

WIDTH = 4096                             # ~9.8 km per pixel at the equator, close to the 10 km cells
OUT = Path(__file__).resolve().parent / "images"

# dataset, variable, column, title, unit, colour scale, normalisation, ticks
LAYERS = [
    ("CPC-Monthly", "precip", "cpc_precip_mean", "Precipitation, monthly normal, CPC (rain gauges), 1991–2020",
     "mm per month (log scale)", DRY_WET, LogNorm(vmin=4, vmax=400), [4, 10, 25, 50, 100, 200, 400]),
]


def cell_index(width: int = WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """(height, width) indices into the (row, column) array of the cell under each pixel centre (north up)."""
    lat = 90 - (np.arange(width // 2) + 0.5) * 180 / (width // 2)
    lon = -180 + (np.arange(width) + 0.5) * 360 / width
    dlat = 180 / ROWS
    row = np.clip(np.floor(lat / dlat).astype(int), -ROWS // 2, ROWS // 2 - 1)      # Major TOM row of each latitude
    n_cols = np.maximum(np.ceil(2 * np.pi * R_KM * np.cos(np.radians(row * dlat)) / D_KM), 1).astype(int)
    col = np.floor(lon[None, :] / (360 / n_cols[:, None])).astype(int)              # column within that row
    col = np.clip(col, -(n_cols[:, None] // 2), n_cols[:, None] - 1 - n_cols[:, None] // 2)
    return np.broadcast_to((ROWS - (row + ROWS // 2) - 1)[:, None], col.shape), col + COLS // 2


def picture(values: np.ndarray, at: tuple[np.ndarray, np.ndarray], cmap, norm) -> Image.Image:
    """RGBA picture: each pixel the colour of its cell's value; transparent where there is none."""
    v = values[at]
    # outside the scale -> its end colours (a log scale has no place for 0 mm: a month without rain is driest)
    rgba = (cmap(norm(np.clip(np.ma.masked_invalid(v), norm.vmin, norm.vmax))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(v), 255, 0)
    return Image.fromarray(rgba, "RGBA")


def build(release: Path) -> None:
    at = cell_index()
    listing = []
    for dataset, variable, column, title, unit, cmap, norm, ticks in LAYERS:
        t = normals(release, dataset, [column])
        folder = OUT / dataset / variable
        folder.mkdir(parents=True, exist_ok=True)
        for month in range(1, 13):
            m = t[t["month"] == month]
            image = grid_image(m.grid_row_u.to_numpy(), m.grid_col_r.to_numpy(), m[column].to_numpy())
            picture(image, at, cmap, norm).save(folder / f"{month:02d}.png", optimize=True)
            print(f"{dataset} {variable} {month:02d}", flush=True)
        listing.append({"dataset": dataset, "variable": variable, "title": title, "unit": unit,
                        "path": f"{dataset}/{variable}/{{month}}.png", "scale": "log",
                        "min": norm.vmin, "max": norm.vmax, "ticks": ticks,
                        "colours": [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
                                    for r, g, b, _ in cmap(np.linspace(0, 1, 9))]})
    (OUT / "images.json").write_text(json.dumps(listing, indent=1))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
