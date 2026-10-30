"""Images for the climate globe: one picture per dataset, variable and month, one pixel per Major TOM cell.

    climatology/band=NN.parquet (grid_row_u, grid_col_r, month, <variable>_mean)       (the published release)
      -> per month, every cell's normal on the (row, column) array, north up     (docs/evidence/climate_figures.py)
      -> colour scale as in the README figures (log scale for precipitation); no cell or no value: transparent
      -> images/<dataset>/<variable>/MM.png (4008 x 2004: pixel (x, y) is the cell grid_col_r = x - 2004,
         grid_row_u = 1001 - y; the page finds a point's cell as for live data, so colours and cell outlines match)
    Units and colour scales are those of the page's layers (data.js), so a preview and a live month look alike.
    Runs in the miko-torch environment (shares the figure script's imports):

    sbatch ~/slurm-templates/cpu.sbatch python apps/climate-globe/build_images.py /data/Major-TOM/climate-release
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "evidence"))
from climate_figures import COLD_HOT, DRY_WET, HAZE, HOURS_IN, grid_image, normals  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"

# dataset, folder, climatology column, scale per month (to the page's unit), colour scale, normalisation
LAYERS = [
    ("CPC-Monthly", "precip", "cpc_precip_mean", lambda m: 1, DRY_WET, LogNorm(vmin=4, vmax=400)),          # mm/month
    ("IMERG-Monthly", "precip", "imerg_precip_mean", lambda m: HOURS_IN[m], DRY_WET, LogNorm(vmin=4, vmax=400)),  # mm/h
    ("CPC-Monthly", "tmax", "cpc_tmax_mean", lambda m: 1, COLD_HOT, TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)),   # °C
    ("CAMS-Monthly", "pm25", "cams_pm2p5_mean", lambda m: 1e9, HAZE, LogNorm(vmin=2, vmax=150)),          # kg/m³
]


def picture(v: np.ndarray, cmap, norm) -> Image.Image:
    """RGBA picture of the (row, column) array: each pixel its cell's colour; transparent where there is none."""
    # outside the scale -> its end colours (a log scale has no place for 0 mm: a month without rain is driest)
    rgba = (cmap(norm(np.clip(np.ma.masked_invalid(v), norm.vmin, norm.vmax))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(v), 255, 0)
    return Image.fromarray(rgba, "RGBA")


def build(release: Path) -> None:
    for dataset, variable, column, per_month, cmap, norm in LAYERS:
        t = normals(release, dataset, [column])
        folder = OUT / dataset / variable
        folder.mkdir(parents=True, exist_ok=True)
        for month in range(1, 13):
            m = t[t["month"] == month]
            image = grid_image(m.grid_row_u.to_numpy(), m.grid_col_r.to_numpy(), m[column].to_numpy() * per_month(month))
            picture(image, cmap, norm).save(folder / f"{month:02d}.png", optimize=True)
            print(f"{dataset} {variable} {month:02d}", flush=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
