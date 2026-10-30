"""Images for the climate globe: one picture per dataset, variable and month, one pixel per Major TOM cell.

    climatology/band=NN.parquet (grid_row_u, grid_col_r, month, <variable>_mean)       (the published release)
      -> per month, every cell's normal on the (row, column) array, north up     (docs/evidence/climate_figures.py)
      -> colour scale as in the README figures (log scale for precipitation); no cell or no value: transparent
      -> images/<dataset>/<variable>/MM.png (4008 x 2004: pixel (x, y) is the cell grid_col_r = x - 2004,
         grid_row_u = 1001 - y; the page finds a point's cell as for live data, so colours and cell outlines match)
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
from climate_figures import DRY_WET, grid_image, normals  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"

# dataset, variable, column, title, unit, colour scale, normalisation, ticks
LAYERS = [
    ("CPC-Monthly", "precip", "cpc_precip_mean", "Precipitation, monthly normal, CPC (rain gauges), 1991–2020",
     "mm per month (log scale)", DRY_WET, LogNorm(vmin=4, vmax=400), [4, 10, 25, 50, 100, 200, 400]),
]


def picture(v: np.ndarray, cmap, norm) -> Image.Image:
    """RGBA picture of the (row, column) array: each pixel its cell's colour; transparent where there is none."""
    # outside the scale -> its end colours (a log scale has no place for 0 mm: a month without rain is driest)
    rgba = (cmap(norm(np.clip(np.ma.masked_invalid(v), norm.vmin, norm.vmax))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(v), 255, 0)
    return Image.fromarray(rgba, "RGBA")


def build(release: Path) -> None:
    listing = []
    for dataset, variable, column, title, unit, cmap, norm, ticks in LAYERS:
        t = normals(release, dataset, [column])
        folder = OUT / dataset / variable
        folder.mkdir(parents=True, exist_ok=True)
        for month in range(1, 13):
            m = t[t["month"] == month]
            image = grid_image(m.grid_row_u.to_numpy(), m.grid_col_r.to_numpy(), m[column].to_numpy())
            picture(image, cmap, norm).save(folder / f"{month:02d}.png", optimize=True)
            print(f"{dataset} {variable} {month:02d}", flush=True)
        listing.append({"dataset": dataset, "variable": variable, "title": title, "unit": unit,
                        "path": f"{dataset}/{variable}/{{month}}.png", "scale": "log",
                        "min": norm.vmin, "max": norm.vmax, "ticks": ticks,
                        "colours": [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
                                    for r, g, b, _ in cmap(np.linspace(0, 1, 9))]})
    (OUT / "images.json").write_text(json.dumps(listing, indent=1))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
