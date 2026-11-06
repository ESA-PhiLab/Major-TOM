"""A picture of every product from every archive, for the same Major TOM cell and month where possible.

    samples from tests/test_products.py  (MAJORTOM_EVIDENCE_DIR=<dir> pytest -m network tests/test_products.py)
      <dir>/samples/<archive>/<product>/000000.npz + .json
      -> per product: physical values (stored x scale + offset, Sentinel-2 offset removed), one display rule
      -> one row per product (grouped by kind), one column per archive, the same stretch along a row
      -> docs/reports/<date>-product-gallery.png

    python docs/evidence/gallery.py <dir> [date]
A missing tile says whether the archive has no such product or the sample failed.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.build.archives import ARCHIVES                       # noqa: E402
from majortom.build.collections import COLLECTIONS                 # noqa: E402

REPORTS = Path(__file__).resolve().parents[2] / "docs" / "reports"
KINDS = ["optical", "radar", "elevation", "land cover", "temperature", "night lights", "biomass", "water"]
def physical(meta: dict, arrays, band: str) -> np.ndarray:
    """Stored values as physical ones: the declared scale and offset (which include any Sentinel-2 offset), else
    the recorded Sentinel-2 offset removed and reflectance x 10 000 undone."""
    key = next(k for k in arrays.files if k == band or k.startswith(f"{band}_"))
    values = arrays[key].astype(float)
    scaling = (meta.get("scaling") or {}).get(band)
    offset = (meta.get("offset") or {}).get(band)
    if scaling:
        values = values * (scaling.get("scale") or 1) + (scaling.get("offset") or 0)
    elif offset is not None:                                     # Sentinel-2 without declared scaling
        values = (values - offset) * 0.0001
    db = COLLECTIONS[meta["collection"]].preview.db                # radar backscatter in dB (Preview.db)
    if db == "power":                                              # linear gamma0
        values = 10 * np.log10(np.clip(values, 1e-5, None))
    elif db == "amplitude":                                        # amplitude DN, uncalibrated
        values = 20 * np.log10(np.clip(values, 1, None))
    elif db == "palsar":                                           # DN -> gamma0 dB (JAXA formula)
        values = 10 * np.log10(np.clip(values, 1, None) ** 2) - 83
    return values


def image(name: str, meta: dict, arrays) -> tuple[np.ndarray, str | None]:
    """What to draw for one sample, from its product's Preview: an RGB stack, one band with its colour map, or
    classes (one colour per class present)."""
    view = COLLECTIONS[name].preview
    if view.rgb and all(b in meta["bands"] for b in view.rgb):
        return np.dstack([physical(meta, arrays, b) for b in view.rgb]), None
    if view.rgb:                                                 # optional bands missing (ASTER without VNIR)
        return physical(meta, arrays, meta["bands"][0]), "gray"
    if view.classes:
        values = physical(meta, arrays, view.classes)
        _, index = np.unique(values, return_inverse=True)
        return index.reshape(values.shape).astype(float), "tab20"
    return physical(meta, arrays, view.band), view.cmap


def stretch(images: list[np.ndarray]) -> tuple[float, float]:
    """2nd and 98th percentile over every archive's sample of one product, so a row shares one stretch."""
    parts = [im[np.isfinite(im)].ravel() for im in images if im.size]
    if not parts:
        return 0.0, 1.0
    values = np.concatenate(parts)
    return (float(np.percentile(values, 2)), float(np.percentile(values, 98))) if values.size else (0.0, 1.0)


def main(samples: Path, date: str) -> Path:
    archives = list(ARCHIVES)
    names = sorted(COLLECTIONS, key=lambda n: (KINDS.index(COLLECTIONS[n].kind), n))
    fig, axes = plt.subplots(len(names), len(archives), figsize=(2.3 * len(archives), 2.5 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        found = {}
        for archive in archives:
            stem = samples / "samples" / archive / name / "000000"
            if stem.with_suffix(".npz").exists():
                meta = json.loads(stem.with_suffix(".json").read_text())
                found[archive] = (meta, *image(name, meta, np.load(stem.with_suffix(".npz"))))
        limits = {cmap: stretch([im for _, im, cm in found.values() if cm == cmap])   # one stretch per style
                  for cmap in {cm for _, _, cm in found.values()}}
        for c, archive in enumerate(archives):
            ax = axes[r][c]
            ax.set_xticks([]), ax.set_yticks([])
            if archive in found:
                meta, im, cmap = found[archive]
                low, high = (0, 19) if cmap == "tab20" else limits[cmap]
                shown = np.clip((im - low) / (high - low or 1), 0, 1)
                ax.imshow(shown if cmap is None else im, cmap=cmap, vmin=None if cmap is None else low,
                          vmax=None if cmap is None else high, interpolation="nearest")
                note = f"  ({meta['bands'][0]} only)" if COLLECTIONS[name].preview.rgb and cmap == "gray" else ""
                ax.set_title(f"{meta['datetime'][:10]}  {meta['items'][0][-26:]}{note}", fontsize=6)
            else:
                why = "not in this archive" if COLLECTIONS[name].source(name, archive) is None else "no sample"
                ax.text(0.5, 0.5, why, ha="center", va="center", fontsize=7, color="0.5", transform=ax.transAxes)
                ax.set_facecolor("0.95")
            if r == 0:
                ax.set_xlabel(archive, fontsize=9)
                ax.xaxis.set_label_position("top")
            if c == 0:
                ax.set_ylabel(f"{name}\n({COLLECTIONS[name].kind})", fontsize=8)
    cells = sorted({json.loads(p.read_text())["grid_cell"] for p in (samples / "samples").glob("*/*/000000.json")})
    fig.suptitle(f"majortom.build products, cell {', '.join(cells)} ({date})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    out = REPORTS / f"{date}-product-gallery.png"
    fig.savefig(out, dpi=70)
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else dt.date.today().isoformat()))
