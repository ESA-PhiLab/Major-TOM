"""Colour scales for one monthly Climate dataset: one per variable, from its normals, written to scales/<dataset>.json.

    <dataset>/climatology_overviews/MM.parquet (block means of each <column>_mean, 12 calendar months): the columns
    + docs/datasets/climate/<dataset>.md (variables table: column, name, units), or NAMES for columns the table groups
      -> per variable, `*_valid` left out (coverage shares, not physical quantities):
         2nd and 98th percentile of the block normals over the 12 months
      -> scale: K or °C: linear, cold-hot; signed: centred on 0 (diverging);
         positive and skewed (median below 15% of the top): log; otherwise linear (sequential)
      -> scales/<dataset>.json: {column: {name, unit, scale: {type, min, max, centre?, ramp, ticks}}}, in the
         overviews' column order; "overviews": the columns with series overviews (globe-scale monthly maps)

    python build_scales.py /data/Major-TOM/climate-release/<dataset>        (Slurm, .venv311 or miko-torch)
    datasets: CPC-Monthly, IMERG-Monthly, CAMS-Monthly, ERA5Land-Monthly, ERA5-Monthly
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

APP = Path(__file__).resolve().parent
READMES = APP.parents[1] / "docs" / "datasets" / "climate"
ROW = re.compile(r"^\| `([a-z0-9]+_\w+)` \| (.+?) \| (.*?) \|$")

# CAMS-Monthly.md groups several columns into one table row; their names and units, row by row from that table
NAMES = {
    **{f"cams_tc_{g}": (f"total column of {n}", "kg m⁻²") for g, n in
       [("no2", "NO₂"), ("o3", "O₃"), ("co", "CO"), ("so2", "SO₂"), ("hcho", "formaldehyde"), ("ch4", "methane")]},
    **{f"cams_aod550_{t}": (f"aerosol optical depth at 550 nm, {n}", "dimensionless") for t, n in
       [("dust", "dust"), ("seasalt", "sea salt"), ("bc", "black carbon"), ("om", "organic matter"),
        ("sulphate", "sulphate")]},
    "cams_pm2p5": ("particulate matter below 2.5 µm, at the surface", "kg m⁻³"),
    "cams_pm10": ("particulate matter below 10 µm, at the surface", "kg m⁻³"),
    **{f"cams_{g}": (f"{n} mass mixing ratio at the lowest model level", "kg kg⁻¹") for g, n in
       [("no2", "NO₂"), ("o3", "O₃"), ("co", "CO"), ("so2", "SO₂"), ("hcho", "formaldehyde")]},
}


def variables(dataset: str) -> dict[str, tuple[str, str]]:
    """column -> (name, unit), from the dataset README's single-column table rows, then NAMES."""
    rows = [ROW.match(line) for line in (READMES / f"{dataset}.md").read_text().splitlines()]
    return {**{m[1]: (m[2], m[3]) for m in rows if m}, **NAMES}


def nice(lo: float, hi: float, n: int = 5) -> list[float]:
    """About n round ticks between lo and hi (1, 2, 5 x 10^k steps)."""
    raw = (hi - lo) / max(n - 1, 1)
    k = 10 ** math.floor(math.log10(raw))
    step = min((s * k for s in (1, 2, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(lo / step) * step
    return [float(f"{first + i * step:.6g}") for i in range(int((hi - first) // step) + 1)]


def log_ticks(lo: float, hi: float) -> list[float]:
    """1, 2, 5 x 10^k between lo and hi."""
    out = [m * 10 ** k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1) for m in (1, 2, 5)]
    out = [float(f"{t:.6g}") for t in out if lo <= t <= hi]
    return out if len(out) <= 7 else out[::2]


def scale(values: np.ndarray, unit: str) -> dict:
    v = values[np.isfinite(values)]
    lo, hi = (float(f"{x:.3g}") for x in np.percentile(v, [2, 98]))
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    if unit in ("K", "°C"):                                 # temperatures: cold to hot, evenly over the range
        return {"type": "linear", "min": lo, "max": hi, "ramp": "coldHot", "ticks": nice(lo, hi)}
    if lo < 0 < hi:
        m = max(-lo, hi)
        return {"type": "centred", "min": -m, "centre": 0, "max": m, "ramp": "diverging", "ticks": nice(-m, m)}
    if lo >= 0 and hi > 0 and float(np.median(v)) < 0.15 * hi:
        positive = v[v > 0]
        floor = max(float(np.percentile(positive, 2)) if len(positive) else hi / 1000, hi / 1000)
        floor = float(f"{floor:.2g}")
        return {"type": "log", "min": floor, "max": hi, "ramp": "magnitude", "ticks": log_ticks(floor, hi)}
    return {"type": "linear", "min": lo, "max": hi, "ramp": "magnitude", "ticks": nice(lo, hi)}


def main(folder: Path) -> None:
    dataset = folder.name                                                     # e.g. CPC-Monthly
    months = [pq.read_table(folder / "climatology_overviews" / f"{m:02d}.parquet") for m in range(1, 13)]
    columns = [n.removesuffix("_mean") for n in months[0].column_names
               if n.endswith("_mean") and not n.endswith("_valid_mean")]
    names = variables(dataset)
    missing = [c for c in columns if c not in names]
    assert not missing, f"no name or unit for {missing}: add them to {dataset}.md or NAMES"
    out = {}
    for column in columns:
        name, unit = names[column]
        values = np.concatenate([t.column(f"{column}_mean").to_numpy(zero_copy_only=False) for t in months])
        out[column] = {"name": name, "unit": unit, "scale": scale(values.astype(np.float64), unit)}
    overview = next(folder.glob("overviews/year=*/*.parquet"))
    with_overview = [n for n in pq.read_schema(overview).names if n in out]
    path = APP / "scales" / f"{dataset}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"overviews": with_overview, "variables": out}, indent=1, ensure_ascii=False) + "\n")
    kinds = {}
    for v in out.values():
        kinds[v["scale"]["type"]] = kinds.get(v["scale"]["type"], 0) + 1
    print(f"{len(out)} variables -> {path}: {kinds}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
