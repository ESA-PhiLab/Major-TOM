"""Colour scales for the ERA5 layer: one per variable, from its normals, written to scales.json.

    docs/datasets/climate/ERA5-Monthly.md (variables table: column, name, units)
    + <dataset>/climatology_overviews/MM.parquet (block means of each <column>_mean, 12 calendar months)
      -> per variable: 2nd and 98th percentile of the block normals over the 12 months
      -> scale: K: linear, cold-hot; signed: centred on 0 (diverging);
         positive and skewed (median below 15% of the top): log; otherwise linear (sequential)
      -> scales.json: {column: {name, unit, scale: {type, min, max, centre?, ramp, ticks}}}, in the README's order;
         "overviews": the columns with series overviews (globe-scale monthly maps)

    python build_scales.py /data/Major-TOM/climate-release/ERA5-Monthly        (Slurm, .venv311 or miko-torch)
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
README = APP.parents[1] / "docs" / "datasets" / "climate" / "ERA5-Monthly.md"
ROW = re.compile(r"^\| `(era5_\w+)` \| (.+?) \| (.*?) \|$")


def variables() -> list[tuple[str, str, str]]:
    """(column, name, unit) of every row of the README's variables table."""
    rows = [ROW.match(line) for line in README.read_text().splitlines()]
    return [(m[1], m[2], m[3]) for m in rows if m]


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
    if unit == "K":                                         # temperatures: cold to hot, evenly over the range
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


def main(dataset: str) -> None:
    folder = Path(dataset)
    months = [pq.read_table(folder / "climatology_overviews" / f"{m:02d}.parquet") for m in range(1, 13)]
    out = {}
    for column, name, unit in variables():
        values = np.concatenate([t.column(f"{column}_mean").to_numpy(zero_copy_only=False) for t in months])
        out[column] = {"name": name, "unit": unit, "scale": scale(values.astype(np.float64), unit)}
    overview = next(folder.glob("overviews/year=*/*.parquet"))
    columns = [n for n in pq.read_schema(overview).names if n.startswith("era5_")]
    path = APP / "scales.json"
    path.write_text(json.dumps({"overviews": columns, "variables": out}, indent=1, ensure_ascii=False) + "\n")
    kinds = {}
    for v in out.values():
        kinds[v["scale"]["type"]] = kinds.get(v["scale"]["type"], 0) + 1
    print(f"{len(out)} variables -> {path}: {kinds}")


if __name__ == "__main__":
    main(sys.argv[1])
