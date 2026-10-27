"""ERA5-Land group 1 from Earth Data Hub against the Earth Engine build, on the acquisition days both wrote.

    <edh>/hourly|daily/year=YYYY/<day>.parquet  and  <ee>/hourly|daily/...   (same acquisitions, same columns)
      -> for every day both have: join on grid_cell, acquired, step
      -> per column: max |difference|, max relative difference (to the column's largest |value|), rows compared,
         rows where only one side has a value
      -> also: days written by the Earth Data Hub build against the acquisition days expected
    Expected agreement: within ~4e-4 relative (Earth Data Hub's 13-bit rounding; docs/CLIMATE.md 6.1).

    python docs/evidence/era5land_edh_validate.py /data/Major-TOM/Core-Weather/era5land-edh-g1 /data/Major-TOM/Core-Weather/era5land
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(edh: Path, ee: Path, kind: str, step: str) -> pd.DataFrame:
    both = sorted({p.relative_to(edh / kind) for p in (edh / kind).rglob("*.parquet")} &
                  {p.relative_to(ee / kind) for p in (ee / kind).rglob("*.parquet")})
    stats: dict[str, list] = {}
    for rel in both:
        a = pd.read_parquet(edh / kind / rel)
        b = pd.read_parquet(ee / kind / rel)
        j = a.merge(b, on=["grid_cell", "acquired", step], suffixes=("", "_ee"))
        for col in [c for c in a.columns if c.startswith("era5land_") and f"{c}_ee" in j.columns]:
            x, y = j[col].to_numpy(float), j[f"{col}_ee"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            one = np.isfinite(x) ^ np.isfinite(y)
            s = stats.setdefault(col, [0.0, 0.0, 0, 0])
            if ok.any():
                s[0] = max(s[0], float(np.abs(x[ok] - y[ok]).max()))
                s[1] = max(s[1], float(np.abs(y[ok]).max()))
            s[2] += int(ok.sum())
            s[3] += int(one.sum())
    out = pd.DataFrame(stats, index=["max_abs_diff", "max_abs_value", "rows", "one_side_only"]).T
    out["max_rel_diff"] = out.max_abs_diff / out.max_abs_value.replace(0, np.nan)
    print(f"\n{kind}: {len(both)} days in both builds")
    print(out.to_string(float_format=lambda v: f"{v:.4g}"))
    return out


def main(edh: str, ee: str) -> None:
    edh, ee = Path(edh), Path(ee)
    written = sorted((edh / "daily").rglob("*.parquet"))
    acq = pd.read_parquet(edh / "acquisitions.parquet") if (edh / "acquisitions.parquet").exists() else None
    if acq is not None:
        days = pd.to_datetime(acq["acquired"]).dt.normalize().nunique()
        print(f"Earth Data Hub build: {len(written)} daily files written, {days} acquisition days expected")
    for kind, step in (("hourly", "step_h"), ("daily", "step_d")):
        result = compare(edh, ee, kind, step)
        worst = result["max_rel_diff"].max()
        print(f"{kind}: worst relative difference {worst:.2e} -> {'OK' if worst < 1e-3 else 'CHECK'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
