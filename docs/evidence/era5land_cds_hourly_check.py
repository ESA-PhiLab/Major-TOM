"""era5-land-cds-south checked end to end where both ERA5-Land copies have land: Loch Lomond (624U_29L), its first
Core-S2L2A acquisition, through stream(); compared with Earth Data Hub's Core-Weather build (era5land-edh-g1).

    era5-land-cds-south with a small area and grid around the loch (3 variables: t2m, tp, ssrd) -> one request
      (one week) -> hourly (±3 h) and daily (step 0) rows -> same rows of era5land-edh-g1 -> differences per column
    Result 2026-11-02: t2m within 2.7e-4 (0.074 K), ssrd 8.2e-5, tp equal, valid shares identical.

    python docs/evidence/era5land_cds_hourly_check.py <scratch folder>          (Slurm; one CDS request of 504 fields)
"""
import dataclasses, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.environ["MAJORTOM_CLIMATE_CACHE"] = sys.argv[1] + "/cache"
from majortom.climate.sources import SOURCES
from majortom.climate.weights import Grid
from majortom.climate.stream import stream

base = SOURCES["era5-land-cds-south"]
SOURCES["era5-land-cds-test"] = dataclasses.replace(base, grid=Grid(west=-5.05, north=56.55, step=0.1, width=9, height=8),
    options={**base.options, "area": (56.5, -5.0, 55.8, -4.2), "groups": {1: ["t2m", "tp", "ssrd"]}})
meta = pd.read_parquet("/data/Major-TOM/Core-S2L2A/metadata.parquet", columns=["grid_cell", "grid_row_u", "grid_col_r", "timestamp", "product_id"])
sample = meta[meta.grid_cell == "624U_29L"].head(1).assign(dataset="Core-S2L2A")
out = Path(sys.argv[1]) / "loch"
print(stream(sample, out, "era5-land-cds-test", ["t2m", "tp", "ssrd"], hourly=3, daily=0, workers=4), "fields read")
for kind, step in [("hourly", "step_h"), ("daily", "step_d")]:
    for p in sorted((out / kind).glob("year=*/*.parquet")):
        new = pd.read_parquet(p)
        old = pd.read_parquet(Path("/data/Major-TOM/Core-Weather/era5land-edh-g1") / kind / p.parent.name / p.name)
        m = new.merge(old, on=["grid_cell", "acquired", step], suffixes=("_cds", "_edh"))
        print(kind, p.stem, len(m), "rows; valid", m.era5land_valid_cds.tolist()[:3], m.era5land_valid_edh.tolist()[:3])
        for c in [c for c in new.columns if c.startswith("era5land_") and c != "era5land_valid"]:
            a, b = m[f"{c}_cds"].to_numpy(float), m[f"{c}_edh"].to_numpy(float)
            print(f"   {c}: cds {np.round(a, 6).tolist()}\n   {' ' * len(c)}  edh {np.round(b, 6).tolist()}\n   max |diff| / max |edh| {np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(b)):.2g}")
