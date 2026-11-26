"""Check the ERA5-Land south fill (docs/CLIMATE.md 6.1) in the merged Core-Weather folder before re-uploading it.

    merged/{hourly,daily}/year=*/<day>.parquet  + .publish-sums.json -> files changed since publishing (upload list)
    merged/daily (every day) -> rows south of 57.1°S, with ERA5-Land values (era5land_valid > 0) or without, per year
    era5land-cds-south-YYYY-g1/daily -> which of those rows the fill supplied (before = after - filled)
    filled rows -> ERA5-Land minus ERA5 2 m temperature; ranges and empty shares of the 52 era5land_* columns
    published copies on source.coop (HTTPS, column subset) -> spot check: south ERA5-Land empty before the fill
    -> <out>/south_fill.json, by_year.csv, variables.csv, cells.csv, series.csv, rows.csv

    python docs/evidence/south_fill_check.py <out folder>        (Slurm CPU job)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import cell_from_name                                    # noqa: E402

CW = Path("/data/Major-TOM/Core-Weather")
MERGED = CW / "merged"
PUBLISHED = "https://data.source.coop/major-tom/core-weather"
SOUTH = -57.1                                                                # Earth Data Hub's grid ends here
KEYS = ["grid_cell", "acquired", "step_d"]
LAND = [c for c in pq.read_schema(next(MERGED.glob("daily/year=2022/*.parquet"))).names if c.startswith("era5land_")]


@lru_cache(maxsize=None)
def centre(name: str) -> tuple[float, float]:
    """(lon, lat) of a cell's centre."""
    return cell_from_name(name).centroid


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def changed_files() -> pd.DataFrame:
    """Local day files that differ from the published ones: size or mtime first, then sha256 to confirm."""
    sums = json.loads((MERGED / ".publish-sums.json").read_text())
    rows = []
    for p in sorted(MERGED.glob("*/year=*/*.parquet")):
        key, st = str(p.relative_to(MERGED)), p.stat()
        old = sums.get(key)
        if old and old["size"] == st.st_size and abs(old["mtime"] - st.st_mtime) < 1:
            continue
        if old is None or sha256(p) != old["sha256"]:
            rows.append({"key": key, "kind": key.split("/")[0], "year": year_of(key),
                         "bytes": st.st_size, "new": old is None})
    return pd.DataFrame(rows)


def year_of(key: str) -> int:
    """'daily/year=2016/2016-02-02.parquet' -> 2016."""
    return int(key.split("year=")[1][:4])


def fill_rows(key: str) -> pd.DataFrame:
    """Rows the fill gave values on one day (era5land_valid > 0), from the group-1 fill folder."""
    f = CW / f"era5land-cds-south-{year_of(key)}-g1" / key
    if not f.exists():
        return pd.DataFrame(columns=KEYS)
    t = pd.read_parquet(f, columns=KEYS + ["era5land_valid"])
    return t[t.era5land_valid > 0][KEYS]


def scan_day(key: str) -> dict:
    """One merged daily file: its south rows, which carry ERA5-Land, and which of those came from the fill."""
    t = pd.read_parquet(MERGED / key, columns=KEYS + LAND + ["era5_t2m_mean"])
    lat = t.grid_cell.map(lambda n: centre(n)[1])
    t = t[lat.to_numpy() < SOUTH]
    supplied = fill_rows(key)
    if supplied.empty:                                          # no fill on this day (or year)
        filled = np.zeros(len(t), bool)
    else:
        filled = t.merge(supplied.assign(_f=True), on=KEYS, how="left")._f.fillna(False).to_numpy(bool)
    has = (t.era5land_valid > 0).to_numpy()                     # share of the cell with values; 0 at sea
    return {"key": key, "year": year_of(key), "south": len(t), "with_land": int(has.sum()),
            "filled": int((filled & has).sum()), "fill_not_applied": int((filled & ~has).sum()),
            "rows": t[filled & has]}


def spot_check(keys: list[str]) -> list[dict]:
    """Published copies (before) against local ones (after): south rows with era5land_valid, read over HTTPS."""
    out = []
    for key in keys:
        cols = ["grid_cell", "era5land_valid"]
        before = pd.read_parquet(f"{PUBLISHED}/{key}", columns=cols)
        after = pd.read_parquet(MERGED / key, columns=cols)
        b = before[before.grid_cell.map(lambda n: centre(n)[1]) < SOUTH]
        a = after[after.grid_cell.map(lambda n: centre(n)[1]) < SOUTH]
        out.append({"key": key, "south_rows": len(a), "with_land_before": int((b.era5land_valid > 0).sum()),
                    "with_land_after": int((a.era5land_valid > 0).sum())})
    return out


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    changed = changed_files()
    changed.to_csv(out / "changed.csv", index=False)
    days = sorted(str(p.relative_to(MERGED)) for p in MERGED.glob("daily/year=*/*.parquet"))
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        scans = list(pool.map(scan_day, days, chunksize=8))
    rows = pd.concat([s.pop("rows") for s in scans], ignore_index=True)
    by_day = pd.DataFrame(scans)
    by_year = by_day.groupby("year")[["south", "with_land", "filled", "fill_not_applied"]].sum()
    by_year["with_land_before"] = by_year.with_land - by_year.filled
    by_year.to_csv(out / "by_year.csv")

    rows["lon"], rows["lat"] = zip(*rows.grid_cell.map(centre))
    rows["d_t2m"] = rows.era5land_t2m_mean - rows.era5_t2m_mean
    variables = pd.DataFrame({"empty_share": rows[LAND].isna().mean(), "min": rows[LAND].min(),
                              "median": rows[LAND].median(), "max": rows[LAND].max()})
    variables.to_csv(out / "variables.csv")
    cells = rows.groupby("grid_cell").agg(lon=("lon", "first"), lat=("lat", "first"), rows=("acquired", "size"),
                                          t2m=("era5land_t2m_mean", "mean"), d_t2m=("d_t2m", "mean"))
    cells.to_csv(out / "cells.csv")
    peninsula = cells[(cells.lon > -70) & (cells.lon < -55)].sort_values("rows").index[-1]
    rows[rows.grid_cell == peninsula].sort_values(["acquired", "step_d"])[
        KEYS + ["era5land_t2m_mean", "era5_t2m_mean", "era5land_sd_mean", "era5land_valid"]].to_csv(
        out / "series.csv", index=False)
    rows.sample(12, random_state=0)[KEYS + ["lon", "lat"] + LAND[:14] + ["era5_t2m_mean"]].to_csv(
        out / "rows.csv", index=False)

    d = rows.d_t2m.dropna()
    spot = spot_check([k for k in by_day.sort_values("filled").key.iloc[-3:]])
    summary = {"changed_files": len(changed), "changed_bytes": int(changed.bytes.sum()),
               "changed_by_kind": changed.groupby("kind").size().to_dict(),
               "changed_new_files": int(changed.new.sum()), "filled_rows_daily": len(rows),
               "filled_cells": int(rows.grid_cell.nunique()), "peninsula_cell": peninsula,
               "d_t2m": {"mean": d.mean(), "std": d.std(), "p05": d.quantile(.05), "p95": d.quantile(.95),
                         "corr": rows[["era5land_t2m_mean", "era5_t2m_mean"]].corr().iloc[0, 1], "n": len(d)},
               "fill_not_applied": int(by_year.fill_not_applied.sum()), "spot_check": spot}
    (out / "south_fill.json").write_text(json.dumps(summary, indent=1, default=float))
    print(by_year.to_string(), json.dumps(summary, indent=1, default=float), sep="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
