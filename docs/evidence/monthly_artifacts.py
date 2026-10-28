"""How many implausible source values do CPC-Monthly and IMERG-Monthly carry? (for the dataset READMEs; no downloads)

    CPC:   raw daily files (<cache>/cpc/precip.YYYY.nc, 0.5°) -> per pixel and month, days with >= 200 mm
             -> pixel-months with >= 20 such days (the pattern of Feb 1991 in central Mexico: ~370 mm every day)
             -> list with place, month, monthly total -> <out>/cpc_suspect.csv
           CPC-Monthly series -> months with any cell above 4,000 mm -> <out>/cpc_big_months.csv
             (found 2026-10-28: 6 months, 437 cell-months, all at places and seasons where this cannot happen)
    IMERG: IMERG-Monthly series -> cells beyond 55° latitude with a monthly mean above 1 mm/h (~740 mm a month)
             -> counts by year, largest values -> <out>/imerg_suspect.csv

    python docs/evidence/monthly_artifacts.py /data/Major-TOM <out folder>      (Slurm CPU job)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import n_rows                       # noqa: E402
from majortom.climate.readers import band_times         # noqa: E402

HEAVY_MM, HEAVY_DAYS = 200.0, 20                        # CPC: a day of >= 200 mm on >= 20 days of one month
HIGH_LAT, HIGH_MM_H = 55.0, 1.0                         # IMERG: beyond 55° and above 1 mm/h as a monthly mean
BIG_MM = 4000.0                                         # CPC: monthly totals above this (none are real outside
                                                        # Meghalaya and the Western Ghats, and none occur there)


def cpc(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "Core-Weather/cache/cpc").glob("precip.*.nc")):
        times = pd.DatetimeIndex(band_times(str(path), "precip")[0])
        with rasterio.open(f'netcdf:"{path}":precip') as src:
            data = src.read().astype(np.float32)        # days x lat x lon, ~0.4 GB per year
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            transform = src.transform
        for month in range(1, 13):
            days = data[times.month == month]
            heavy = (days >= HEAVY_MM).sum(axis=0)
            for r, c in zip(*np.nonzero(heavy >= HEAVY_DAYS)):
                lon, lat = transform * (c + 0.5, r + 0.5)
                rows.append(dict(month=f"{times[0].year}-{month:02d}", lat=round(lat, 2),
                                 lon=round((lon + 180) % 360 - 180, 2), heavy_days=int(heavy[r, c]),
                                 total_mm=round(float(np.nansum(days[:, r, c])), 1)))
        print(path.name, len(rows), flush=True)
    return pd.DataFrame(rows)


def cpc_totals(root: Path) -> pd.DataFrame:
    """CPC-Monthly months with any cell above BIG_MM: cells, largest total, median place."""
    from majortom.cells import cell_from_name
    from majortom.climate.layout import centroids

    rows = []
    for f in sorted((root / "CPC-Monthly/series").glob("year=*/*.parquet")):
        t = pq.read_table(f, columns=["grid_cell", "cpc_precip"]).to_pandas()
        big = t[t.cpc_precip > BIG_MM]
        if len(big):
            rc = np.array([(c.row, c.col) for c in map(cell_from_name, big.grid_cell)])
            lon, lat = centroids(rc[:, 0], rc[:, 1])
            rows.append(dict(month=f"{f.parent.name[5:]}-{f.stem}", cells=len(big), max_mm=round(big.cpc_precip.max()),
                             lat=round(float(np.median(lat)), 1), lon=round(float(np.median(lon)), 1)))
    return pd.DataFrame(rows)


def imerg(root: Path) -> pd.DataFrame:
    rows = []
    dlat = 180 / n_rows(10.0)
    for f in sorted((root / "IMERG-Monthly/series").glob("year=*/*.parquet")):
        t = pq.read_table(f, columns=["grid_cell", "imerg_precip"]).to_pandas()
        row = t.grid_cell.str.extract(r"^(\d+)([UD])")
        lat = row[0].astype(int) * dlat * np.where(row[1] == "U", 1, -1)
        hit = t[(lat.abs() > HIGH_LAT) & (t.imerg_precip > HIGH_MM_H)]
        rows += [dict(month=f"{f.parent.name[5:]}-{f.stem}", grid_cell=g, mm_h=round(float(v), 3))
                 for g, v in zip(hit.grid_cell, hit.imerg_precip)]
    return pd.DataFrame(rows, columns=["month", "grid_cell", "mm_h"])


def main(root: str, out: str) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c = cpc(root)
    c.to_csv(out / "cpc_suspect.csv", index=False)
    print(f"\nCPC: {len(c)} pixel-months with >= {HEAVY_DAYS} days of >= {HEAVY_MM:.0f} mm")
    if len(c):
        print(c.sort_values("total_mm", ascending=False).to_string(index=False))
    t = cpc_totals(root)
    t.to_csv(out / "cpc_big_months.csv", index=False)
    print(f"\nCPC: {len(t)} months with cells above {BIG_MM:.0f} mm, {t.cells.sum() if len(t) else 0} cell-months")
    if len(t):
        print(t.sort_values("max_mm", ascending=False).to_string(index=False))
    i = imerg(root)
    i.to_csv(out / "imerg_suspect.csv", index=False)
    print(f"\nIMERG: {len(i)} cell-months beyond {HIGH_LAT:.0f}° above {HIGH_MM_H} mm/h, in "
          f"{i.month.nunique()} months")
    if len(i):
        print(i.groupby(i.month.str[:4]).size().rename("cell-months").to_string())
        print(i.nlargest(10, "mm_h").to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
