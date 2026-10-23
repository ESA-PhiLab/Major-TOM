"""Which 24 hours a NOAA CPC daily precipitation value covers: tested against ERA5 hourly precipitation.

    cells near Sydney, Brisbane, Denver (far from UTC) and Glasgow (control); January and July 2021
      -> CPC daily precipitation per cell (labelled by its start date, 00:00)
      -> ERA5 hourly precipitation per cell, summed over 24 h windows starting at 00:00 UTC + s hours, s = -14..14
      -> correlation per shift s: the best s is where CPC's day starts (0: UTC days; -10 near Sydney: local days)

    python -u docs/evidence/cpc_day.py
First run, 2026-10-23: best starts between -13 h and +1 h (mostly -4 to -10 h); +12 h worst everywhere; -12 h beats
0 h in 6 of 8 place-months (Denver Jan: r 0.68 against 0.36). Not local days (Denver would be +7 h). Most consistent
with a 12Z-to-12Z gauge day ending on the labelled date; the peak is broad (one cell, ~30 days per test).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import cell_at                                         # noqa: E402
from majortom.climate import cell_means, weights                            # noqa: E402
from majortom.climate.sources import SOURCES, field                         # noqa: E402

PLACES = {"Sydney (UTC+11 in Jan)": (151.0, -33.8), "Brisbane (UTC+10)": (153.0, -27.5),
          "Denver (UTC-7)": (-104.9, 39.7), "Glasgow (UTC+0/+1)": (-4.25, 55.86)}
MONTHS = ["2021-01", "2021-07"]
SHIFTS = range(-14, 15)


def table(cells) -> pd.DataFrame:
    return pd.DataFrame({"grid_row_u": [c.row for c in cells], "grid_col_r": [c.col for c in cells]})


def main() -> None:
    cells = [cell_at(*p) for p in PLACES.values()]
    era5, cpc = SOURCES["era5"], SOURCES["cpc"]
    w_era5, w_cpc = weights(table(cells), era5.grid), weights(table(cells), cpc.grid)
    for month in MONTHS:
        days = pd.date_range(month, periods=pd.Period(month).days_in_month, freq="D")
        hours = pd.date_range(days[0] - pd.Timedelta(hours=15), days[-1] + pd.Timedelta(hours=40), freq="h")
        with ThreadPoolExecutor(32) as pool:
            hourly = np.array(list(pool.map(lambda h: cell_means(w_era5, field(era5, "tp", h))[0], hours)))
        hourly = pd.DataFrame(hourly, index=hours)                   # hour stamp H = total over the hour ending at H
        gauge = np.array([cell_means(w_cpc, field(cpc, "precip", (int(np.datetime64(d.date()).astype(int)), 0)))[0]
                          for d in days])
        print(f"{month}:")
        for i, name in enumerate(PLACES):
            scores = {}
            for s in SHIFTS:
                start = days + pd.Timedelta(hours=s)
                sums = [hourly.loc[t + pd.Timedelta(hours=1):t + pd.Timedelta(hours=24), i].sum() for t in start]
                scores[s] = np.corrcoef(gauge[:, i], sums)[0, 1]
            best = max(scores, key=lambda s: np.nan_to_num(scores[s], nan=-1))
            print(f"   {name:24s} best start {best:+d} h (r = {scores[best]:.2f}); at 0 h r = {scores[0]:.2f}, "
                  f"at -12 h r = {scores[-12]:.2f}, at +12 h r = {scores[12]:.2f}")


if __name__ == "__main__":
    main()
