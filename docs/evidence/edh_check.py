"""Earth Data Hub ERA5-Land hourly: layout, request count, speed, and a cross-check against the Earth Engine build.

    store (zarr v3, key as basic auth; every HTTP request counted by an aiohttp trace)
      -> 1. layout: variables, t2m's shape, chunks and shards, coordinate ends
      -> 2. one chunk (1440 h x 50 x 100) of t2m and of tp around a Core cell: requests, MB, seconds
      -> 3. cell means with the build's weights vs the Earth Engine build's hourly values for the same
            acquisitions (best hour offset, max difference); consecutive hours must differ

    python docs/evidence/edh_check.py /data/Major-TOM/Core-Weather/era5land 2016-03-15      (Slurm CPU job)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import aiohttp
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.build.credentials import EDH_STORE, edh_key          # noqa: E402
from majortom.climate.weights import Grid, cell_means, weights      # noqa: E402

GRID = Grid(west=-0.05, north=90.05, step=0.1, width=3600, height=1472)   # 90°N .. 57.1°S, 0 .. 359.9°E
CHUNK_T, CHUNK_Y, CHUNK_X = 1440, 50, 100
counts = {"requests": 0, "bytes": 0}


def traced() -> aiohttp.TraceConfig:
    """Count every HTTP request and the bytes each response declares."""
    trace = aiohttp.TraceConfig()

    async def start(session, ctx, params):
        counts["requests"] += 1

    async def end(session, ctx, params):
        counts["bytes"] += params.response.content_length or 0

    trace.on_request_start.append(start)
    trace.on_request_end.append(end)
    return trace


def open_store() -> xr.Dataset:
    fs = fsspec.filesystem("https", client_kwargs={"auth": aiohttp.BasicAuth("edh", edh_key()),
                                                   "trace_configs": [traced()]}, asynchronous=True)
    store = zarr.storage.FsspecStore(fs, path=EDH_STORE, read_only=True)
    return xr.open_zarr(store, chunks=None, consolidated=True)


def rowcol(names: pd.Series) -> pd.DataFrame:
    """grid_row_u and grid_col_r (signed) from names such as 245D_1567R."""
    p = names.str.extract(r"(\d+)([UD])_(\d+)([LR])")
    return pd.DataFrame({"grid_cell": names.to_numpy(),
                         "grid_row_u": p[0].astype(int).to_numpy() * np.where(p[1] == "U", 1, -1),
                         "grid_col_r": p[2].astype(int).to_numpy() * np.where(p[3] == "R", 1, -1)})


def main(ee_root: str, day: str) -> None:
    t = time.time()
    ds = open_store()
    print(f"1. opened in {time.time() - t:.1f} s, {counts['requests']} requests, {counts['bytes'] / 1e6:.2f} MB")
    print(f"   {len(ds.data_vars)} variables: {sorted(ds.data_vars)}")
    arr = ds["t2m"]
    print(f"   t2m dims {arr.dims} shape {arr.shape} encoding chunks {arr.encoding.get('chunks')} "
          f"shards {arr.encoding.get('shards')}")
    lat, lon, vt = ds.latitude.to_numpy(), ds.longitude.to_numpy(), ds.valid_time.to_numpy()
    print(f"   latitude {lat[0]} .. {lat[-1]} ({len(lat)}), longitude {lon[0]} .. {lon[-1]} ({len(lon)}), "
          f"time {vt[0]} .. {vt[-1]}; after coordinates: {counts['requests']} requests")
    assert len(lat) == GRID.height and abs(lat[0] - 90) < 1e-6 and abs(lon[0]) < 1e-6

    # the Earth Engine build's acquisitions on `day` with full data at step 0; the chunk around the first one
    ee = pd.read_parquet(Path(ee_root) / "hourly" / f"year={day[:4]}" / f"{day}.parquet")
    full = ee[(ee.step_h == 0) & (ee.era5land_valid == 1)]
    first = rowcol(full.grid_cell.iloc[:1])
    cell_lat = first.grid_row_u.iloc[0] * 180 / 2004 + 0.05
    w0 = weights(first, GRID)
    pix = w0.pixel[np.argmax(w0.weight)]
    y, x = divmod(int(pix), GRID.width)
    y0, x0 = y // CHUNK_Y * CHUNK_Y, x // CHUNK_X * CHUNK_X
    hour = np.datetime64(f"{day}T00", "h")
    i = int((hour - vt[0].astype("datetime64[h]")) / np.timedelta64(1, "h"))
    t0 = i // CHUNK_T * CHUNK_T
    print(f"2. chunk: time {vt[t0]} +1440 h, rows {y0}..{y0 + CHUNK_Y}, cols {x0}..{x0 + CHUNK_X} "
          f"(cell {first.grid_cell.iloc[0]}, ~{cell_lat:.1f}°)")
    blocks = {}
    for v in ("t2m", "tp"):
        r0, b0, t = counts["requests"], counts["bytes"], time.time()
        blocks[v] = ds[v].isel(valid_time=slice(t0, t0 + CHUNK_T), latitude=slice(y0, y0 + CHUNK_Y),
                               longitude=slice(x0, x0 + CHUNK_X)).to_numpy()
        dt = time.time() - t
        mb = (counts["bytes"] - b0) / 1e6
        print(f"   {v}: {counts['requests'] - r0} requests, {mb:.1f} MB in {dt:.1f} s ({mb / dt:.1f} MB/s), "
              f"NaN share {np.isnan(blocks[v]).mean():.2f}")
    same = [h for h in range(CHUNK_T - 1) if np.array_equal(blocks["t2m"][h], blocks["t2m"][h + 1], equal_nan=True)]
    print(f"   repeated t2m fields in consecutive hours: {len(same)} of {CHUNK_T - 1}")

    # 3. cell means for the acquisitions whose cells lie inside this chunk, vs the Earth Engine build
    cells = rowcol(full.grid_cell)
    w = weights(cells, GRID)
    inside = np.zeros(len(cells), bool)
    py, px = np.divmod(w.pixel, GRID.width)
    ok = (py >= y0) & (py < y0 + CHUNK_Y) & (px >= x0) & (px < x0 + CHUNK_X)
    inside[np.unique(w.cell[ok])] = True
    outside = np.unique(w.cell[~ok])
    inside[outside] = False                                              # cells straddling the chunk edge drop out
    picked = cells[inside].grid_cell.unique()[:5]
    print(f"3. {len(picked)} acquisitions on {day} with cells inside the chunk: {list(picked)}")
    field = np.full((GRID.height, GRID.width), np.nan, np.float32)
    for v, ee_col in (("t2m", "era5land_t2m"), ("tp", "era5land_tp")):
        worst = {}
        for name in picked:
            acq = ee[ee.grid_cell == name].sort_values("step_h")
            sub = rowcol(pd.Series([name]))
            ws = weights(sub, GRID)
            base = int((acq.acquired.iloc[0].floor("h").to_datetime64().astype("datetime64[h]")
                        - vt[t0].astype("datetime64[h]")) / np.timedelta64(1, "h"))
            series = []
            for step in acq.step_h:
                h = base + int(step)
                if not 0 <= h < CHUNK_T:
                    series.append(np.nan)
                    continue
                field[y0:y0 + CHUNK_Y, x0:x0 + CHUNK_X] = blocks[v][h]
                series.append(cell_means(ws, field)[0][0])
            series, ref = np.array(series), acq[ee_col].to_numpy()
            best = min(range(-2, 3), key=lambda k: np.nanmax(np.abs(np.roll(series, k) - ref)[2:-2]))
            worst[name] = (best, float(np.nanmax(np.abs(np.roll(series, best) - ref)[2:-2])))
            if v == "tp" and name == picked[0]:
                print("   tp, first acquisition, steps -3..+3: EDH", np.round(series[21:28] * 1000, 3),
                      "| Earth Engine", np.round(ref[21:28] * 1000, 3), "(mm)")
        print(f"   {v}: best hour offset and max |EDH - Earth Engine| per acquisition: {worst}")
    print(f"total: {counts['requests']} requests, {counts['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
