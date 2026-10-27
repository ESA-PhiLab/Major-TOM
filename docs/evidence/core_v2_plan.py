"""Which Core v1 samples can become v2 samples by cutting, and which must be read again (Core-S2L2A, Core-S2L1C).

    Core metadata (grid cell, crs, product_id) -> per sample, from geometry alone (no pixels read):
      zone:  is the sample's CRS the cell's own (cell_crs)? v2 windows are defined in the cell's zone
      fits:  is the v2 window inside the v1 window at 10, 20 and 60 m (Core v1 corner rounded per band grid)?
      -> "cut"           zone ok and fits: crop_core_to_v2 gives the v2 pixels exactly, no download
         "read, zone ok"  the v2 window sticks out of the v1 sample: read from the archive
         "read, other zone" v1 came from a product in another UTM zone: read in the cell's zone
      -> counts overall, by latitude band and by processing baseline (what a re-read would get)
    Vectorised per CRS; checked against majortom.spec (legacy_window, window, crop_core_to_v2) on random samples.

    python docs/evidence/core_v2_plan.py Core-S2L2A <out.parquet>       (Slurm CPU job)
"""
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import Cell, n_rows                                   # noqa: E402
from majortom.spec import MARGIN, cell_crs, crop_core_to_v2, default_origin, legacy_window, window   # noqa: E402

V1_PAD, V1_SIDE = 340.0, 10_680.0


def geometry(meta: pd.DataFrame) -> pd.DataFrame:
    rows = meta.grid_row_u.to_numpy()
    cols = meta.grid_col_r.to_numpy()
    dlat = 180 / n_rows(10.0)
    lat0 = rows * dlat
    ncols = np.maximum(np.ceil(2 * np.pi * 6378.137 * np.cos(np.radians(lat0)) / 10), 1)
    dlon = 360 / ncols
    lon0 = cols * dlon
    out = pd.DataFrame(index=meta.index)
    out["fits"] = False
    side = 2 * 60 * np.ceil((1 + MARGIN) * 10_000 / 120 - 1e-9)                     # v2 side at 10 km, 60 m lattice
    for crs, idx in meta.groupby("crs").groups.items():
        i = np.asarray([meta.index.get_loc(k) for k in idx])
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x0, y0 = default_origin(crs, 60.0)
        # v1: grid point (SW corner) - 340 m, side 10,680 m; each band's corner rounded to its own grid
        sx, sy = tr.transform(lon0[i], lat0[i])
        left1, top1 = sx - V1_PAD, sy - V1_PAD + V1_SIDE
        # v2: centroid snapped to the 60 m lattice
        cx, cy = tr.transform(lon0[i] + dlon[i] / 2, lat0[i] + dlat / 2)
        cx = x0 + 60 * np.round((cx - x0) / 60)
        cy = y0 + 60 * np.round((cy - y0) / 60)
        left2, top2 = cx - side / 2, cy + side / 2
        ok = np.ones(len(i), bool)
        for res, n1 in ((10, 1068), (20, 534), (60, 178)):
            c1 = x0 + res * np.round((left1 - x0) / res)
            t1 = y0 + res * np.round((top1 - y0) / res)
            col = np.round((left2 - c1) / res)
            row = np.round((t1 - top2) / res)
            n2 = round(side / res)
            ok &= (col >= 0) & (row >= 0) & (col + n2 <= n1) & (row + n2 <= n1)
        out.iloc[i, out.columns.get_loc("fits")] = ok
    out["zone_ok"] = [cell_crs(Cell(int(r), int(c))) == crs for r, c, crs in zip(rows, cols, meta.crs)]
    out["plan"] = np.where(~out.zone_ok, "read, other zone", np.where(out.fits, "cut", "read, zone ok"))
    return out


def verify(meta: pd.DataFrame, g: pd.DataFrame, n: int = 300) -> None:
    """The vectorised answer equals crop_core_to_v2 on the spec's own windows, for random samples in the cell's zone."""
    pick = g[g.zone_ok].sample(n, random_state=0).index
    bad = 0
    for k in pick:
        r = meta.loc[k]
        cell = Cell(int(r.grid_row_u), int(r.grid_col_r))
        fits = True
        for res, n1 in ((10, 1068), (20, 534), (60, 178)):
            t = legacy_window(cell, r.crs).transform(res)
            fits &= crop_core_to_v2(np.zeros((n1, n1)), t, r.crs, cell, "Core-S2L2A") is not None
        bad += fits != g.loc[k, "fits"]
    print(f"verified against majortom.spec on {n} samples: {n - bad} agree", flush=True)
    assert bad == 0


def main(dataset: str, out: str) -> None:
    with fsspec.open(f"https://huggingface.co/datasets/Major-TOM/{dataset}/resolve/main/metadata.parquet").open() as f:
        meta = pq.read_table(f, columns=["grid_cell", "grid_row_u", "grid_col_r", "product_id", "crs",
                                         "centre_lat", "nodata"]).to_pandas()
    g = geometry(meta)
    verify(meta, g)
    meta = meta.join(g)
    meta["baseline"] = meta.product_id.str.split("_").str[3]
    meta.drop(columns=["centre_lat"]).to_parquet(out, index=False)
    n = len(meta)
    print(f"\n{dataset}: {n:,} samples")
    print((meta.plan.value_counts() / n * 100).round(1).astype(str).add(" %").to_string())
    band = pd.cut(meta.centre_lat.abs(), [0, 30, 50, 60, 70, 90], labels=["0-30", "30-50", "50-60", "60-70", "70-90"])
    print("\nshare cut, by |latitude|:")
    print((meta.plan.eq("cut").groupby(band, observed=True).mean() * 100).round(1).to_string())
    print("\nre-reads by baseline (what the archive now serves may be a reprocessing):")
    print(meta[meta.plan != "cut"].baseline.value_counts().head(10).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
