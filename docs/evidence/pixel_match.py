"""Compare a Major TOM band with the same band read natively from the raw S2 COG.

Finds the integer pixel offset in the native grid where the arrays match exactly,
and compares that with the offset implied by the Major TOM geotransform.
"""
import json
import sys

import numpy as np
import pyarrow.parquet as pq
import rasterio
from fsspec.parquet import open_parquet_file
from rasterio.io import MemoryFile
from rasterio.windows import Window

CELL, COG, BAND, RES, JSONL = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
rec = next(json.loads(l) for l in open(JSONL) if json.loads(l)["grid_cell"] == CELL)

with open_parquet_file(rec["parquet_url"], columns=[BAND], row_groups=[rec["parquet_row"]]) as f:
    tab = pq.ParquetFile(f).read_row_group(rec["parquet_row"], columns=[BAND])
with MemoryFile(tab[BAND][0].as_py()) as m, m.open() as src:
    mt, mt_tr = src.read(1), src.transform
n = mt.shape[0]

with rasterio.open(COG) as cog:
    col_f = (mt_tr.c - cog.transform.c) / RES
    row_f = (cog.transform.f - mt_tr.f) / RES
    print(f"native UL=({cog.transform.c},{cog.transform.f}) MT UL=({mt_tr.c:.3f},{mt_tr.f:.3f})")
    print(f"MT origin in native pixel coords: col={col_f:.4f} row={row_f:.4f}")
    pad = 3
    c0, r0 = int(np.floor(col_f)) - pad, int(np.floor(row_f)) - pad
    nat = cog.read(1, window=Window(c0, r0, n + 2 * pad, n + 2 * pad), boundless=True, fill_value=0)

best = []
for dr in range(2 * pad + 1):
    for dc in range(2 * pad + 1):
        sub = nat[dr:dr + n, dc:dc + n]
        eq = (sub == mt).mean()
        best.append((eq, r0 + dr, c0 + dc))
best.sort(reverse=True)
for eq, r, c in best[:3]:
    print(f"native offset row={r} col={c}: exact-equal fraction={eq:.5f}")
print(f"round(col_f)={round(col_f)} round(row_f)={round(row_f)}  floor=({int(np.floor(row_f))},{int(np.floor(col_f))})")

print("MT stats", mt.min(), mt.max(), mt.mean(), "zeros", (mt == 0).mean())
sub = nat[pad:pad + n, pad:pad + n]
print("native stats", sub.min(), sub.max(), sub.mean(), "zeros", (sub == 0).mean())
cors = []
for dr in range(2 * pad + 1):
    for dc in range(2 * pad + 1):
        s = nat[dr:dr + n, dc:dc + n].astype(float)
        cors.append((np.corrcoef(s.ravel(), mt.astype(float).ravel())[0, 1], r0 + dr, c0 + dc))
cors.sort(reverse=True)
print("top correlations:", [(round(c, 4), r, cc) for c, r, cc in cors[:3]])
c, r, cc = cors[0]
s = nat[r - r0:r - r0 + n, cc - c0:cc - c0 + n].astype(int)
d = mt.astype(int) - s
print("diff at best shift: median", np.median(d), "mean", d.mean(), "exact-equal", (d == 0).mean(),
      "equal after const offset", (d == np.median(d)).mean())
