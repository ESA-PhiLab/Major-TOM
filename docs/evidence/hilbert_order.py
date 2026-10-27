"""The Hilbert writing order on Core-S2L2A, and the partitions taco-eo's size cut makes from it (docs/RELEASE.md 7).

    Core-S2L2A metadata (2,245,886 acquisitions) -> hilbert_order -> taco-eo's rule: a new partition before the
    next sample would pass max_bytes (3 GB), each sample 6.2 MB (lossless rumi, measured)
      -> partition sizes, cells per partition, area, cells split across two partitions
      -> docs/figures/hilbert_order.png: every cell coloured by its position in the order (light: early, dark: late)

    python docs/evidence/hilbert_order.py docs/figures            (Slurm CPU job)
"""
import sys
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import cell_from_name                                # noqa: E402
from majortom.release.order import hilbert_order                         # noqa: E402

BYTES, MAX = 6_200_000, 3 * 10**9
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

with fsspec.open("https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet").open() as f:
    m = pq.read_table(f, columns=["grid_cell", "timestamp"]).to_pandas()
s = pd.DataFrame({"grid_cell": m.grid_cell, "acquired": pd.to_datetime(m.timestamp, format="mixed")})
ordered = s.loc[hilbert_order(s)].reset_index(drop=True)
part = (np.arange(len(ordered)) * BYTES) // MAX                         # taco-eo's cut with equal sample sizes
cells = ordered["grid_cell"].unique()
xy = dict(zip(cells, (cell_from_name(c).centroid for c in cells)))
ordered["lon"] = ordered["grid_cell"].map(lambda c: xy[c][0])
ordered["lat"] = ordered["grid_cell"].map(lambda c: xy[c][1])
ordered["part"] = part
g = ordered.groupby("part")
km = np.sqrt((g.lon.max() - g.lon.min()) * 111 * np.cos(np.radians(g.lat.mean())) * (g.lat.max() - g.lat.min()) * 111)
split = (ordered.groupby("grid_cell")["part"].nunique() > 1).sum()
print(f"{len(ordered):,} samples, {len(cells):,} cells -> {part.max() + 1:,} partitions of {g.size().median() * BYTES / 1e9:.2f} GB"
      f" (last {g.size().iloc[-1] * BYTES / 1e9:.2f}); cells per partition median {g.grid_cell.nunique().median():.0f};"
      f" area ~{km.median():.0f} km across (median), p90 {km.quantile(.9):.0f} km, max {km.max():.0f} km;"
      f" cells split across two partitions: {split} of {len(cells):,}")

first = ordered.drop_duplicates("grid_cell")
fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
dots = ax.scatter(first.lon, first.lat, c=first.index, cmap=BLUES, s=0.05, linewidths=0, rasterized=True)
ax.set(xlim=(-180, 180), ylim=(-90, 90), aspect="equal", xticks=[], yticks=[])
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title("Core-S2L2A cells in Hilbert writing order: light first, dark last. Neighbouring cells get neighbouring "
             "positions,\nso each run of ~480 cells (one 3 GB partition) covers a compact patch", loc="left", fontsize=10)
fig.colorbar(dots, ax=ax, shrink=0.6, pad=0.01, label="position in the writing order (sample number)")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "hilbert_order.png", dpi=150)
