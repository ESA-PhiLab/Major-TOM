"""Why three published versions of the Major TOM 10 km grid have three cell counts.

    majortom/cells.py formulas (every row -1002..1001, its n_cols columns)      -> "formula"   (the reference)
    Major-TOM/Spatial-Reference-Grid global/MT_grid_10km_global.parquet (HF)  -> "srg"       grid_row_u, grid_col_r
    Major-TOM/index global.parquet (HF), column id (the cell name)            -> "index"     parsed to row, column
    major-tom/climate ERA5-Monthly/cells.parquet (local release copy)         -> "climate"   grid_row_u, grid_col_r
      -> per set: rows, duplicates, cells outside the formula's columns
      -> per set against the formula: missing and extra cells, by 10° latitude band and by position in the row
         (first or last column of the row, or interior)
      -> <out>/grid-cell-counts.md (tables) and <out>/grid-cell-differences.png (where the differences are)
Only the name/row/column columns are read from Hugging Face (parquet column reads over https).

    python docs/evidence/grid_cell_counts.py <out>
"""
from __future__ import annotations

import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.cells import Cell, n_rows  # noqa: E402

HF = "https://huggingface.co/datasets/Major-TOM/{}/resolve/main/{}"
CLIMATE = Path("/data/Major-TOM/climate-release/ERA5-Monthly/cells.parquet")


def formula() -> pd.DataFrame:
    """Every cell of the 10 km grid from the formulas: row r has n_cols columns, -(n // 2) .. n - 1 - n // 2."""
    half = n_rows(10.0) // 2
    rows, cols = [], []
    for r in range(-half, n_rows(10.0) - half):
        n = Cell(r, 0).n_cols
        rows.append(np.full(n, r, np.int32))
        cols.append(np.arange(-(n // 2), n - n // 2, dtype=np.int32))
    return pd.DataFrame({"row": np.concatenate(rows), "col": np.concatenate(cols)})


def remote(repo: str, path: str, columns: list[str]) -> pd.DataFrame:
    with fsspec.open(HF.format(repo, path), "rb", block_size=2**22) as f:
        return pq.read_table(f, columns=columns).to_pandas()


def from_names(names: pd.Series) -> pd.DataFrame:
    """Cell names such as 624U_29L (or MT10_624U_29L) -> signed row and column (as majortom.cells.cell_from_name, vectorised)."""
    parts = names.str.extract(r"^(?:MT10_)?(\d+)([UD])_(\d+)([RL])$")             # the index writes MT10_624U_29L
    if parts.isna().any().any():
        raise ValueError(f"not cell names: {names[parts.isna().any(axis=1)].head().tolist()}")
    return pd.DataFrame({"row": parts[0].astype(np.int32) * np.where(parts[1] == "U", 1, -1),
                         "col": parts[2].astype(np.int32) * np.where(parts[3] == "R", 1, -1)})


def load() -> dict[str, pd.DataFrame]:
    srg = remote("Spatial-Reference-Grid", "global/MT_grid_10km_global.parquet", ["grid_cell", "grid_row_u", "grid_col_r"])
    named = from_names(srg.grid_cell)
    assert (named.row.to_numpy() == srg.grid_row_u.to_numpy()).all() and (named.col.to_numpy() == srg.grid_col_r.to_numpy()).all()
    index = from_names(remote("index", "global.parquet", ["id"]).id)
    climate = pq.read_table(CLIMATE, columns=["grid_row_u", "grid_col_r"]).to_pandas()
    return {"formula": formula(), "srg": named, "index": index,
            "climate": climate.rename(columns={"grid_row_u": "row", "grid_col_r": "col"})}


def table(d: pd.DataFrame) -> str:
    """A Markdown table (pandas' to_markdown needs tabulate, which the project environment does not have)."""
    cell = lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)       # noqa: E731
    rows = [list(d.columns), ["---"] * d.shape[1]] + [[cell(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def describe(sets: dict[str, pd.DataFrame], ref: pd.DataFrame) -> pd.DataFrame:
    """Per set: rows, duplicates, missing from and extra to the formula, rows of the grid touched."""
    key = lambda d: set(zip(d.row.tolist(), d.col.tolist()))                       # noqa: E731
    want = key(ref)
    out = []
    for name, d in sets.items():
        have = key(d)
        out.append({"set": name, "rows": len(d), "duplicates": len(d) - len(have),
                    "missing": len(want - have), "extra": len(have - want),
                    "grid rows": d.row.nunique(), "row range": f"{d.row.min()}..{d.row.max()}"})
    return pd.DataFrame(out)


def differences(d: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Cells missing from (or extra to) `d` against the formula, with latitude and position in the row."""
    m = ref.merge(d.drop_duplicates(), how="outer", on=["row", "col"], indicator=True)
    m = m[m._merge != "both"].copy()
    m["kind"] = np.where(m._merge == "left_only", "missing", "extra")
    n = m.row.map({r: Cell(int(r), 0).n_cols for r in m.row.unique()})
    first, last = -(n // 2), n - 1 - n // 2
    m["position"] = np.select([m.col < first, m.col > last, m.col == first, m.col == last],
                              ["west of row", "east of row", "first column", "last column"], "interior")
    m["lat"] = (m.row + 0.5) * 180 / n_rows(10.0)
    m["lon"] = (m.col + 0.5) * 360 / n
    m["band"] = (np.floor(m.lat / 10) * 10).astype(int)
    return m[["row", "col", "kind", "position", "lat", "lon", "band"]]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sets = load()
    ref = sets["formula"]
    lines = ["# Major TOM 10 km grid: cell counts of four versions", "", table(describe(sets, ref)), ""]
    diffs = {}
    for name in ("srg", "index", "climate"):
        diff = differences(sets[name], ref)
        diffs[name] = diff
        diff.to_parquet(out / f"grid-differences-{name}.parquet")
        if diff.empty:
            lines += [f"## {name}: identical to the formula", ""]
            continue
        lines += [f"## {name} against the formula", "",
                  table(diff.groupby(["kind", "position"]).size().rename("cells").reset_index()), "",
                  table(diff.groupby(["kind", "band"]).size().rename("cells").reset_index()), ""]
    (out / "grid-cell-counts.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    plot(diffs, out / "grid-cell-differences.png")


def plot(diffs: dict[str, pd.DataFrame], path: Path) -> None:
    """One panel per published set: its missing and extra cells at their centres, lon/lat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True, constrained_layout=True)
    for ax, (name, d) in zip(axes, diffs.items()):
        for kind, colour in (("missing", "#eb6834"), ("extra", "#2a78d6")):
            k = d[d.kind == kind]
            ax.scatter(k.lon, k.lat, s=1, color=colour, label=f"{kind} ({len(k):,})", rasterized=True)
        ax.set_xlim(-180, 180), ax.set_ylim(-90, 90)
        ax.set_title(f"{name}: cells missing or extra against the formula", fontsize=9, loc="left")
        ax.grid(color="#e1e0d9", linewidth=0.5)
        ax.legend(fontsize=8, frameon=False, loc="lower left", markerscale=6)
    axes[0].set_ylabel("latitude, °")
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
