"""Regenerate published Core embeddings with majortom.embedder and compare them, fragment by fragment.

    published embedding file (local copy) -> N products spread through the file
      -> their Core metadata rows (local metadata.parquet) -> each sample's image row group over HTTP
      -> MajorTOM_Embedder(model).forward(row, row_meta) -> regenerated fragments
      -> matched to the published fragments on (product_id, pixel_bbox)
      -> cosine similarity, unique_id equal, published unique_id recomputed from its own columns

    python docs/evidence/embedding_reproducibility.py <out folder> [products per dataset]     (Slurm GPU job)

Acceptance (docs/STRATEGY.md WS6): cosine >= 0.999 for every fragment.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import torch
from fsspec.parquet import open_parquet_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.embedder import MajorTOM_Embedder                       # noqa: E402
from majortom.embedder.models import SigLIP_S2RGB_Embedder, SSL4EO_S2L1C_Embedder   # noqa: E402

DATA = Path("/data/Major-TOM")
CASES = {   # published embeddings: (model, local embedding file, Core dataset of the images)
    "Core-S2RGB-SigLIP": (SigLIP_S2RGB_Embedder, DATA / "Core-S2RGB-SigLIP/embeddings/part_02801-02900.parquet",
                          "Core-S2L2A"),
    "Core-S2L1C-SSL4EO": (SSL4EO_S2L1C_Embedder, DATA / "Core-S2L1C-SSL4EO/embeddings/part_00001-00050.parquet",
                          "Core-S2L1C"),
}
COLUMNS = ["unique_id", "embedding", "timestamp", "product_id", "grid_cell", "geometry", "pixel_bbox"]


def checksum(geometry_wkb: bytes, timestamp: str, product_id: str, embedding) -> str:
    """unique_id from a published row's own columns (MajorTOM_Embedder.calculate_checksum)."""
    text = f"{shapely.from_wkb(geometry_wkb)}_{timestamp}_{product_id}_{np.asarray(embedding, dtype=np.float64)}"
    return hashlib.sha256(text.encode()).hexdigest()


def core_row(meta_row: pd.Series, bands: list[str]):
    """One sample's image row group, read over HTTP (only that row group and those bands)."""
    row_group = int(meta_row.parquet_row)
    with open_parquet_file(meta_row.parquet_url, columns=bands, row_groups=[row_group]) as f:
        return pq.ParquetFile(f).read_row_group(row_group, columns=bands)


def compare(name: str, n: int, device: str) -> pd.DataFrame:
    model, path, core = CASES[name]
    published = pq.read_table(path, columns=COLUMNS).to_pandas()
    samples = published[["product_id", "grid_cell"]].drop_duplicates()       # a sample: one product at one cell
    chosen = samples.iloc[np.linspace(0, len(samples) - 1, n).astype(int)]
    published = published.merge(chosen, on=["product_id", "grid_cell"])
    meta = pd.read_parquet(DATA / core / "metadata.parquet",
                           filters=[("product_id", "in", list(chosen.product_id))])
    embedder = MajorTOM_Embedder(model()).to(device).eval()      # inference: no dropout, BatchNorm running statistics
    out = []
    for (product_id, cell), theirs in published.groupby(["product_id", "grid_cell"]):
        m = meta[(meta.product_id == product_id) & (meta.grid_cell == cell)].head(1)
        ours = embedder(core_row(m.iloc[0], embedder.bands()), m, device=device)
        ours["key"] = ours.pixel_bbox.map(tuple)
        theirs = theirs.assign(key=theirs.pixel_bbox.map(tuple))
        both = theirs.merge(ours[["key", "embedding", "unique_id"]], on="key", suffixes=("_pub", "_new"))
        a, b = np.stack(both.embedding_pub).astype(np.float64), np.stack(both.embedding_new).astype(np.float64)
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        out.append(pd.DataFrame({
            "dataset": name, "product_id": product_id, "grid_cell": cell, "pixel_bbox": both.key.map(list),
            "fragments_published": len(theirs), "fragments_new": len(ours), "cosine": cos,
            "max_abs_diff": np.abs(a - b).max(1), "unique_id_equal": both.unique_id_pub == both.unique_id_new,
            "published_id_recomputes": [checksum(g, t, p, e) == u for g, t, p, e, u in
                                        zip(both.geometry, both.timestamp, both.product_id, both.embedding_pub,
                                            both.unique_id_pub)]}))
        print(name, product_id, cell, f"{len(both)} fragments, min cosine {cos.min():.6f}", flush=True)
    return pd.concat(out, ignore_index=True)


def main(out: Path, n: int = 5) -> None:
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = pd.concat([compare(name, n, device) for name in CASES], ignore_index=True)
    table.to_csv(out / "embedding_reproducibility.csv", index=False)
    summary = table.groupby("dataset").agg(samples=("grid_cell", lambda c: len(set(zip(table.loc[c.index, "product_id"], c)))), fragments=("cosine", "size"),
                                           cosine_min=("cosine", "min"), cosine_median=("cosine", "median"),
                                           max_abs_diff=("max_abs_diff", "max"),
                                           unique_id_equal=("unique_id_equal", "mean"),
                                           published_id_recomputes=("published_id_recomputes", "mean"))
    summary["passes"] = summary.cosine_min >= 0.999
    (out / "embedding_reproducibility.json").write_text(summary.reset_index().to_json(orient="records", indent=1))
    print(summary.to_string())
    if device == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), *(int(a) for a in sys.argv[2:3]))
