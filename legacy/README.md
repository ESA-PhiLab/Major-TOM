> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed (added 2026-11-06).

# Legacy notebooks

Notebooks from before the `majortom` package refactor (2026), kept unchanged as a record. Some imports no longer
resolve (`MajorTom.grid`, `src.grid`). The notebooks in the repository root replace them.

| Notebook | Content | Replaced by |
|---|---|---|
| `01-Major-TOM-Filter.ipynb` | filter and download Core samples; PyTorch Dataset; streaming; filter by points (volcanoes) | `01-PyTorch-Training.ipynb` |
| `02-Major-TOM-Quick-Viewer.ipynb` | ipywidgets and leafmap viewer of Core samples | the [Core Viewer](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer) Space (`apps/viewer/`) |
| `03-Filtering-in-Colab.ipynb` | `01-Major-TOM-Filter.ipynb` for Colab, without the volcano section | `01-PyTorch-Training.ipynb` |
| `05-Generate-Major-TOM-Embeddings.ipynb` | regenerate the Core embeddings | `02-Generate-Embeddings-and-Search.ipynb` |
| `06-Read-AlphaEarth-Embeddings.ipynb` | read AlphaEarth embeddings for Major TOM cells | `03-Read-AlphaEarth.ipynb` |
| `Embeddings-FAISS-Search.ipynb` | FAISS index over the embeddings, text query (was `majortom/extras/`) | `02-Generate-Embeddings-and-Search.ipynb` |
| `extract-sample-from-raw-S2.ipynb` | cut a sample from a raw Sentinel-2 product (was `majortom/extras/`) | `majortom.build` (`00-Major-TOM.ipynb`, section 4) |
