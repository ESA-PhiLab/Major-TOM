"""Sample Major TOM Core rows via HTTP range reads and inspect band geotransforms.

Reads one random row from N random metadata row groups, then fetches only
the requested band columns of that single row group from the data parquet.
"""
import json
import random
import sys
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file
from rasterio.io import MemoryFile

DS = sys.argv[1]
N = int(sys.argv[2])
BANDS = sys.argv[3].split(",")
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUT = Path(__file__).parent / f"results_{DS}_{SEED}.jsonl"

random.seed(SEED)
meta_url = f"https://huggingface.co/datasets/Major-TOM/{DS}/resolve/main/metadata.parquet"
mf = pq.ParquetFile(fsspec.open(meta_url).open())
rgs = random.sample(range(mf.metadata.num_row_groups), N)

with OUT.open("w") as out:
    for rg in rgs:
        t = mf.read_row_group(rg).to_pandas()
        row = t.iloc[random.randrange(len(t))]
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        try:
            with open_parquet_file(row.parquet_url, columns=BANDS,
                                   row_groups=[int(row.parquet_row)]) as f:
                pf = pq.ParquetFile(f)
                tab = pf.read_row_group(int(row.parquet_row), columns=BANDS)
            for b in BANDS:
                with MemoryFile(tab[b][0].as_py()) as m, m.open() as src:
                    tr = src.transform
                    rec[b] = dict(crs=str(src.crs), shape=[src.height, src.width],
                                  a=tr.a, b=tr.b, c=tr.c, d=tr.d, e=tr.e, f=tr.f,
                                  dtype=src.dtypes[0])
        except Exception as e:  # keep going on individual failures
            rec["error"] = repr(e)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec.get("grid_cell"), rec.get("product_id"), rec.get("crs"),
              {b: rec.get(b) for b in BANDS}, rec.get("error", ""), flush=True)
