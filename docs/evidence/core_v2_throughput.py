"""How fast can Core-S2L1C v1 acquisitions be re-read at v2 windows? Gentle, escalating runs per archive.

    Core-S2L1C plan -> random samples that must be re-read (plan != "cut"), each pinned to its v1 acquisition day
      -> majortom.build.download(archive, workers) for N samples per run, runs in order:
           CDSE at 4, then 8, then 16 workers (stops escalating after a run with rate-limit errors or many retries)
           Earth Search at 64 workers
      -> per run: built, failed, samples per hour, retries, errors, baseline actually read -> <out>/throughput.csv
    python docs/evidence/core_v2_throughput.py /data/Major-TOM/core-v2/Core-S2L1C-plan.parquet /data/Major-TOM/core-v2/throughput
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
RUNS = [("cdse", 4, 100), ("cdse", 8, 150), ("cdse", 16, 200), ("earth-search", 64, 400)]


def main(plan_path: str, out: str) -> None:
    from majortom.build import download
    from majortom.cells import Cell, _table

    out = Path(out)
    plan = pd.read_parquet(plan_path, columns=["grid_row_u", "grid_col_r", "product_id", "plan"])
    todo = plan[plan.plan != "cut"].sample(frac=1, random_state=0)
    rows, used, stop_cdse = [], 0, False
    for archive, workers, n in RUNS:
        if archive == "cdse" and stop_cdse:
            print(f"{archive} {workers} workers: skipped (rate limits at the previous step)", flush=True)
            continue
        pick = todo.iloc[used:used + n]
        used += n
        cells = _table([Cell(int(r), int(c)) for r, c in zip(pick.grid_row_u, pick.grid_col_r)])
        cells["datetime"] = pd.to_datetime(pick.product_id.str.split("_").str[2], format="%Y%m%dT%H%M%S").dt.strftime(
            "%Y-%m-%d").to_numpy()
        folder = out / f"{archive}-{workers}"
        t = time.perf_counter()
        failed = download(cells, "sentinel-2-l1c", archive=archive, cloud=None, prefer="nearest", outdir=folder,
                          workers=workers)
        hours = (time.perf_counter() - t) / 3600
        sides = [json.loads(p.read_text()) for p in folder.rglob("*.json") if p.name != "acquisitions.json"]
        sides = [s for s in sides if "timing" in s]
        retries = sum(s["timing"].get("retries", 0) for s in sides)
        errors = failed["error"].str.slice(0, 60).value_counts().head(3).to_dict() if len(failed) else {}
        throttled = any(k in str(errors) for k in ("429", "Too Many", "SlowDown", "503", "rate"))
        baselines = pd.Series([s.get("processing_baseline") for s in sides]).value_counts().to_dict()
        rows.append(dict(archive=archive, workers=workers, asked=n, built=len(sides), failed=len(failed),
                         hours=round(hours, 3), per_hour=round(len(sides) / hours) if hours else None, retries=retries,
                         throttled=throttled, errors=errors, baselines=baselines))
        print(rows[-1], flush=True)
        if archive == "cdse" and (throttled or retries > len(sides)):
            stop_cdse = True
    pd.DataFrame(rows).to_csv(out / "throughput.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["errors"]).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
