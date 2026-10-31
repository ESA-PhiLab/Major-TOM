"""Core-S2L1C v1 products against what the archives serve today (catalogue only, no pixels; gentle).

    Core-S2L1C plan (docs/evidence/core_v2_plan.py) -> distinct v1 products (tile + acquisition)
      -> stratified sample: up to PER products per (year, baseline)
      -> per product, CDSE STAC and Earth Search STAC: items for the same tile and acquisition time
         -> the baselines each archive serves: same as v1, newer (reprocessed), or none
      -> shares, weighted back to samples -> <out>.csv and a printed summary
    Paced at ~2 queries per second per archive; stops an archive after repeated errors (429 or others).

    python docs/evidence/core_v2_catalogue.py /data/Major-TOM/core-v2/Core-S2L1C-plan.parquet <out.csv>
"""
import sys
import time

import pandas as pd
import pystac_client

ARCHIVES = {"cdse": ("https://stac.dataspace.copernicus.eu/v1", "sentinel-2-l1c"),
            "earth-search": ("https://earth-search.aws.element84.com/v1", "sentinel-2-l1c")}
PER, PAUSE = 60, 0.5


def baseline_of(item) -> str | None:
    """Processing baseline as N0xxx: from the properties, else from the product name."""
    b = item.properties.get("s2:processing_baseline") or item.properties.get("processing:version")
    if b:
        return "N" + str(b).replace(".", "").zfill(4)
    for part in item.id.split("_"):
        if part.startswith("N") and len(part) == 5 and part[1:].isdigit():
            return part
    return None


def main(plan_path: str, out: str) -> None:
    plan = pd.read_parquet(plan_path, columns=["product_id", "baseline", "plan"])
    products = plan.groupby("product_id").agg(baseline=("baseline", "first"), samples=("plan", "size")).reset_index()
    products["year"] = products.product_id.str.slice(11, 15)
    products["tile"] = products.product_id.str.split("_").str[5].str[1:]
    products["when"] = pd.to_datetime(products.product_id.str.split("_").str[2], format="%Y%m%dT%H%M%S")
    pick = products.sample(frac=1, random_state=0).groupby(["year", "baseline"]).head(PER)
    print(f"{len(products):,} distinct v1 products; checking {len(pick)} (up to {PER} per year and baseline)", flush=True)
    clients = {a: (pystac_client.Client.open(url), coll) for a, (url, coll) in ARCHIVES.items()}
    errors = {a: 0 for a in ARCHIVES}
    rows = []
    for i, r in enumerate(pick.itertuples()):
        row = dict(product_id=r.product_id, year=r.year, baseline=r.baseline, samples=r.samples)
        for archive, (client, collection) in clients.items():
            if errors[archive] >= 10:
                row[archive] = "stopped"
                continue
            t0 = (r.when - pd.Timedelta(minutes=2)).isoformat() + "Z"
            t1 = (r.when + pd.Timedelta(minutes=2)).isoformat() + "Z"
            try:
                mgrs = {"mgrs:utm_zone": {"eq": int(r.tile[:2])}, "mgrs:latitude_band": {"eq": r.tile[2]},
                        "mgrs:grid_square": {"eq": r.tile[3:]}}            # Earth Search: tile without a leading zero
                items = [it for it in client.search(collections=[collection], datetime=f"{t0}/{t1}", max_items=20,
                                                    query=mgrs if archive == "earth-search" else None).items()
                         if f"T{r.tile}" in (it.properties.get("s2:product_uri") or it.id) or r.tile in it.id]
                served = sorted({b for b in (baseline_of(it) for it in items) if b})
                row[archive] = ",".join(served) if served else "none"
            except Exception as error:
                errors[archive] += 1
                row[archive] = f"error: {type(error).__name__}"
            time.sleep(PAUSE)
        rows.append(row)
        if i % 100 == 0:
            print(f"{i}/{len(pick)} {row}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    for archive in ARCHIVES:
        same = df.apply(lambda x: x.baseline in str(x[archive]).split(","), axis=1)
        newer = df.apply(lambda x: any(b > x.baseline for b in str(x[archive]).split(",") if b.startswith("N")), axis=1)
        print(f"\n{archive}: by v1 baseline, share of products whose v1 version is still served / a newer one exists")
        summary = df.assign(same=same, newer=newer).groupby("baseline").agg(products=("same", "size"),
                                                                            same=("same", "mean"), newer=("newer", "mean"))
        print(summary.round(2).to_string())
        w = df.samples
        print(f"weighted by samples: v1 version served {((same * w).sum() / w.sum()):.1%}, "
              f"newer version exists {((newer * w).sum() / w.sum()):.1%}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
