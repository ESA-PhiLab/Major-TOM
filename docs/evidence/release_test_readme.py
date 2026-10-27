"""README and figures for major-tom/release-test, generated from the build, the releases and the checks.

    <root>/build/<source>/<query>/<product>/*.npz + .json      (sidecars: archive, bands, timing, datetime)
    <root>/release-test/<source>/  (partitions, .tacocat)    <root>/release-test/check.csv
      -> figures/cells.png    the test cells on a map of land cells (ERA5Land-Monthly's cell list)
      -> figures/gallery.png  one quicklook per source, first cell available (Loch Lomond first)
      -> README.md            what this is; cells; gallery; modalities table; access points with quotas and
                              measured speeds; how to read it; how it was made
    python docs/evidence/release_test_readme.py --root /data/Major-TOM/release-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom.build.collections import COLLECTIONS                       # noqa: E402
from majortom.cells import cell_from_name                                # noqa: E402

CELLS = {"624U_29L": "Loch Lomond", "518U_90R": "Dolomites (Corvara)", "35D_667L": "Amazon (Manaus)",
         "429U_786L": "St Louis (fallback)"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CLASSES = ListedColormap(["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
RGB = {"sentinel-2-l2a": ("B04", "B03", "B02"), "sentinel-2-l1c": ("B04", "B03", "B02"),
       "hls2-s30": ("B04", "B03", "B02"), "hls2-l30": ("B04", "B03", "B02"),
       "landsat-c2-l2": ("red", "green", "blue"), "landsat-c2-l1": ("B4", "B3", "B2"),
       "aster-l1t": ("B3N", "B02", "B01"), "modis-09a1": ("sur_refl_b01", "sur_refl_b04", "sur_refl_b03"),
       "modis-43a4": ("Nadir_Reflectance_Band1", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band3"),
       "sentinel-3-olci-l1": ("Oa08_radiance", "Oa06_radiance", "Oa04_radiance")}
RADAR_DB = {"sentinel-1-rtc", "opera-rtc-s1", "sentinel-1-grd", "alos-palsar-mosaic"}   # linear or DN: shown in dB
RADAR = RADAR_DB | {"sentinel-1-grd-tc"}                                               # all radar shown in grey
SHOW = {"dynamic-world": "label", "esa-worldcover": "map", "esa-cci-lc": "lccs_class", "jrc-gsw": "occurrence",
        "opera-dswx-hls": "WTR", "opera-dswx-s1": "WTR", "opera-dist-alert-hls": "VEG-DIST-STATUS",
        "hgb": "aboveground", "viirs-dnb-monthly": "avg_rad", "modis-11a2": "LST_Day_1km", "modis-13q1": "NDVI",
        "sentinel-3-olci-l2-lfr": "OTCI", "modis-09q1": "sur_refl_b02"}
ARCHIVES = {
    "planetary-computer": ("Microsoft Planetary Computer", "none (anonymous; signed URLs)", "no quota; fair use",
                           "STAC + COG over HTTPS"),
    "earthdata": ("NASA Earthdata (LP DAAC, PO.DAAC, ASF)", "Earthdata token (60 days); ASF approval for OPERA RTC",
                  "no quota", "STAC + COG over HTTPS"),
    "cdse": ("Copernicus Data Space Ecosystem", "S3 keys (free account)", "per-user S3 rate limits",
             "STAC + JPEG2000/SAFE over S3"),
    "earth-search": ("Element 84 Earth Search (AWS)", "none (anonymous)", "no quota", "STAC + COG over HTTPS"),
    "earth-engine": ("Google Earth Engine", "Cloud project + service account", "noncommercial tiers: Community "
                     "~150, Contributor 1,000 EECU-hours/month (measured: ~7.8 EECU-s per 600 px ERA5-Land tile)",
                     "computePixels / getPixels"),
}


def _samples(root: Path) -> pd.DataFrame:
    rows = []
    for js in sorted((root / "build").rglob("*.json")):
        side = json.loads(js.read_text())
        if "grid_cell" not in side:
            continue
        rows.append(dict(source=js.parts[len(root.parts) + 1], npz=js.with_suffix(".npz"), cell=side["grid_cell"],
                         archive=side["archive"], datetime=side.get("datetime"), seconds=side["timing"]["total"],
                         cloud=side.get("cloud") if side.get("cloud") is not None else 0.0,
                         missing=side.get("nodata") or 0.0,                     # share of the window without data
                         search=side["timing"].get("search", float("nan")),
                         choice=side["timing"].get("scene_choice", float("nan")),
                         reads=sum(side["timing"]["bands"].values()) if isinstance(side["timing"].get("bands"), dict)
                         else float("nan"),
                         bytes=js.with_suffix(".npz").stat().st_size, side=side))
    return pd.DataFrame(rows)


def _fill(x: np.ndarray, side: dict) -> np.ndarray:
    """Values to show: NaN where the build wrote its fill (0 for integers outside the source files, when part of
    the window is missing) and at MODIS-style integer fill values."""
    integer = np.issubdtype(x.dtype, np.integer)
    x = x.astype(float)
    if integer and (side.get("nodata") or 0) > 0:     # part of the window has no data: its pixels were written as 0
        x[x == 0] = np.nan
    if integer:
        x[np.abs(x) >= 32767] = np.nan
    return x


def _dominance(npz: Path, source: str, side: dict) -> float:
    """Share of the most common value of the band shown: 1.0 for an empty or all-cloud image."""
    band = (RGB.get(source) or [SHOW.get(source) or side["bands"][0]])[0]
    _, counts = np.unique(np.load(npz)[band], return_counts=True)
    return counts.max() / counts.sum()


def _stretch(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [2, 98]) if np.isfinite(x).any() else (0, 1)
    return np.clip((x - lo) / (hi - lo or 1), 0, 1)


def _quicklook(source: str, npz: Path, side: dict):
    """(image, colour map or None, caption) for one sample."""
    a = np.load(npz)
    if source in RGB:
        planes = [_fill(a[b], side) for b in RGB[source]]
        missing = np.any([np.isnan(q) for q in planes], axis=0)              # missing in any band: background
        rgb = np.dstack([np.where(missing, 1.0, np.nan_to_num(_stretch(q), nan=1.0)) for q in planes])
        caption = " / ".join(RGB[source]) if sum(map(len, RGB[source])) <= 18 else "true colour"
        return rgb, None, caption if source != "aster-l1t" else "false colour (NIR / red / green)"
    band = SHOW.get(source) or ("vv" if "vv" in side["bands"] else "VV" if "VV" in side["bands"] else side["bands"][0])
    x = _fill(a[band], side) if COLLECTIONS[source].resampling != "nearest" else a[band].astype(float)
    if COLLECTIONS[source].kind == "elevation":
        return LightSource(315, 45).hillshade(np.nan_to_num(x), vert_exag=0.05, dx=30, dy=30), "gray", f"{band}, hillshade"
    if source in RADAR:
        if source in RADAR_DB:
            x = 10 * np.log10(np.where(x > 0, x, np.nan))
        return _stretch(x), "gray", f"{band}, dB"
    if COLLECTIONS[source].resampling == "nearest" and source not in ("jrc-gsw", "hgb"):
        codes = np.unique(x[np.isfinite(x)])
        idx = np.searchsorted(codes, np.nan_to_num(x, nan=codes[0] if len(codes) else 0))
        return np.ma.masked_invalid(np.where(np.isfinite(x), idx % 8, np.nan)), CLASSES, f"{band}, classes"
    return _stretch(x), BLUES, f"{band}, 2-98% stretch"


def gallery(samples: pd.DataFrame, out: Path) -> None:
    order = list(CELLS)
    ranked = samples.assign(rank=samples.cell.map({c: i for i, c in enumerate(order)}).fillna(9),
                            dominance=[_dominance(r.npz, r.source, r.side) for r in samples.itertuples()])
    ranked["uniform"] = ranked.dominance > 0.95                         # all cloud, all zero: shown only if nothing else
    firsts = (ranked.sort_values(["source", "rank", "uniform", "missing", "cloud", "dominance", "datetime"])
              .drop_duplicates("source"))                    # Loch Lomond first: not uniform, least missing, clearest
    n = len(firsts)
    cols = 6
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(cols * 2.3, -(-n // cols) * 2.75), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, firsts.itertuples()):
        image, cmap, caption = _quicklook(r.source, r.npz, r.side)
        ax.imshow(image, cmap=cmap, interpolation="nearest", vmin=0 if cmap is CLASSES else None,
                  vmax=7 if cmap is CLASSES else None)
        when = (r.datetime or "")[:10]
        ax.set_title(f"{r.source}\n{CELLS.get(r.cell, r.cell)}  {when}\n{caption}", fontsize=7, color=INK, loc="left")
    fig.savefig(out / "gallery.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def cell_map(samples: pd.DataFrame, out: Path) -> None:
    land = pd.read_parquet("/data/Major-TOM/ERA5Land-Monthly/cells.parquet", columns=["grid_row_u", "grid_col_r"])
    land = land.sample(min(len(land), 400_000), random_state=0)
    lat = land.grid_row_u.to_numpy() * 180 / 2004
    n = np.maximum(1, np.ceil(2 * np.pi * 6378.137 * np.cos(np.radians(lat)) / 10))
    lon = land.grid_col_r.to_numpy() * 360 / n
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.scatter(lon, lat, s=0.05, color="#d9d8d3", linewidths=0, rasterized=True)
    used = samples.groupby("cell").source.nunique()
    for cell, label in CELLS.items():
        if cell not in used:
            continue
        lon_c, lat_c = cell_from_name(cell).centroid
        ax.scatter([lon_c], [lat_c], s=60, color="#2a78d6", edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{label}\n{cell} ({used[cell]} source{'s' if used[cell] != 1 else ''})", (lon_c, lat_c), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set(xlim=(-180, 180), ylim=(-60, 85), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out / "cells.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _size(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


def _payload(folder: Path) -> int:
    """Bytes of the rumi files inside the partitions: the size in each /vsisubfile/<offset>_<size>,... location."""
    import taco

    path = folder / ".tacocat" if (folder / ".tacocat").exists() else next(folder.glob("*.zip"))
    table = taco.open_dataset(str(path)).sql("SELECT * FROM dataset").to_pandas()
    total = 0
    for column in (c for c in table.columns if c.endswith("::location")):
        total += sum(int(v.split(",")[0].rsplit("_", 1)[1]) for v in table[column])
    return total


def _folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and "stage" not in p.parts[-2])


def readme(root: Path, samples: pd.DataFrame) -> str:
    check = pd.read_csv(root / "release-test" / "check.csv").set_index("collection")
    rows = []
    for source, g in samples.groupby("source"):
        c = COLLECTIONS[source]
        side = g.iloc[0].side
        a = np.load(g.iloc[0].npz)
        sizes = sorted({round(side["window_m"] / a[b].shape[-1], 3) for b in side["bands"]})
        dtypes = sorted({str(a[b].dtype) for b in side["bands"]})
        folder = root / "release-test" / source
        released = _folder_size(folder) if folder.exists() else 0
        payload = _payload(folder) if folder.exists() else 0
        raw = sum(sum(np.load(p)[b].nbytes for b in json.loads(p.with_suffix(".json").read_text())["bands"])
                  for p in g.npz)
        k = check.loc[source] if source in check.index else None
        rows.append(f"| `{source}` | {c.kind} | {', '.join(f'{s:g} m' for s in sizes)} | {len(side['bands'])} | "
                    f"{', '.join(dtypes)} | {c.time}{' (classes)' if c.resampling == 'nearest' else ''} | "
                    f"{side['archive']} | {len(g)} | {_size(released)} | "
                    f"{raw / payload if payload else float('nan'):.2f} | "
                    f"{int(k.partitions) if k is not None and k.get('partitions') == k.get('partitions') else '-'} | "
                    f"{k.status if k is not None else '-'} |")
    speed = (samples.assign(mb=samples.bytes / 1e6).groupby("archive")
             .agg(sources=("source", "nunique"), samples=("source", "size"), median_s=("seconds", "median"),
                  p90_s=("seconds", lambda s: s.quantile(.9)), mb=("mb", "median")))
    access = [f"| {ARCHIVES[a][0]} (`{a}`) | {ARCHIVES[a][1]} | {ARCHIVES[a][2]} | {ARCHIVES[a][3]} | "
              f"{int(r.sources)} | {r.median_s:.0f} s (p90 {r.p90_s:.0f} s) | {r.mb:.1f} MB |"
              for a, r in speed.iterrows()]
    per = (samples.groupby(["source", "archive"])
           .agg(n=("seconds", "size"), total=("seconds", "median"), search=("search", "median"),
                choice=("choice", "median"), reads=("reads", "median"), mb=("bytes", "median"))
           .sort_values("total"))
    per_rows = []
    for (source, archive), r in per.iterrows():
        parts = {"search": r.search, "scene choice": r.choice, "reads": r.reads}
        main = max(parts, key=lambda k: parts[k] if parts[k] == parts[k] else -1)
        per_rows.append(f"| `{source}` | {archive} | {int(r.n)} | {r.total:.1f} s | {r.search:.1f} s | {r.choice:.1f} s | "
                        f"{r.reads:.1f} s | {main} | {_size(r.mb)} |")
    remote_rows, remote_note = [], ""
    if (root / "release-test" / "remote.csv").exists():
        rc = pd.read_csv(root / "release-test" / "remote.csv")
        mine = rc[~rc.source.str.startswith("core-dem")]
        dem = rc[rc.source.str.startswith("core-dem")]
        exact = f"{int(mine.exact.sum())}/{int(mine.files.sum())}"
        remote_rows = [f"| `{r.source}` | {r.open_s:.2f} s | {r.read_median_s:.2f} s | {r.read_max_s:.2f} s | {int(r.exact)}/{int(r.files)} |"
                       for r in mine.itertuples()]
        remote_note = (f"All {exact} files read from source.coop were identical to the built samples. For comparison, "
                       f"Core-DEM (us-west-2, 1.7 million samples): open {float(dem.open_s.iloc[0]):.1f} s (a much larger "
                       f"catalogue), one sample {float(dem.read_median_s.iloc[0]):.2f} s (median of 10).") if len(dem) else ""
    total = sum(_folder_size(f) for f in (root / "release-test").iterdir() if f.is_dir())
    moved = sorted(samples[samples.cell == "429U_786L"].source.unique())
    fallback = (f"Sources with no data at any of the three cells use **St Louis** (`429U_786L`): {', '.join(f'`{m}`' for m in moved)}."
                if moved else "Every source had data at these cells.")
    return f"""# Major TOM release test

> **Not for use.** A test of the Major TOM release framework (`majortom.release`), published to try the whole path,
> from building samples to reading them back from source.coop, before the first real release
> (Sentinel-2 L2A, Major TOM v2). Every number on this page is generated from the data it describes.

{len(samples.source.unique())} sources, {len(samples)} samples, {total / 1e9:.2f} GB, each source its own TACO dataset
in its own folder, made exactly as a real release would be (`python -m majortom.release build`).

## Where

![The test cells](https://data.source.coop/major-tom/release-test/figures/cells.png)

Three cells of the Major TOM 10 km grid: **Loch Lomond** (`624U_29L`), the **Dolomites near Corvara** (`518U_90R`)
and the **Amazon near Manaus** (`35D_667L`). {fallback} Per cell: two dates for scene products (the clearest of
May-June and of August-September 2023), two years for yearly products (2020, 2021), one sample for static products.
Fewer samples mean the archive had nothing for that cell and period.

## What

![One quicklook per source](https://data.source.coop/major-tom/release-test/figures/gallery.png)

| Source | Kind | Pixel sizes | Bands | Data types | Time | Archive | Samples | Released (all files) | Compression (rumi) | Partitions | Check |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

"Compression" is the raw size of the arrays over the size of their rumi files (lossless, GeoZL); "Released" also
counts TACO's catalogue in every partition, which dominates for the smallest sources. Below 1, the data are
smaller than the rumi file's own header (MODIS 1 km: 12 x 12 pixels, ~300 bytes per image). "Check": `taco.validate` passes
and every file reads back identical to the built sample.

## Access points, quotas and speeds

| Archive | Account or key | Quota | Access | Sources | Build time per sample (median) | Built sample, uncompressed (median) |
|---|---|---|---|---|---|---|
{chr(10).join(access)}

Per source (medians over its samples; fastest first):

| Source | Archive | Samples | Build time | Search | Scene choice | Reads | Mostly | Built sample, uncompressed |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(per_rows)}

Build times are measured per sample (search, scene choice, reads of every band; `timing` in each sample's sidecar),
built with 3 workers from `cirrus` on a ~63 MB/s link shared with other large downloads at the time: they show which
sources are slow and why, not their best speed (Sentinel-2 L2A took 97 s per sample here, 18 s in a quieter
benchmark with 16 workers, docs/reports/2026-10-22-build-speed.md in the Major TOM repository).
Notes from building Major TOM: Earth Engine bills reading stored images far above what its profiler reports
(ERA5-Land hourly: ~30x); Destination Earth's Earth Data Hub serves ERA5-Land as zarr with a free quota of
500,000 requests a month (one request per 60-day chunk).

## Reading it from source.coop: measured

Opened and read from `https://data.source.coop/major-tom/release-test/<source>/` without credentials, from `cirrus`
(UK); the data are stored in AWS eu-central-1 (Frankfurt). "Open": `taco.open_dataset` and the catalogue query;
"read": one file of one sample with `rumi.read` over HTTP (byte ranges inside the partition ZIP).

| Source | Open | Read (median) | Read (max) | Identical to build |
|---|---|---|---|---|
{chr(10).join(remote_rows)}

{remote_note}

## Reading it

```python
import taco, rumi
ds = taco.open_dataset("https://data.source.coop/major-tom/release-test/sentinel-2-l2a/")
rows = ds.sql('SELECT id, "10m.rumi::location" AS loc, "10m.rumi::header" AS hdr FROM dataset').to_pylist()
array = rumi.read(rows[0]["loc"], rows[0]["hdr"])               # (bands, y, x), only the bytes of this sample

from majortom.release import read                                # the same sample as a GeoTIFF, for any GDAL tool
with read(rows[0]["loc"], rows[0]["hdr"], as_="cog") as mem, mem.open() as f:
    print(f.crs, f.transform, f.descriptions)
```

Whole datasets or selections as COG files: `python -m majortom.release to-cog --dataset <url> --out <folder>
--where "<SQL condition>"`. Each sample also carries STAC time and footprint, its Major TOM cell (`majortom:code`),
its source products (`source:*`) and environmental attributes (`geoenrich:*`).

## How it was made

`majortom.build` (one window per cell, Major TOM v2) -> `majortom.release`: raster groups by pixel size and data
type -> lossless rumi files (GeoZL, all predictor families tried) -> samples written in Hilbert order over cell
centroids -> taco-eo partitions by size and writes the TACOCAT -> published with `python -m majortom.release
publish`. Design and measurements: `docs/RELEASE.md` in the Major TOM repository; this page:
`docs/evidence/release_test_readme.py`.
"""


def main(root: str) -> None:
    root = Path(root)
    out = root / "release-test"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    samples = _samples(root)
    cell_map(samples, figures)
    gallery(samples, figures)
    (out / "README.md").write_text(readme(root, samples))
    print(f"README.md and figures for {samples.source.nunique()} sources, {len(samples)} samples in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
