"""Build Major TOM samples from Planetary Computer (needs `pip install 'majortom[build]'`).

    cells (from points or an area)  ->  per sample: window -> search -> pick a scene -> read -> write

    from majortom import build
    cells = build.cells_from_points(points, d=10)         # or build.cells_in(bbox, d=10)
    errors = build.download(cells, "sentinel-2-l2a", "2024-06-01/2024-08-31")

Each sample is written as chips/<collection>/<index>.npz (one array per band) and a .json sidecar with
its georeferencing and Major TOM fields, the layout used by the taco workshop notebook.

Per-sample queries: optional columns in `cells` override the arguments of `download` for that row:
`datetime` (an interval "start/end" or one time), `days` (search +- days around that time) and
`margin` (window margin as a fraction of the grid spacing).

Which scene: `prefer` sets the order scenes are tried in ("clearest", "cloudiest", "nearest" to the
date, "newest"); `cloud` sets the accepted share of cloudy pixels in the window, e.g. (0, 0.05) for
clear or (0.3, 1) for cloudy samples, or None for any. Both are recorded in every sample.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from tqdm.auto import tqdm

from ..cells import Cell, cells_from_points, cells_in   # noqa: F401  re-exported for convenience
from ..spec import MARGIN, cell_coverage, window
from .collections import COLLECTIONS, Collection
from .read import GDAL_OPTIONS, layout, read_asset
from .search import acquired, scenes, tiles_for

PROFILE = "native-v0"          # recorded in every sample: native grid if the CRS matches, else resampled
CANDIDATES = 5                 # scenes examined per sample


def _field(row, name: str):
    """A per-sample column's value, or None when the column is missing or empty."""
    value = getattr(row, name, None)
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value


def interval(when, days) -> str | None:
    """STAC datetime for a sample: an interval as given, or one time widened by +- `days`."""
    if when is None:
        return None
    if days is None:
        return str(when)
    t, pad = pd.Timestamp(when), pd.Timedelta(days=float(days))
    return f"{(t - pad).isoformat()}/{(t + pad).isoformat()}"


def search_box(cell: Cell, margin: float) -> tuple[float, float, float, float]:
    """Lon/lat box around the cell's widest possible window, for the catalogue search."""
    win = window(cell, lattice=10, margin=margin)
    return transform_bounds(win.crs, "EPSG:4326", *win.bounds, densify_pts=21)


@dataclass(frozen=True)
class Choice:
    """How a sample's scene is chosen. Recorded in every sample.

    prefer      order in which scenes are tried: "clearest", "cloudiest", "nearest" (to the sample's
                date, or the middle of its range) or "newest"
    cloud       accepted share of cloudy pixels in the window, as (low, high); None accepts any
    max_nodata  accepted share of window pixels without data
    strict      if no scene tried is acceptable: True fails the sample, False takes the one closest to
                the accepted cloud range (its real cloud share is recorded)
    """
    prefer: str = "clearest"
    cloud: tuple[float, float] | None = (0.0, 0.05)
    max_nodata: float = 0.01
    strict: bool = False

    def accepts(self, cloud: float | None, nodata: float) -> bool:
        return nodata <= self.max_nodata and self.cloud_gap(cloud) == 0

    def cloud_gap(self, cloud: float | None) -> float:
        """How far a cloud share is outside the accepted range (0 inside it, or without a range)."""
        if cloud is None or self.cloud is None:
            return 0.0
        low, high = self.cloud
        return max(low - cloud, cloud - high, 0.0)


def pick_scene(cell, name, collection, query, margin, choice: Choice):
    """The first acceptable scene in the order of `choice.prefer`, among the first few.

    Returns (tiles, window, pixel sizes, native, cloud share, no-data share, mask asset data or None,
    note on the dates used).
    """
    mask_asset = collection.clouds[0] if collection.clouds else collection.assets[0]
    grid_assets = list(collection.assets) + ([mask_asset] if collection.clouds else [])
    tried = []
    found, note = scenes(name, collection, search_box(cell, margin), query, choice.prefer)
    for scene in found[:CANDIDATES]:
        for crs, tiles in tiles_for(scene, cell, margin):   # same scene, another zone if this one is empty
            win, res, native = layout(cell, tiles[0], collection, grid_assets, margin, crs)
            method = "nearest" if collection.clouds else collection.resampling   # without a mask: the data
            data, covered = read_asset(tiles, mask_asset, win, res[mask_asset], method)
            cloud = None
            if collection.clouds:                       # the mask also marks no-data pixels inside files
                clouds, fill = collection.clouds[1](data[0])
                covered &= ~fill
                cloud = float(clouds[covered].mean()) if covered.any() else 1.0
            nodata = 1 - covered.mean()
            tried.append((tiles, win, res, native, cloud, nodata, None if collection.clouds else data, note))
            if choice.accepts(cloud, nodata):
                return tried[-1]
            if nodata <= choice.max_nodata:             # data is there, only the clouds are off: next scene
                break
    if not tried:
        raise LookupError(f"no scene found ({note})")
    if choice.strict:
        raise LookupError(f"none of {len(tried)} scenes has cloud in {choice.cloud} "
                          f"and no-data <= {choice.max_nodata}")
    return min(tried, key=lambda t: (t[5] > choice.max_nodata, choice.cloud_gap(t[4]), t[5]))


def band_names(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One 2-D plane per name; an asset with several bands becomes asset_1, asset_2, ..."""
    planes = {}
    for asset, array in arrays.items():
        for i, plane in enumerate(array):
            planes[asset if len(array) == 1 else f"{asset}_{i + 1}"] = plane
    return planes


def _same_request(sidecar: Path, cell: Cell, query, margin: float, choice: Choice) -> bool:
    """Whether a saved sample was built for this cell, dates, margin and scene choice."""
    try:
        old = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return False
    return (old.get("grid_cell"), old.get("grid_km"), old.get("query", {}).get("datetime"),
            old.get("query", {}).get("margin"), old.get("choice")) == (
            cell.name, cell.d, query, margin, json.loads(json.dumps(asdict(choice))))


def build_one(row, name: str, collection: Collection, datetime, days, margin, choice: Choice,
              outdir: Path, force: bool) -> None:
    """Build and write one sample."""
    path = outdir / f"{row.Index:06d}.npz"
    cell = Cell(int(row.row), int(row.col), float(row.grid_km))
    margin = float(_field(row, "margin") if _field(row, "margin") is not None else margin)
    days = _field(row, "days") if _field(row, "days") is not None else days
    query = interval(_field(row, "datetime") or datetime, days)
    if path.exists():
        # reuse a sample on disk only if it was built for this cell, query and choice
        if not force and _same_request(path.with_suffix(".json"), cell, query, margin, choice):
            return
        path.unlink()                                   # never leave an old sample under this index
        path.with_suffix(".json").unlink(missing_ok=True)

    with rasterio.Env(**GDAL_OPTIONS):
        tiles, win, res, native, cloud, nodata, first, note = pick_scene(cell, name, collection, query, margin,
                                                                          choice)
        arrays = {}
        for asset in collection.assets:
            reuse = first is not None and asset == collection.assets[0]
            arrays[asset] = first if reuse else read_asset(tiles, asset, win, res[asset], collection.resampling)[0]

    planes = band_names(arrays)
    finest = min(res[a] for a in collection.assets)
    meta = dict(
        index=int(row.Index), lon=float(row.lon), lat=float(row.lat), collection=name,
        items=[it.id for it in tiles], datetime=acquired(tiles[0]).floor("us").isoformat(),
        cloud=cloud, nodata=float(nodata), crs=win.crs, epsg=int(win.crs.split(":")[1]),
        left=win.left, top=win.top, res=finest, size=win.pixels(finest), bands=list(planes),
        grid_cell=cell.name, grid_km=cell.d, window_m=win.side, cell_coverage=cell_coverage(cell, win),
        profile=PROFILE, resampled=[] if native else list(collection.assets),
        query=dict(datetime=query, margin=margin, dates_used=note), choice=asdict(choice),
    )
    # sidecar first, then the arrays atomically: a chip on disk is always a complete pair
    path.with_suffix(".json").write_text(json.dumps(meta))
    staging = path.with_name(path.name + ".part")
    with open(staging, "wb") as handle:
        np.savez(handle, **planes)
    os.replace(staging, path)


def download(cells: pd.DataFrame, collection: str, datetime: str | None = None, days: float | None = None,
             margin: float = MARGIN, prefer: str = "clearest", cloud: tuple[float, float] | None = (0.0, 0.05),
             max_nodata: float = 0.01, strict: bool = False,
             outdir: str | Path = "chips", workers: int = 16, force: bool = False) -> pd.DataFrame:
    """Build one sample per row of `cells` from a Planetary Computer collection.

    datetime     dates to search ("start/end" or one time). Yearly products: selects the version.
                 Static products: ignored
    days         widen a single `datetime` by +- days
    margin       window margin as a fraction of the grid spacing (0.056: 10,560 m at 10 km)
    prefer       order scenes are tried: "clearest", "cloudiest", "nearest" (to the date) or "newest"
    cloud        accepted share of cloudy pixels in the window, (low, high); None accepts any
    max_nodata   accepted share of window pixels without data
    strict       True: a sample with no acceptable scene fails; False: the closest to the range is used
    force        True: rebuild every sample. False: reuse a sample on disk if it was built for the same
                 cell, dates, margin and scene choice. Samples in outdir for rows not in `cells` are removed
    Returns the samples that failed, with the reason.
    """
    choice = Choice(prefer, tuple(cloud) if cloud is not None else None, max_nodata, strict)
    if prefer not in ("clearest", "cloudiest", "nearest", "newest"):
        raise ValueError(f"prefer must be clearest, cloudiest, nearest or newest, not {prefer!r}")
    if collection not in COLLECTIONS:
        raise KeyError(f"unknown collection {collection!r}; pick one of {sorted(COLLECTIONS)}")
    if cells.index.has_duplicates:
        raise ValueError("cells must have a unique index; call cells.reset_index(drop=True) first")
    out = Path(outdir) / collection
    out.mkdir(parents=True, exist_ok=True)
    keep = {f"{i:06d}" for i in cells.index}
    for old in out.glob("*.npz"):                        # samples from earlier, different cell tables
        if old.stem not in keep:
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)
    errors = []
    with ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(build_one, row, collection, COLLECTIONS[collection], datetime, days, margin,
                            choice, out, force): row for row in cells.itertuples()}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc=collection):
            try:
                job.result()
            except Exception as error:
                row = jobs[job]
                errors.append(dict(index=row.Index, grid_cell=row.grid_cell,
                                   error=f"{type(error).__name__}: {error}"[:200]))
    print(f"{len(list(out.glob('*.npz'))):,} samples in {out}, {len(errors):,} failed")
    return pd.DataFrame(errors, columns=["index", "grid_cell", "error"])
