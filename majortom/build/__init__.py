"""Build Major TOM samples from an archive, Planetary Computer by default (needs `pip install 'majortom[build]'`).

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

Which archive: `archive` names one of ARCHIVES (archives.py); it is recorded in every sample.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from rasterio.warp import transform_bounds
from tqdm.auto import tqdm

from ..cells import Cell, cells_from_points, cells_in   # noqa: F401  re-exported for convenience
from ..spec import MARGIN, cell_coverage, window
from .archives import ARCHIVES, Archive
from .collections import COLLECTIONS, Collection, Source, declared_scaling, find_asset, processing_baseline
from .backends import backend, retries
from .read import layout, read_asset
from .search import acquired, item_crs, same_pass, scenes, tiles_for

PROFILE = "native-v0"          # recorded in every sample: native grid if the CRS matches, else resampled
CANDIDATES = 5                 # scenes examined per sample


def _field(row, name: str):
    """A per-sample column's value, or None when the column is missing or empty."""
    value = getattr(row, name, None)
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value


def outside(period: tuple[str, str | None] | None, query: str | None) -> bool:
    """Whether a STAC datetime query lies wholly outside a source's period (first day, last day or None)."""
    if period is None or not query:
        return False
    naive = [pd.Timestamp(t) for t in query.split("/") if t not in ("", "..")]
    ends = [t.tz_convert(None) if t.tzinfo else t for t in naive]
    first = pd.Timestamp(period[0])
    last = pd.Timestamp(period[1]) + pd.Timedelta(days=1) if period[1] else None   # through the last day
    return max(ends) < first or (last is not None and min(ends) >= last)


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


def scene_cloud(tiles) -> float | None:
    """The catalogue's cloud cover of a scene's tiles, as a share; None if not given."""
    covers = [it.properties["eo:cloud_cover"] for it in tiles if it.properties.get("eo:cloud_cover") is not None]
    return float(np.mean(covers)) / 100 if covers else None


def borrowed_clouds(archive: Archive, collection: Collection, tiles, crs: str, win, res: float,
                    covered: np.ndarray, bbox) -> tuple[float | None, str, list[str]]:
    """Cloud share in the window from the mask of the same pass in `collection.clouds_from` (L1C: L2A SCL).

    The mask is read at `res`, the pixel size of `covered` (nearest neighbour). Returns (cloud share, cloud
    source, items used). Without a matching pass: the scene's cloud cover.
    """
    other = COLLECTIONS[collection.clouds_from]
    where = other.source(collection.clouds_from, archive.name)
    mates = [it for it in same_pass(archive, where, tiles, bbox) if item_crs(it) == crs] if where else []
    if not mates:
        return scene_cloud(tiles), "scene", []
    key = where.key(other.clouds[0])
    data, seen = read_asset(archive, mates, key, win, res, "nearest")
    clouds, fill = other.clouds[1](data[0])
    valid = covered & seen & ~fill
    return (float(clouds[valid].mean()) if valid.any() else 1.0), collection.clouds_from, [it.id for it in mates]


class Pick(NamedTuple):
    """One way of reading a scene, as examined by pick_scene."""
    tiles: list                 # STAC items, stitched in this order
    win: object                 # spec.Window in the read CRS
    res: dict                   # pixel size per band
    native: bool                # pixels copied unchanged
    cloud: float | None         # cloud share in the window
    nodata: float               # share of window pixels without data
    first: np.ndarray | None    # the first band, if already read while choosing (no own mask)
    first_seconds: float        # time that read took
    note: str                   # dates used
    cloud_source: str | None    # "own", the product borrowed from, "scene", or None
    cloud_items: list           # items the cloud share came from, if borrowed


def pick_scene(archive: Archive, source: Source, cell, collection, query, margin,
               choice: Choice) -> tuple[Pick, dict]:
    """The first acceptable scene in the order of `choice.prefer`, among the first few.

    Returns the pick and its timing in seconds: catalogue search, and scene choice (masks read, other searches).
    """
    start = time.perf_counter()
    own_mask = collection.clouds is not None
    mask_asset = collection.clouds[0] if own_mask else collection.assets[0]
    tried = []
    bbox = search_box(cell, margin)
    found, note = scenes(archive, source, collection, bbox, query, choice.prefer)
    searched = time.perf_counter()
    for scene in found[:CANDIDATES]:
        for crs, tiles in tiles_for(scene, cell, margin):   # same scene, another zone if this one is empty
            bands = present(collection, source, tiles)
            grid_assets = bands + ([mask_asset] if own_mask else [])
            win, res, native = layout(archive, cell, tiles[0], collection, source, grid_assets, margin, crs)
            method = "nearest" if own_mask else collection.resampling   # without a mask: the data
            read_start = time.perf_counter()
            data, covered = read_asset(archive, tiles, source.key(mask_asset), win, res[mask_asset], method)
            read_seconds = time.perf_counter() - read_start
            cloud, cloud_source, cloud_items = None, None, []
            if own_mask:                                # the mask also marks no-data pixels inside files
                clouds, fill = collection.clouds[1](data[0])
                covered &= ~fill
                cloud, cloud_source = (float(clouds[covered].mean()) if covered.any() else 1.0), "own"
            elif collection.fill is not None:            # no mask: missing pixels are stored as the fill value
                covered &= (data != collection.fill).any(axis=0)
            if not own_mask and collection.clouds_from:
                cloud, cloud_source, cloud_items = borrowed_clouds(archive, collection, tiles, crs, win,
                                                                   res[mask_asset], covered, bbox)
            nodata = 1 - covered.mean()
            tried.append(Pick(tiles, win, res, native, cloud, nodata, None if own_mask else data, read_seconds,
                              note, cloud_source, cloud_items))
            if choice.accepts(cloud, nodata):
                return tried[-1], timing(start, searched)
            if nodata <= choice.max_nodata:             # data is there, only the clouds are off: next scene
                break
    if not tried:
        raise LookupError(f"no scene found ({note})")
    if choice.strict:
        raise LookupError(f"none of {len(tried)} scenes has cloud in {choice.cloud} "
                          f"and no-data <= {choice.max_nodata}")
    best = min(tried, key=lambda p: (p.nodata > choice.max_nodata, choice.cloud_gap(p.cloud), p.nodata))
    return best, timing(start, searched)


def timing(start: float, searched: float) -> dict:
    """Seconds spent on the catalogue search and on choosing the scene, rounded for the sidecar."""
    return dict(search=round(searched - start, 3), scene_choice=round(time.perf_counter() - searched, 3))


def present(collection: Collection, source: Source, tiles) -> list[str]:
    """Bands to save from these tiles: all required ones, and the optional ones every tile has."""
    extra = [b for b in collection.optional if all(find_asset(it, source.asset(b)) for it in tiles)]
    return list(collection.assets) + extra


def band_names(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One 2-D plane per name; an asset with several bands becomes asset_1, asset_2, ..."""
    planes = {}
    for asset, array in arrays.items():
        for i, plane in enumerate(array):
            planes[asset if len(array) == 1 else f"{asset}_{i + 1}"] = plane
    return planes


def _same_request(sidecar: Path, archive: Archive, cell: Cell, query, margin: float, choice: Choice) -> bool:
    """Whether a saved sample was built from this archive for this cell, dates, margin and scene choice."""
    try:
        old = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return False
    return (old.get("archive", "planetary-computer"),   # samples from before archives: Planetary Computer
            old.get("grid_cell"), old.get("grid_km"), old.get("query", {}).get("datetime"),
            old.get("query", {}).get("margin"), old.get("choice")) == (
            archive.name, cell.name, cell.d, query, margin, json.loads(json.dumps(asdict(choice))))


def build_one(row, archive: Archive, name: str, collection: Collection, datetime, days, margin, choice: Choice,
              outdir: Path, force: bool) -> None:
    """Build and write one sample."""
    path = outdir / f"{row.Index:06d}.npz"
    cell = Cell(int(row.row), int(row.col), float(row.grid_km))
    margin = float(_field(row, "margin") if _field(row, "margin") is not None else margin)
    days = _field(row, "days") if _field(row, "days") is not None else days
    query = interval(_field(row, "datetime") or datetime, days)
    source = collection.source(name, archive.name)
    build_start = time.perf_counter()
    retries(reset=True)
    if path.exists():
        # reuse a sample on disk only if it was built for this cell, query and choice
        if not force and _same_request(path.with_suffix(".json"), archive, cell, query, margin, choice):
            return
        path.unlink()                                   # never leave an old sample under this index
        path.with_suffix(".json").unlink(missing_ok=True)

    with backend(archive).session(archive):             # credentials, if any, live only inside this block
        pick, seconds = pick_scene(archive, source, cell, collection, query, margin, choice)
        tiles, win, res, native, cloud, nodata = pick.tiles, pick.win, pick.res, pick.native, pick.cloud, pick.nodata
        arrays, band_seconds = {}, {}
        for asset in present(collection, source, tiles):
            if pick.first is not None and asset == collection.assets[0]:   # read while choosing the scene
                arrays[asset], band_seconds[asset] = pick.first, pick.first_seconds
                continue
            read_start = time.perf_counter()
            arrays[asset] = read_asset(archive, tiles, source.key(asset), win, res[asset], collection.resampling)[0]
            band_seconds[asset] = time.perf_counter() - read_start
    seconds["bands"] = {band: round(s, 3) for band, s in band_seconds.items()}
    seconds["total"] = round(time.perf_counter() - build_start, 3)
    seconds["retries"] = retries()                      # waits after failed requests (rate limits, drops)

    planes = band_names(arrays)
    offset = None
    if collection.offset:                               # value added to stored values, per band
        bands, rule = collection.offset
        offset = {b: rule(tiles[0], source.key(b)) for b in collection.assets if b in bands}
    finest = min(res[a] for a in arrays)
    scaling = {b: declared_scaling(tiles[0], source.key(b)) for b in arrays}
    scaling = {b: v for b, v in scaling.items() if v} or None
    meta = dict(
        index=int(row.Index), lon=float(row.lon), lat=float(row.lat),
        archive=archive.name, collection=name, source_collection=source.collection,
        platform=tiles[0].properties.get("platform"), scaling=scaling,          # physical = stored x scale + offset
        processing_baseline=processing_baseline(tiles[0]) if collection.offset else None, offset=offset,
        items=[it.id for it in tiles], datetime=acquired(tiles[0]).floor("us").isoformat(),
        cloud=cloud, cloud_source=pick.cloud_source, cloud_items=pick.cloud_items, nodata=float(nodata),
        crs=win.crs, epsg=int(win.crs.split(":")[1]),
        left=win.left, top=win.top, res=finest, size=win.pixels(finest), bands=list(planes),
        grid_cell=cell.name, grid_km=cell.d, window_m=win.side, cell_coverage=cell_coverage(cell, win),
        profile=PROFILE, resampled=[] if native else list(arrays),
        query=dict(datetime=query, margin=margin, dates_used=pick.note), choice=asdict(choice),
        timing=seconds,                                 # seconds: search, scene_choice, bands, total; retries
    )
    # sidecar first, then the arrays atomically: a chip on disk is always a complete pair
    path.with_suffix(".json").write_text(json.dumps(meta))
    staging = path.with_name(path.name + ".part")
    with open(staging, "wb") as handle:
        np.savez(handle, **planes)
    os.replace(staging, path)


def download(cells: pd.DataFrame, collection: str, datetime: str | None = None, days: float | None = None,
             margin: float = MARGIN, prefer: str = "clearest", cloud: tuple[float, float] | None = (0.0, 0.05),
             max_nodata: float = 0.01, strict: bool = False, archive: str = "planetary-computer",
             outdir: str | Path = "chips", workers: int | None = None, force: bool = False) -> pd.DataFrame:
    """Build one sample per row of `cells` from a Planetary Computer collection.

    datetime     dates to search ("start/end" or one time). Yearly products: selects the version.
                 Static products: ignored
    days         widen a single `datetime` by +- days
    margin       window margin as a fraction of the grid spacing (0.056: 10,560 m at 10 km)
    prefer       order scenes are tried: "clearest", "cloudiest", "nearest" (to the date) or "newest"
    cloud        accepted share of cloudy pixels in the window, (low, high); None accepts any
    max_nodata   accepted share of window pixels without data
    strict       True: a sample with no acceptable scene fails; False: the closest to the range is used
    archive      where to search and read, one of ARCHIVES
    workers      parallel samples; None: what the archive allows
    force        True: rebuild every sample. False: reuse a sample on disk if it was built for the same
                 cell, dates, margin and scene choice. Samples in outdir for rows not in `cells` are removed
    Returns the samples that failed, with the reason.
    """
    choice = Choice(prefer, tuple(cloud) if cloud is not None else None, max_nodata, strict)
    if prefer not in ("clearest", "cloudiest", "nearest", "newest"):
        raise ValueError(f"prefer must be clearest, cloudiest, nearest or newest, not {prefer!r}")
    if archive not in ARCHIVES:
        raise KeyError(f"unknown archive {archive!r}; pick one of {sorted(ARCHIVES)}")
    source = ARCHIVES[archive]
    with backend(source).session(source):                # missing credentials: one clear error, before any sample
        pass
    if collection not in COLLECTIONS:
        raise KeyError(f"unknown collection {collection!r}; pick one of {sorted(COLLECTIONS)}")
    product = COLLECTIONS[collection]
    if product.source(collection, archive) is None:
        raise KeyError(f"{archive} does not have {collection}; it is in {product.archives()}")
    if product.time == "scene" and outside(product.source(collection, archive).period, interval(datetime, days)):
        covering = [a for a in product.archives() if not outside(product.source(collection, a).period,
                                                                 interval(datetime, days))]
        raise ValueError(f"{archive} has {collection} for {product.source(collection, archive).period}, not "
                         f"{datetime}; archives that cover it: {covering or 'none'}")
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
    with ThreadPoolExecutor(workers or source.workers) as pool:
        jobs = {pool.submit(build_one, row, source, collection, COLLECTIONS[collection], datetime, days, margin,
                            choice, out, force): row for row in cells.itertuples()}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc=collection):
            try:
                job.result()
            except Exception as error:
                row = jobs[job]
                message = f"{type(error).__name__}: {error}"
                if source.refused and any(s in message for s in ("401", "403", "Access Denied", "AccessDenied")):
                    message = f"{source.refused}: {message}"
                errors.append(dict(index=row.Index, grid_cell=row.grid_cell, error=message[:300]))
    print(f"{len(list(out.glob('*.npz'))):,} samples in {out}, {len(errors):,} failed")
    return pd.DataFrame(errors, columns=["index", "grid_cell", "error"])
