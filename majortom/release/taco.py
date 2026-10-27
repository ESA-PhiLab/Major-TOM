"""Built Major TOM samples -> a TACO dataset (docs/RELEASE.md sections 6-7), as Core-DEM is laid out.

    built samples (.npz + .json, one product)
      -> groups() -> write_rumi per raster group into <stage>/<sample id>/<group>.rumi
         (lossless by default; one GeoZL graph per raster group, chosen on the first sample and reused)
      -> hilbert_order(): samples sorted along a Hilbert curve over cell centroids (order.py)
      -> taco-eo writer: Collection + Contract, one Sample per acquisition, added in that order;
         partition_size=max_bytes cuts the stream into ZIPs, so each partition is a compact run of neighbours
           sample level:   stac (STAC extension: proj_code / proj_shape / proj_transform / datetime -> centroid),
                           majortom (MajorTOM extension: majortom:code), source:* (from the sidecar),
                           geoenrich (optional; GeoEnrich extension)
           children level: rumi (Rumi extension: header, statistics), content:* (bands, nodata, offset, graph),
                           scaling (declared physical scaling, when the archive declares one)
      -> ZIP partitions + TACOCAT (taco-eo), validated
taco-eo does the container, staging, partitioning by size, catalogue, statistics and validation; this module adds
the raster groups, the rumi payloads and the writing order.
"""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from ..cells import cell_from_name
from .groups import RasterGroup, groups
from .order import hilbert_order
from .rumi import write_rumi

MAX_BYTES = 3 * 10**9      # partition size limit (taco-eo's partition_size); partitions end just under it


class Source(BaseModel):
    archive: str = Field(description="Archive the sample was read from")
    collection: str = Field(description="majortom.build collection")
    items: list[str] = Field(description="Source products (STAC item ids)")
    platform: str | None = Field(default=None, description="Satellite")
    processing_baseline: str | None = Field(default=None, description="Processing baseline (Sentinel-2)")
    cloud: float | None = Field(default=None, description="Cloud share of the window, 0-1")
    cell_coverage: float | None = Field(default=None, description="Share of the grid cell inside the window")
    acquired: str = Field(description="Acquisition time as recorded, to the microsecond (ISO 8601)")


class Content(BaseModel):
    bands: list[str] = Field(description="Bands in array order")
    nodata: float | None = Field(default=None, description="Value marking missing pixels")
    stored_offset: float | None = Field(default=None, description="Shift carried by stored values (Sentinel-2: +1000 "
                                                                  "from processing baseline 04.00)")
    categorical: bool = Field(description="Classes or codes: stored lossless")
    graph: str = Field(description="GeoZL graph used for this file")
    error: str | None = Field(default=None, description="Error bound of the compression profile (None: lossless)")


def sample_id(cell_name: str, d: float, acquired: str | None) -> str:
    """Core-DEM's cell code plus the acquisition time: MT10km_0624U_0029L_20210714T112119 (no time: cell only)."""
    cell = cell_from_name(cell_name, d)
    code = (f"MT{d:g}km_{abs(cell.row):04d}{'U' if cell.row >= 0 else 'D'}_"
            f"{abs(cell.col):04d}{'R' if cell.col >= 0 else 'L'}")
    if acquired is None:
        return code
    return f"{code}_{pd.Timestamp(acquired).tz_convert('UTC') if pd.Timestamp(acquired).tzinfo else pd.Timestamp(acquired):%Y%m%dT%H%M%S}"


def _when(acquired: str) -> datetime:
    t = pd.Timestamp(acquired)
    return (t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")).to_pydatetime()


def _scaling(group: RasterGroup):
    """taco's asset Scaling, only when the archive declared one (physical = stored x scale + offset)."""
    from taco.metadata.asset import Scaling

    s = group.scaling
    if not s or (s.get("scale") is None and s.get("offset") is None):
        return None
    n = len(group.bands)
    return Scaling(scale_factor=[s.get("scale") or 1.0] * n, scale_offset=[s.get("offset") or 0.0] * n)


def release(samples: Sequence[str | Path], out: str | Path, *, id: str, description: str, licenses: Sequence[str],
            providers: Sequence, title: str | None = None, keywords: Sequence[str] | None = None,
            profile: str = "lossless", max_bytes: int = MAX_BYTES, with_time: bool = True, stats: bool = True,
            geoenrich: Sequence[str] | None = None, stage: str | Path | None = None, overwrite: bool = False,
            progress: bool = False) -> Path:
    """Write built samples of one product as a partitioned TACO dataset; returns the path taco-eo reports.

    `out`: a .zip path (partitions + TACOCAT beside it) or a folder. `with_time=False` for products with one
    sample per cell (e.g. a DEM): ids are then the cell code alone, as in Core-DEM. `geoenrich`: variables for
    taco's GeoEnrich extension (needs network), e.g. ["elevation", "admin_countries"].
    """
    import geozl
    import rumi
    import taco
    from taco.metadata.asset import Scaling

    out = Path(out)
    stage = Path(stage) if stage else out.parent / f".{out.stem}-stage"
    stage.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, str] = {}
    records, structure = [], None
    for path in samples:
        found, side = groups(path)
        names = [g.name for g in found]
        if structure is None:
            structure = names
        elif names != structure:
            raise ValueError(f"{Path(path).name}: raster groups {names} differ from the first sample's {structure}")
        acquired = side.get("datetime")
        sid = sample_id(side["grid_cell"], side["grid_km"], acquired if with_time else None)
        files, size = [], 0
        for g in found:
            target = stage / sid / f"{g.name}.rumi"
            written = write_rumi(target, g.data, g.transform, g.crs, g.bands, acquired or "2000-01-01",
                                 profile_name="lossless" if g.categorical else profile, graph=graphs.get(g.name),
                                 categorical=g.categorical)
            graphs.setdefault(g.name, written.graph)
            files.append((g, written))
            size += target.stat().st_size
        records.append(dict(id=sid, grid_cell=side["grid_cell"], acquired=pd.Timestamp(acquired or "2000-01-01"),
                            bytes=size, side=side, files=files))
    if not records:
        raise ValueError("no samples given")
    table = pd.DataFrame(records)
    if table["id"].duplicated().any():
        raise ValueError(f"duplicate sample ids, e.g. {table.loc[table['id'].duplicated(), 'id'].iloc[0]}")
    table = table.loc[hilbert_order(table[["grid_cell", "acquired"]], records[0]["side"]["grid_km"])]

    sample_level = dict(stac=taco.extensions.STAC(), majortom=taco.extensions.MajorTOM(
                            dist_km=records[0]["side"]["grid_km"]), source=Source)
    if geoenrich:
        sample_level["geoenrich"] = taco.extensions.GeoEnrich(list(geoenrich))
    contract = taco.Contract(
        structure=[f"{name}.rumi" for name in structure],
        metadata=[taco.Level("sample", **sample_level),
                  taco.Level("children", rumi=taco.extensions.Rumi(header=True, stats=stats), content=Content,
                             scaling=Scaling | None)])
    collection = taco.Collection(
        contract=contract, id=id, title=title, description=description, licenses=list(licenses),
        providers=list(providers), keywords=list(keywords) if keywords else None,
        release={"payload": "rumi", "profile": profile, "graphs": graphs, "rumi": _version("rumi-eo"),
                 "geozl": getattr(geozl, "__version__", None) or _version("geozl"),
                 "partition_max_bytes": max_bytes, "sample_order": "hilbert (docs/RELEASE.md 7)",
                 "spec": "docs/SPEC.md (Major TOM v2 windows)"})
    with taco.open_writer(collection, out, partition_size=max_bytes, overwrite=overwrite,
                          progress=progress) as writer:
        for r in table.itertuples():
            side = r.side
            first = r.files[0][0]
            stac = taco.metadata.sample.STAC(proj_code=first.crs, proj_shape=first.data.shape[-2:],
                                             proj_transform=tuple(first.transform)[:6],
                                             datetime=_when(side["datetime"]) if side.get("datetime") else None)
            source = Source(archive=side["archive"], collection=side["collection"], items=list(side.get("items", [])),
                            platform=side.get("platform"), processing_baseline=side.get("processing_baseline"),
                            cloud=side.get("cloud"), cell_coverage=side.get("cell_coverage"),
                            acquired=side.get("datetime") or "")
            assets = []
            for g, w in r.files:
                groups_ = dict(content=Content(bands=g.bands, nodata=g.nodata, stored_offset=g.stored_offset,
                                               categorical=g.categorical, graph=w.graph,
                                               error=None if w.error is None else str(w.error)))
                if (scaling := _scaling(g)) is not None:              # only when the archive declares one
                    groups_["scaling"] = scaling
                assets.append(taco.Asset(w.path, path=f"{g.name}.rumi", metadata=taco.Metadata(**groups_)))
            writer.add(taco.Sample(id=r.id, assets=assets, metadata=taco.Metadata(
                stac=stac, source=source)))
        result = writer.run()
    report = taco.validate(result.path)
    if not report.ok:
        raise RuntimeError(f"taco.validate failed for {result.path}: {report}")
    shutil.rmtree(stage, ignore_errors=True)
    return Path(result.path)


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None
