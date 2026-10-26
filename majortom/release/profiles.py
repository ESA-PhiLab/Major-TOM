"""rumi compression profiles: named GeoZL (OpenZL) graphs with an error bound (docs/RELEASE.md section 4).

    profile name -> GeoZLProfile(graph, error)
      graph given: used as is;  graph None: choose_graph runs geozl.profile on sample frames and takes the best ratio
    A release resolves each raster group's graph once and records it (with the bound and library versions), so
    files can be decoded and re-encoded later. New profiles (e.g. Cesar's lossy ones) are new entries here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeoZLProfile:
    """A named compression profile: a GeoZL graph recipe (or None to profile one) and an error bound."""
    graph: str | None                     # e.g. "planar>zigzag>transpose>entropy"; None: chosen by geozl.profile
    error: int | float | str | None       # None: lossless; 2: absolute, in data units; "1%": relative
    description: str


RUMI_PROFILES = {
    "lossless": GeoZLProfile(None, None, "exact; graph chosen per raster group"),
    "near-lossless-1": GeoZLProfile(None, 1, "every pixel within 1 unit"),
    "near-lossless-2": GeoZLProfile(None, 2, "every pixel within 2 units"),
}


def profile(name: str) -> GeoZLProfile:
    """The profile called `name`; a clear error listing the valid names otherwise."""
    if name not in RUMI_PROFILES:
        raise ValueError(f"unknown rumi profile {name!r}; available: {', '.join(RUMI_PROFILES)}")
    return RUMI_PROFILES[name]


def choose_graph(frames: list[np.ndarray], prof: GeoZLProfile) -> str:
    """The profile's graph, or the graph geozl.profile ranks best (by total size) over these frames.

    Frames are (bands, h, w) arrays as rumi.frames makes them; a few from the first samples of a raster group are
    enough. Every candidate is ranked by the bytes it gives summed over all frames, at the profile's bound.
    """
    if prof.graph is not None:
        return prof.graph
    import geozl

    total: dict[str, int] = {}
    for frame in frames:
        for row in geozl.profile(frame, error=prof.error, reps=1):
            total[row["graph"]] = total.get(row["graph"], 0) + row["bytes"]
    return min(total, key=total.get)
