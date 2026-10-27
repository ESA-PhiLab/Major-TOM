"""Writing one raster group as a .rumi file with a compression profile (docs/RELEASE.md section 5).

    (bands, y, x) + transform + CRS + band names + time
      -> rumi.frames: 176 px frames, all bands of a tile in one frame
      -> graph: given (a release's cached choice), the profile's recipe, or chosen on this group's frames
      -> geozl.compress each frame -> rumi.write -> .rumi + header
      -> verify: read back, identical (lossless) or within the profile's bound, else raise
      -> RumiWritten(path, header, graph, error) for the collection metadata
Times: rumi records whole seconds, so an acquisition time is rounded down to the second (the exact time belongs
in the sample's STAC metadata); a (start, end) pair is written as an interval (yearly and static products).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine

from .profiles import choose_graph, profile

FRAME = 176                                        # as the COG tiles (docs/RELEASE.md 3.1)
PATTERN = "b (row h) (col w) -> row col (b h w)"   # all bands of a tile in one frame
SAMPLE_FRAMES = 4                                  # frames profiled when no graph is given


@dataclass(frozen=True)
class RumiWritten:
    path: Path
    header: bytes
    graph: str                                     # the graph actually used, for the collection metadata
    error: int | float | str | None


def _bound(error, data: np.ndarray) -> float:
    """The largest error a profile allows on `data`: 0, an absolute bound, or a percentage of each value."""
    if error is None or error == 0:
        return 0.0
    if isinstance(error, str) and error.endswith("%"):
        return float(error[:-1]) / 100 * float(np.abs(data).max())
    return float(error)


def _rumi_time(time):
    """rumi's time value: a whole-second datetime, or a (start, end) pair of them."""
    import pandas as pd

    def second(t):
        t = pd.Timestamp(t)
        return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t).floor("s").to_pydatetime()

    if isinstance(time, (tuple, list)):
        return tuple(second(t) for t in time)
    return second(time)


def write_rumi(path: str | Path, data: np.ndarray, transform: Affine, crs: str, bands: list[str], time,
               profile_name: str = "lossless", graph: str | None = None, verify: bool = True,
               categorical: bool = False) -> RumiWritten:
    """Write one raster group (bands, y, x) as a .rumi file; returns its header and the graph used.

    `graph` overrides the profile's choice (a release passes the graph it resolved once per raster group).
    `time` is the acquisition time (one step), or a (start, end) pair. `categorical` data (classes, codes)
    refuse lossy profiles. With `verify`, the file is read back and checked.
    """
    import geozl
    import rumi

    prof = profile(profile_name)
    if categorical and prof.error not in (None, 0):
        raise ValueError(f"profile {profile_name!r} is lossy; categorical data (classes, codes) must stay lossless")
    frames = rumi.frames(np.ascontiguousarray(data), PATTERN, tile_size=FRAME)
    if graph is None:
        listed = list(frames)
        step = max(1, len(listed) // SAMPLE_FRAMES)
        graph = choose_graph([f.data for f in listed[::step][:SAMPLE_FRAMES]], prof)
    built: dict = {}
    for frame in frames:
        shape = frame.data.shape
        if shape not in built:
            built[shape] = geozl.graph(frame.data, graph, error=prof.error)
        frame.compressed = geozl.compress(frame.data, graph=built[shape])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, header = rumi.write(str(path), frames, bands=list(bands), time=[_rumi_time(time)],
                           transform=tuple(transform)[:6], crs=crs)
    if verify:
        back = np.asarray(rumi.read(str(path), header))
        back = back[0] if back.ndim == 4 and back.shape[0] == 1 else back
        worst = float(np.abs(back.astype(np.float64) - data.astype(np.float64)).max())
        if worst > _bound(prof.error, data):
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{path.name}: read-back error {worst} exceeds profile {profile_name!r} "
                               f"(bound {prof.error}, graph {graph}); file removed")
    return RumiWritten(path, header, graph, prof.error)
