"""How an archive is searched and read: one module per access method, all with the same four functions.

    search(archive, collection, **query)              catalogue search -> STAC items (bbox, datetime, sortby, max_items)
    header(archive, item, key)                        CRS and pixel grid of one asset, without reading pixels
    read_tile(archive, item, key, win, res, method)   one tile onto the window grid -> (bands, covered pixels)
    session(archive)                                  context in which reads run; credentials exist only inside it

    stac         STAC API + GDAL: Planetary Computer, Earth Search, CDSE, NASA Earthdata

Scene choice, window layout and stitching tiles are the same for every backend (search.py, read.py). A new
archive reachable by STAC and GDAL is only an Archive entry (archives.py); another access method is one more
module here, named by Archive.backend.
"""
from __future__ import annotations

import importlib
import threading
import time
from types import ModuleType
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from ..archives import Archive

T = TypeVar("T")
_counts = threading.local()                             # retries in this thread, for the sample's timing


def backend(archive: Archive) -> ModuleType:
    """The module that searches and reads `archive`."""
    return importlib.import_module(f".{archive.backend}", __name__)


def available(archive: Archive) -> bool:
    """Whether the archive's credentials, if it needs any, are set up."""
    try:
        with backend(archive).session(archive):
            return True
    except PermissionError:
        return False


def retry(fn: Callable[[], T], attempts: int = 4) -> T:
    """Call `fn`, retrying with waits of 1, 2, 4 s (rate limits, dropped connections). Retries are counted."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            _counts.retries = retries() + 1
            time.sleep(2 ** i)


def retries(reset: bool = False) -> int:
    """Retries made in this thread so far; reset=True starts the count again."""
    if reset:
        _counts.retries = 0
    return getattr(_counts, "retries", 0)
