"""Major TOM: grid, metadata and samples for Earth observation datasets (arXiv 2402.12095).

The core (grid, metadata, samples) needs no deep-learning libraries. The torch `Dataset` and the
embedders load on first use and need the extras `majortom[torch]` / `majortom[embed]`.
"""
from __future__ import annotations

from . import _optional
from .grid import *           # noqa: F401,F403  Grid, get_utm_zone_from_latlng
from .metadata import *       # noqa: F401,F403  metadata_from_url, filter_metadata, read_row, filter_download
from .sample import *         # noqa: F401,F403  plot, read_tif_bytes, read_png_bytes

__version__ = "0.2.0.dev0"

# name -> (module that defines it, extra that provides its dependencies)
_LAZY = {
    "MajorTOM": ("majortom.dataset", "torch"),                 # the torch Dataset class
    "MajorTOM_Embedder": ("majortom.embedder", "embed"),
    "fragment_fn": ("majortom.embedder", "embed"),
    "fragment_unfold": ("majortom.embedder", "embed"),
    "crop_footprint": ("majortom.embedder", "embed"),
    "SigLIP_S2RGB_Embedder": ("majortom.embedder", "embed"),
    "DINOv2_S2RGB_Embedder": ("majortom.embedder", "embed"),
    "SSL4EO_S2L1C_Embedder": ("majortom.embedder", "embed"),
    "SSL4EO_S1RTC_Embedder": ("majortom.embedder", "embed"),
}


def __getattr__(name: str):
    """Load torch/embedder names on first use (PEP 562), so `import majortom` stays light."""
    if name in _LAZY:
        module, extra = _LAZY[name]
        return getattr(_optional.load(module, extra), name)
    raise AttributeError(f"module 'majortom' has no attribute {name!r}")
