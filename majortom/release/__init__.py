"""Releasing Major TOM datasets: rumi payloads by default, COG as an option (docs/RELEASE.md).

Needs the `[release]` extra (rumi-eo; rasterio >= 1.5 for the COG profile's tile interleave, Python >= 3.12).
"""
from .cog import COG_PROFILE, write_cog
from .groups import RasterGroup, groups
from .order import hilbert, hilbert_order
from .profiles import RUMI_PROFILES, GeoZLProfile
from .read import read, to_cog
from .rumi import RumiWritten, write_rumi
from .taco import release, sample_id

__all__ = ["COG_PROFILE", "GeoZLProfile", "RUMI_PROFILES", "RasterGroup", "RumiWritten", "groups", "hilbert", "hilbert_order",
           "read", "release", "sample_id", "to_cog", "write_cog", "write_rumi"]
