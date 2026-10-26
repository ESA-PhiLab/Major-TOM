"""Releasing Major TOM datasets: rumi payloads by default, COG as an option (docs/RELEASE.md).

Needs the `[release]` extra (rumi-eo; rasterio >= 1.5 for the COG profile's tile interleave, Python >= 3.12).
"""
from .cog import COG_PROFILE, write_cog
from .profiles import RUMI_PROFILES, GeoZLProfile
from .read import read, to_cog
from .rumi import RumiWritten, write_rumi

__all__ = ["COG_PROFILE", "GeoZLProfile", "RUMI_PROFILES", "RumiWritten", "read", "to_cog", "write_cog", "write_rumi"]
