"""Releasing Major TOM datasets: rumi payloads by default, COG as an option (docs/RELEASE.md).

Needs the `[release]` extra (rumi-eo; rasterio >= 1.5 for the COG profile's tile interleave, Python >= 3.12).
"""
from .cog import COG_PROFILE, write_cog
from .read import read, to_cog

__all__ = ["COG_PROFILE", "read", "to_cog", "write_cog"]
