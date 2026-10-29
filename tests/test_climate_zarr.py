"""ARCO-ERA5 reader: requests carry a stall limit, so a connection that stops sending is retried (no network)."""
import sys
from types import SimpleNamespace

import aiohttp

from majortom.climate.readers import zarr


def test_open_sets_a_stall_limit(monkeypatch):
    seen = {}
    fake = SimpleNamespace(open_zarr=lambda store, **kw: seen.update(kw) or "dataset")
    monkeypatch.setitem(sys.modules, "xarray", fake)
    zarr._open.cache_clear()
    try:
        assert zarr._open("gs://bucket/store.zarr") == "dataset"
    finally:
        zarr._open.cache_clear()
    timeout = seen["storage_options"]["requests_timeout"]
    assert isinstance(timeout, aiohttp.ClientTimeout) and timeout.total is None and timeout.sock_read == 120
