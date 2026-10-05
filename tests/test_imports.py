"""Packaging checks: the core imports without torch, and old `MajorTOM` imports keep working."""
import importlib.util
import subprocess
import sys

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None


def run(code: str) -> str:
    """Run code in a fresh interpreter, so earlier imports in this test session cannot hide problems."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def test_core_import_does_not_load_torch():
    out = run("import sys, majortom; print('torch' in sys.modules, hasattr(majortom, 'Grid'))")
    assert out.split() == ["False", "True"]


def test_old_import_styles_still_work():
    run("import MajorTOM; MajorTOM.Grid")
    run("from MajorTOM import *; Grid; filter_metadata; read_row")
    run("from MajorTOM.grid import *; Grid")
    run("from MajorTOM.metadata_helpers import metadata_from_url, filter_metadata, read_row, filter_download")


def test_old_module_names_inside_majortom():
    from majortom.metadata_helpers import filter_metadata
    from majortom.sample_helpers import read_tif_bytes
    assert callable(filter_metadata) and callable(read_tif_bytes)


@pytest.mark.skipif(HAS_TORCH, reason="checks the message shown when torch is missing")
def test_missing_extra_names_the_extra():
    import majortom
    with pytest.raises(ImportError, match=r"majortom\[embed\]"):
        majortom.MajorTOM_Embedder
    with pytest.raises(ImportError, match=r"majortom\[torch\]"):
        majortom.MajorTOM


def test_grid_still_works():
    from majortom import Grid
    grid = Grid(1000)
    assert len(grid.rows) == 19 and grid.points.shape[0] > 0
