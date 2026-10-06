"""Alias: `import MajorTOM` keeps working; the code lives in the `majortom` package.

This is one file rather than a folder: on macOS and Windows a `MajorTOM/` folder would be the same
path as `majortom/`, and the two would overwrite each other.

    import MajorTOM                                   # majortom's names
    from MajorTOM.metadata_helpers import read_row    # old submodule paths, served from majortom
"""
import importlib
import importlib.abc
import importlib.util
import sys

from majortom import *  # noqa: F401,F403

if importlib.util.find_spec("torch") is not None:   # as before: the Dataset class when torch is installed
    from majortom.dataset import *  # noqa: F401,F403

_OLD_NAMES = {
    "MajorTOM.grid": "majortom.grid",
    "MajorTOM.metadata_helpers": "majortom.metadata",
    "MajorTOM.sample_helpers": "majortom.sample",
    "MajorTOM.MajorTOMDataset": "majortom.dataset",
    "MajorTOM.embedder": "majortom.embedder",
}


class _OldNames(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves the old submodule paths from the new modules, importing each only when asked for."""

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self) if name in _OLD_NAMES else None

    def create_module(self, spec):
        return importlib.import_module(_OLD_NAMES[spec.name])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _OldNames())
__path__ = []                                        # lets `import MajorTOM.grid` treat MajorTOM as a package


def __getattr__(name: str):
    import majortom
    return getattr(majortom, name)
