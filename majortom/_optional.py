"""Import parts of majortom that need an optional extra, with an error that names the extra."""
from __future__ import annotations

import importlib
from types import ModuleType


def load(module: str, extra: str) -> ModuleType:
    """Import `module`; if one of its dependencies is missing, say which `pip install` fixes it."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise ImportError(f"{module} needs '{e.name}', which is not installed. "
                          f"Install it with: pip install 'majortom[{extra}]'") from e
