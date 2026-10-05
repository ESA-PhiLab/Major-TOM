"""Alias package: `import MajorTOM` keeps working. The code lives in `majortom`.

As before, `from MajorTOM import *` also brings the torch `Dataset` class when torch is installed.
"""
import importlib.util as _util

import majortom as _majortom
from majortom import *  # noqa: F401,F403

if _util.find_spec("torch") is not None:          # old behaviour: Dataset exported when available
    from majortom.dataset import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_majortom, name)
