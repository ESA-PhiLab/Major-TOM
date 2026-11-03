"""Every local module is imported by app.js itself (no network).

On the private Space the viewer runs in an iframe, where the login cookie is not sent; a file is served only when its
URL or its referer carries the ?__sign= token. Only index.html and app.js carry it, so a module imported solely from
another module (referer without the token) gets a 401 and the whole app fails to start.
"""
from __future__ import annotations

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
LOCAL = re.compile(r"""^import\s+(?:[^'"]*?\s+from\s+)?['"]\./([\w.-]+\.js)['"]""", re.M)


def local_imports(path: Path) -> set[str]:
    """The ./x.js modules a file imports."""
    return set(LOCAL.findall(path.read_text()))


def test_app_imports_every_local_module():
    """A module imported anywhere is also imported by app.js, so its first request has app.js as referer."""
    imported = set().union(*(local_imports(p) for p in APP.glob("*.js") if p.name != "app.js"))
    missing = imported - local_imports(APP / "app.js")
    assert not missing, f"import these in app.js too: {sorted(missing)}"
