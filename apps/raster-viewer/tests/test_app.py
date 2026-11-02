"""The Raster Viewer in a headless browser, reading the Core images on Hugging Face (network: HF and jsDelivr).

    a local static server for apps/raster-viewer that answers byte ranges (as the Space does)
      -> Chromium (Playwright, software WebGL)
      -> each dataset's footprints (footprints/, built by build_index.py; skipped where absent: not in git)
      -> a click at Loch Lomond (624U_29L): the cell's sample and its thumbnail, decoded by the browser

    cd apps/raster-viewer && pytest -m network        (needs `pip install pytest playwright && playwright install chromium`)
"""
from __future__ import annotations

import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.network

APP = Path(__file__).resolve().parents[1]
ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
LOCH_LOMOND = (-4.62, 56.10)

if not (APP / "footprints" / "datasets.json").exists():
    pytest.skip("footprints/ not built (python build_index.py /data/Major-TOM footprints)", allow_module_level=True)


class Ranges(SimpleHTTPRequestHandler):
    """Static files, with Range: bytes=a-b and bytes=-n answered as 206."""
    def log_message(self, *a): pass

    def do_GET(self):
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = Path(self.translate_path(self.path))
        if not m or not path.is_file():
            return super().do_GET()
        size = path.stat().st_size
        a, b = (size - int(m[2]), size - 1) if not m[1] else (int(m[1]), min(int(m[2] or size - 1), size - 1))
        with open(path, "rb") as f:
            f.seek(max(a, 0)); body = f.read(b - max(a, 0) + 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {max(a, 0)}-{b}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Ranges, directory=str(APP)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/index.html"
    server.shutdown()


@pytest.fixture
def page(url):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(args=ARGS)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("window.globe && globe.loaded()", timeout=60_000)
        yield page
        assert not errors, errors
        browser.close()


def at(page, lon, lat):
    """Screen position of a point on the globe."""
    return page.evaluate(f"(() => {{ const p = globe.project([{lon}, {lat}]); const r = globe.getCanvas()"
                         f".getBoundingClientRect(); return [p.x + r.left, p.y + r.top] }})()")


def thumbnail(page, name):
    """Show dataset `name`, click Loch Lomond, wait for the sample's thumbnail; returns the panel's fields."""
    if page.evaluate("document.body.dataset.loaded") != name:
        page.click(f"input[name=dataset][value={name}]")
    page.wait_for_function(f"document.body.dataset.loaded === '{name}'", timeout=120_000)
    page.evaluate(f"globe.jumpTo({{center: [{LOCH_LOMOND[0]}, {LOCH_LOMOND[1]}], zoom: 6}})")
    page.wait_for_function("globe.loaded()", timeout=60_000)
    page.mouse.click(*at(page, *LOCH_LOMOND))
    page.wait_for_function("(() => { const t = document.getElementById('thumb'); return !t.hidden && t.complete"
                           " && t.naturalWidth > 0 })()", timeout=120_000)
    assert page.text_content("#pick-cell") == "624U_29L"
    return dict(zip(page.locator("#pick-meta dt").all_text_contents(), page.locator("#pick-meta dd").all_text_contents()))


def test_s2l2a_thumbnail_at_loch_lomond(page):
    fields = thumbnail(page, "S2L2A")
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d UTC", fields["Acquired"])
    assert fields["Product"].startswith("S2") and "Cloud cover" in fields
    assert page.text_content("#pick-stats").startswith("Thumbnail:")


def test_s1rtc_and_dem_thumbnails(page):
    assert thumbnail(page, "S1RTC")["Orbit"] in ("ascending", "descending")
    assert "File" in thumbnail(page, "DEM")


def test_cursor_reports_coverage(page):
    page.wait_for_function("document.body.dataset.loaded === 'S2L2A'", timeout=120_000)
    page.evaluate(f"globe.jumpTo({{center: [{LOCH_LOMOND[0]}, {LOCH_LOMOND[1]}], zoom: 6}})")
    page.wait_for_function("globe.loaded()", timeout=60_000)
    page.mouse.move(*at(page, *LOCH_LOMOND))
    assert page.text_content("#cursor-value") == "yes, click to load"
