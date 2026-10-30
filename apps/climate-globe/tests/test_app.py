"""The climate globe in a headless browser, reading the published data (network: source.coop and jsDelivr).

    a local static server for apps/climate-globe -> Chromium (Playwright, software WebGL)
      -> series: a globe month from overviews/, a zoomed month from series/
      -> a click at Loch Lomond (624U_29L): CPC normals and the whole by_cell/ series, against values in the files
      -> keys (left/right change the month, not the map), cursor off the globe, place names from zoom 5
      -> previews of the four layers (skipped where images/ is absent: the images are not in git)

    cd apps/climate-globe && pytest -m network        (needs `pip install pytest playwright && playwright install chromium`)
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.network

APP = Path(__file__).resolve().parents[1]
ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
LOCH_LOMOND = (-4.62, 56.10)
LAYERS = {"cpc-precip": "CPC, rain gauges", "imerg-precip": "IMERG, satellites", "cpc-tmax": "CPC, stations",
          "cams-pm25": "CAMS, reanalysis"}


@pytest.fixture(scope="module")
def url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=APP,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    yield f"http://127.0.0.1:{port}/index.html"
    server.terminate()


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


def loaded(page, prefix, timeout=300_000):
    page.wait_for_function(f"(document.body.dataset.loaded || '').startsWith('{prefix}')", timeout=timeout)


def series(page):
    """Switch to the series read live: the tests that are not about previews do not need images/."""
    page.click("[data-mode=live]")
    loaded(page, "cpc-precip/live")


def at(page, lon, lat):
    """Screen position of a point on the globe."""
    return page.evaluate(f"(() => {{ const p = globe.project([{lon}, {lat}]); const r = globe.getCanvas()"
                         f".getBoundingClientRect(); return [p.x + r.left, p.y + r.top] }})()")


def test_series_month_and_cell(page):
    page.click("[data-mode=live]")
    loaded(page, "cpc-precip/live")
    page.wait_for_function("document.body.dataset.loaded.endsWith('/2')", timeout=300_000)    # globe: overview
    assert page.text_content("#stat-source").startswith("Parquet overview")
    page.evaluate(f"globe.jumpTo({{center: [{LOCH_LOMOND[0]}, {LOCH_LOMOND[1]}], zoom: 4}})")
    page.wait_for_function("document.body.dataset.loaded.endsWith('/0')", timeout=300_000)    # zoomed: series/
    page.mouse.click(*at(page, *LOCH_LOMOND))
    page.wait_for_function("document.getElementById('clim-stats').textContent.includes('row group')", timeout=120_000)
    assert page.text_content("#pick-cell") == "624U_29L"
    normals = page.evaluate("pickState().clim.mean")
    assert normals[0] == pytest.approx(122.892, abs=1e-3) and normals[6] == pytest.approx(78.631, abs=1e-3)
    page.click("#series-load")
    page.wait_for_function("document.getElementById('series-stats').textContent.startsWith('By cell:')", timeout=120_000)
    series = page.evaluate("pickState().series")
    assert len(series) == 564
    assert series[0] == pytest.approx(129.368, abs=1e-3)          # 1979-01, as in series/year=1979/01.parquet
    assert series[324] == pytest.approx(82.015, abs=1e-3)         # 2006-01


def test_keys_change_the_month_not_the_map(page):
    series(page)
    page.mouse.click(*at(page, 12, 22))                            # focus the map
    centre = page.evaluate("globe.getCenter().toArray()")
    month = page.text_content("#time-label")
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(500)
    assert page.text_content("#time-label") != month
    assert page.evaluate("globe.getCenter().toArray()") == centre


def test_cursor_only_on_the_globe(page):
    series(page)
    page.mouse.move(*at(page, 10, 30))
    page.wait_for_timeout(300)
    assert page.text_content("#cursor-cell") != "–"
    page.mouse.move(20, 120)                                       # the space beside the globe
    page.wait_for_timeout(300)
    assert page.text_content("#cursor-cell") == "–"


def test_place_names_from_zoom_five(page):
    series(page)
    names = lambda: page.evaluate("[...document.querySelectorAll('#places .place span')].map(s => s.textContent)")
    assert names() == []                                           # globe scale: none
    page.evaluate("globe.jumpTo({center: [9.5, 46.3], zoom: 6})")
    page.wait_for_timeout(2000)
    shown = names()
    assert 10 <= len(shown) <= 80 and any(n.startswith("Mont Blanc") for n in shown)
    page.uncheck("#show-places")
    page.wait_for_timeout(500)
    assert names() == []


@pytest.mark.skipif(not (APP / "images").exists(), reason="preview images are not in git (build_images.py)")
def test_previews_of_every_layer(page):
    loaded(page, "cpc-precip/preview")
    for layer, label in LAYERS.items():
        page.click(f"text={label}")
        loaded(page, f"{layer}/preview")
        assert page.text_content("#status") == ""
