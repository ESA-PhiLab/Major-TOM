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


def test_viirs_tile(page):
    """VIIRS: the tile from its zip shard (one range request, zstd GeoTIFF), coloured, on the map by its own corners."""
    fields = thumbnail(page, "VIIRS")
    assert fields["Year"].endswith("(annual median)") and ".zip" in fields["File"]
    assert page.locator("#download-actions").is_hidden() and page.locator("#view-row").is_hidden()
    assert page.locator("#weather").is_hidden()
    corners = page.evaluate("sampleCorners()")
    assert all(abs(lon - LOCH_LOMOND[0]) < 0.2 and abs(lat - LOCH_LOMOND[1]) < 0.1 for lon, lat in corners)
    assert page.evaluate("globe.getPaintProperty('sample', 'raster-opacity')") == 1


def test_cursor_reports_coverage(page):
    page.wait_for_function("document.body.dataset.loaded === 'S2L2A'", timeout=120_000)
    page.evaluate(f"globe.jumpTo({{center: [{LOCH_LOMOND[0]}, {LOCH_LOMOND[1]}], zoom: 6}})")
    page.wait_for_function("globe.loaded()", timeout=60_000)
    page.mouse.move(*at(page, *LOCH_LOMOND))
    assert page.text_content("#cursor-value") == "yes, click to load"


def test_other_datasets_load_in_the_background(page):
    """After a click, the cell's samples in the other datasets load behind it: switching shows them at once."""
    thumbnail(page, "S2L2A")
    page.wait_for_timeout(15_000)                              # S2L1C, S1RTC, DEM footprints and thumbnails
    page.click("input[name=dataset][value=S1RTC]")
    page.wait_for_function("document.getElementById('pick-stats').textContent.includes('loaded before')", timeout=60_000)
    assert page.text_content("#pick-cell") == "624U_29L"


def test_open_the_image_full_window(page):
    thumbnail(page, "S2L2A")
    page.click("#open-image")
    page.wait_for_function("(() => { const i = document.getElementById('lightbox-img'); return !document.getElementById('lightbox')"
                           ".hidden && i.complete && i.naturalWidth > 0 && i.style.transform.includes('scale') })()", timeout=30_000)
    before = page.evaluate("document.getElementById('lightbox-img').style.transform")
    page.mouse.move(700, 450); page.mouse.wheel(0, -400)                   # zoom in about the cursor
    assert page.evaluate("document.getElementById('lightbox-img').style.transform") != before
    page.keyboard.press("Escape")
    assert page.evaluate("document.getElementById('lightbox').hidden")


# corners of the Loch Lomond S2L2A image (part_03814, row 299), from its B04 GeoTIFF's bounds with pyproj
GEOTIFF_CORNERS = [[-4.672272, 56.140694], [-4.500466, 56.1429], [-4.49674, 56.046975], [-4.66812, 56.044777]]


def test_show_the_image_on_the_map(page):
    thumbnail(page, "S2L2A")
    assert page.evaluate("Boolean(globe.getLayer('sample'))")                       # always on the map, opaque
    assert page.evaluate("globe.getPaintProperty('sample', 'raster-opacity')") == 1
    corners = page.evaluate("sampleCorners()")
    for (lon, lat), (lon0, lat0) in zip(corners, GEOTIFF_CORNERS):
        assert abs(lon - lon0) < 2e-5 and abs(lat - lat0) < 2e-5                         # under 2 m
    page.click("#pick-clear")
    assert not page.evaluate("Boolean(globe.getLayer('sample'))")


def test_weather_at_acquisition(page):
    thumbnail(page, "S2L2A")
    page.wait_for_function("document.getElementById('wx-stats').textContent.startsWith('Core-Weather:')", timeout=60_000)
    assert page.locator("#wx-hourly .wx-row").count() == 3 and page.locator("#wx-daily .wx-row").count() == 4
    temperature = page.locator("#wx-hourly .wx-value").first.text_content()
    assert temperature.endswith("°C") and -30 < float(temperature.split()[0]) < 30            # Scotland, any season
    assert page.locator("#wx-hourly path.wx-line").first.get_attribute("d").count("L") == 48   # 49 hours, no gap


def view(page, view_id, label):
    page.select_option("#view", view_id)
    page.wait_for_function(f"document.getElementById('pick-stats').textContent.startsWith({label!r})", timeout=120_000)
    page.wait_for_function("(() => { const t = document.getElementById('thumb'); return t.complete && t.naturalWidth > 0 })()")
    return page.evaluate("document.getElementById('thumb').naturalWidth")


def test_bands_and_composites(page):
    thumbnail(page, "S2L2A")
    page.wait_for_function("document.querySelectorAll('#view option').length > 10", timeout=60_000)
    options = page.locator("#view option").evaluate_all("o => o.map(x => x.value)")
    assert options[:2] == ["thumbnail", "rgb:B04,B03,B02"] and {"B01", "B04", "B12", "cloud_mask"} <= set(options)
    assert view(page, "rgb:B04,B03,B02", "True colour") == 1068
    assert view(page, "B01", "B01") == 178                                          # 60 m band at its own size
    assert view(page, "rgb:B12,B8A,B04", "SWIR") == 1068                            # 20 m and 10 m bands together
    assert page.evaluate("Boolean(globe.getLayer('sample'))")                       # the shown view on the map
    view(page, "B04", "B04")                                                         # a single band: values
    page.click("#open-image"); page.mouse.move(700, 450)
    page.wait_for_function("document.getElementById('lightbox-value').textContent.includes('pixel')", timeout=10_000)
    page.keyboard.press("Escape")


def test_s1_and_dem_views(page):
    thumbnail(page, "S1RTC")
    page.wait_for_function("document.querySelectorAll('#view option').length >= 4", timeout=60_000)
    assert view(page, "rgb:vv,vh,ratio", "VV, VH") == 1068                          # this S1 sample: 10 m (others 7.5 m)
    thumbnail(page, "DEM")
    page.wait_for_function("[...document.querySelectorAll('#view option')].some(o => o.value === 'rgb:DEM')", timeout=60_000)
    assert view(page, "rgb:DEM", "Elevation") == 356


def test_downloads(page, tmp_path):
    import io, json, zipfile
    thumbnail(page, "S2L2A")
    page.wait_for_function("document.querySelectorAll('#view option').length > 10", timeout=60_000)
    view(page, "B04", "B04")
    with page.expect_download(timeout=120_000) as d:
        page.click("#dl-band")
    path = tmp_path / d.value.suggested_filename; d.value.save_as(path)
    assert path.name.startswith("624U_29L_S2L2A_") and path.name.endswith("_B04.tif")
    assert path.read_bytes()[:4] == b"II*\x00" and path.stat().st_size > 1_000_000          # a TIFF, 10 m band
    with page.expect_download(timeout=180_000) as d:
        page.click("#dl-all")
    z = zipfile.ZipFile(io.BytesIO(Path(d.value.path()).read_bytes()))
    names = z.namelist()
    assert sum(n.endswith(".tif") for n in names) == 13 and any("thumbnail" in n for n in names)   # 12 bands, cloud_mask
    meta = json.loads(z.read("metadata.json"))
    assert meta["grid_cell"] == "624U_29L" and meta["row_group"] == 299 and meta["source_file"].endswith("part_03814.parquet")
    page.click("#dl-python")
    code = page.evaluate("window.lastSnippet")
    assert "part_03814.parquet" in code and "read_row_group(299" in code


def test_loading_flag_while_a_band_loads(page):
    thumbnail(page, "S2L2A")
    page.wait_for_function("document.querySelectorAll('#view option').length > 10", timeout=60_000)
    page.select_option("#view", "rgb:B08,B04,B03")
    page.wait_for_function("!document.getElementById('loader').hidden && document.getElementById('loader-bytes')"
                           ".textContent.includes(' of ')", timeout=30_000)
    assert "False colour" in page.text_content("#loader-text")
    page.wait_for_function("document.getElementById('pick-stats').textContent.startsWith('False colour')", timeout=120_000)
    assert page.evaluate("document.getElementById('loader').hidden")
