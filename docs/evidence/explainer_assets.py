"""Images for the "How Major TOM works" explainer on asterisk.coop (tools-pages/major-tom.md): one cell, four sources.

    cell 624U_29L (Loch Lomond) -> Core v1 window (10,680 m, EPSG:32630)
      -> majortom.build, window="v1", Planetary Computer: sentinel-2-l2a, sentinel-1-rtc, cop-dem-glo-30
      -> AlphaEarth 2024 (tge-labs/aef COGs on source.coop, 64 bands, int8): the window at 10 m -> PCA to RGB
      -> s2.jpg, s1.jpg, dem.jpg, alphaearth.jpg (534 px, half the 10 m pixels)
    the Sentinel-2 tile of the chosen scene, its "visual" COG read at an overview -> scene.jpg (900 px)
      -> explainer.json: the cell's edges, the window and scene.jpg's corners as lon/lat, dates, sources
    Natural Earth 1:110m land -> land.json (rings rounded to 0.1°), the globe's land fill

    python docs/evidence/explainer_assets.py <out folder>     (Slurm CPU job, .venv311)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from PIL import Image
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from majortom import build                                        # noqa: E402
from majortom.cells import cell_from_name                          # noqa: E402
from majortom.spec import legacy_window                            # noqa: E402
from pyproj import Transformer                                     # noqa: E402

CELL, CRS, EPSG = "624U_29L", "EPSG:32630", 32630
DATES = {"sentinel-2-l2a": "2024-06-01/2024-08-31", "sentinel-1-rtc": "2024-06-01/2024-06-30", "cop-dem-glo-30": None}
AEF_TILES = Path(__file__).resolve().parents[2] / "apps/raster-viewer/footprints/AEF-tiles.parquet"
AEF_FILES = "https://data.source.coop/tge-labs/aef/v1/annual/"
SIDE = 534                                                         # output pixels per side


def stretch(a: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Percentile stretch to 0-255 over all channels together (keeps the colour balance)."""
    p, q = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
    return np.clip((a - p) / (q - p) * 255, 0, 255).astype(np.uint8)


def save(rgb: np.ndarray, path: Path, size: int = SIDE) -> None:
    Image.fromarray(rgb).resize((size, size), Image.LANCZOS).save(path, quality=88)


def built(out: Path) -> dict[str, tuple[dict[str, np.ndarray], dict]]:
    """Build the cell from each collection with Core v1's window; returns collection -> (bands, sidecar)."""
    cells = build.cells_from_points(pd.DataFrame({"lon": [-4.6], "lat": [56.1]}), d=10)
    assert cells.grid_cell.iloc[0] == CELL
    found = {}
    for collection, dates in DATES.items():
        errors = build.download(cells, collection, dates, window="v1", outdir=out / "chips",
                                **({"cloud": None, "prefer": "nearest"} if collection == "sentinel-1-rtc" else {}))
        assert errors.empty, errors
        stem = out / "chips" / collection / "000000"
        found[collection] = (dict(np.load(f"{stem}.npz")), json.loads(Path(f"{stem}.json").read_text()))
    return found


def sentinel2(bands: dict) -> np.ndarray:
    return stretch(np.stack([bands[b].astype(np.float32) for b in ("B04", "B03", "B02")], -1), 1, 99)


def sentinel1(bands: dict) -> np.ndarray:
    """VV, VH and VV/VH in dB as red, green, blue: each channel stretched on its own."""
    vv, vh = (10 * np.log10(np.maximum(bands[b].astype(np.float32), 1e-4)) for b in ("vv", "vh"))
    return np.stack([stretch(c, 2, 98) for c in (vv, vh, vv - vh)], -1)


def dem(bands: dict) -> np.ndarray:
    """Hillshade (sun from the north-west, 45°) blended with elevation: grey, light is high."""
    z = bands["DEM"].astype(np.float32)
    gy, gx = np.gradient(z, 30.0)                                  # 30 m DEM pixels
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    grey = 0.6 * stretch(shade, 1, 99) + 0.4 * stretch(z, 1, 99)
    return np.repeat(grey.astype(np.uint8)[..., None], 3, -1)


def alphaearth(window, year: int = 2024) -> np.ndarray:
    """The window's 64 AlphaEarth bands at 10 m from the year's COG tile, as the first three PCA components."""
    tiles = pq.read_table(AEF_TILES, filters=[("year", "=", year), ("epsg", "=", EPSG)]).to_pandas()
    left, bottom, right, top = window.bounds
    inside = tiles[(tiles.west <= left) & (tiles.west + 81_920 >= right) & (tiles.north >= top) &
                   (tiles.north - 81_920 <= bottom)]
    assert len(inside) == 1, f"window not inside one tile: {len(inside)}"
    with rasterio.open(AEF_FILES + inside.path.iloc[0]) as src:
        inv = ~src.transform                                       # south-up files (row 0 in the south): from_bounds
        (c0, r0), (c1, r1) = inv * (left, top), inv * (right, bottom)   # refuses them, so the window is computed here
        win = Window(round(min(c0, c1)), round(min(r0, r1)), round(abs(c1 - c0)), round(abs(r1 - r0)))
        data = src.read(window=win).astype(np.float32)             # (64, rows, cols); flipped below if south-up
        if src.transform.e > 0:
            data = data[:, ::-1, :]
    valid = (data != -128).all(0)
    x = data[:, valid].T
    x -= x.mean(0)
    _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 50_000)], full_matrices=False)
    pcs = np.full(valid.shape + (3,), np.nan, np.float32)
    pcs[valid] = x @ vt[:3].T
    return np.stack([stretch(pcs[..., i], 2, 98) for i in range(3)], -1)


def scene(item_id: str, window, out: Path) -> dict:
    """The Sentinel-2 tile's true-colour COG at an overview, and where the window and cell sit on it (fractions)."""
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    item = cat.get_collection("sentinel-2-l2a").get_item(item_id)
    with rasterio.open(item.assets["visual"].href) as src:
        assert src.crs.to_epsg() == EPSG
        rgb = src.read(out_shape=(3, 900, 900)).transpose(1, 2, 0)
        b = src.bounds
    Image.fromarray(rgb).save(out / "scene.jpg", quality=85)
    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
    lonlat = lambda x, y: [round(v, 6) for v in to_lonlat(x, y)]   # noqa: E731
    left, bottom, right, top = window.bounds
    return {"item": item_id, "tile_km": round((b.right - b.left) / 1000, 1),
            # top-left, top-right, bottom-left corners as lon/lat: the page places the image by these three points
            "corners": [lonlat(b.left, b.top), lonlat(b.right, b.top), lonlat(b.left, b.bottom)],
            "window": [lonlat(left, top), lonlat(right, top), lonlat(right, bottom), lonlat(left, bottom)]}


def land(out: Path) -> None:
    """Natural Earth 1:110m land as lon/lat rings rounded to 0.1° -> land.json (the globe's land fill)."""
    import urllib.request

    raw = out / "naturalearth"                                     # downloaded data in its own folder
    raw.mkdir(exist_ok=True)
    src = raw / "ne_110m_land.geojson"
    if not src.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
                                   "ne_110m_land.geojson", src)
    rings = []
    for f in json.loads(src.read_text())["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            ring = [[round(x, 1), round(y, 1)] for x, y in poly[0]]
            ring = [p for i, p in enumerate(ring) if i == 0 or p != ring[i - 1]]
            if len(ring) >= 4:
                rings.append(ring)
    (out / "land.json").write_text(json.dumps(rings, separators=(",", ":")))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cell = cell_from_name(CELL)
    window = legacy_window(cell, CRS)
    found = built(out)
    images = {"s2.jpg": sentinel2(found["sentinel-2-l2a"][0]), "s1.jpg": sentinel1(found["sentinel-1-rtc"][0]),
              "dem.jpg": dem(found["cop-dem-glo-30"][0]), "alphaearth.jpg": alphaearth(window)}
    for name, rgb in images.items():
        save(rgb, out / name)
    s2_side = found["sentinel-2-l2a"][1]
    land(out)
    info = {"cell": CELL, "lon": round(cell.centroid[0], 4), "lat": round(cell.centroid[1], 4),
            "edges": {"west": cell.lon, "south": cell.lat, "east": cell.lon + cell.dlon, "north": cell.lat + cell.dlat},
            "window_m": 10_680, "crs": CRS,
            "dates": {"s2": s2_side["datetime"][:10], "s1": found["sentinel-1-rtc"][1]["datetime"][:10],
                      "alphaearth": "2024"},
            "sources": {"s2": "Sentinel-2 L2A (Copernicus)", "s1": "Sentinel-1 RTC (Copernicus; Planetary Computer)",
                        "dem": "Copernicus DEM GLO-30", "alphaearth": "AlphaEarth embeddings (tge-labs/aef, CC-BY 4.0)"},
            "scene": scene(s2_side["items"][0], window, out)}       # the build's first tile: it holds the whole window
    (out / "explainer.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
