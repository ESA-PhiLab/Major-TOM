"""Read georeferencing headers of real products over Snowbird: CRS, pixel size, origin vs the 60 m lattice."""
import planetary_computer
import rasterio
from pystac_client import Client

PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SNOWBIRD = (-111.70, 40.56, -111.62, 40.60)
ASSET = {"sentinel-2-l2a": ["B04", "B05", "B01"], "landsat-c2-l2": ["red", "lwir11"],
         "sentinel-1-rtc": ["vv"], "cop-dem-glo-30": ["data"]}
DATES = {"sentinel-2-l2a": "2026-09-01/2026-10-05", "landsat-c2-l2": "2026-08-01/2026-10-05",
         "sentinel-1-rtc": "2026-09-01/2026-10-05", "cop-dem-glo-30": None}
client = Client.open(PC, modifier=planetary_computer.sign_inplace)
for coll, keys in ASSET.items():
    items = list(client.search(collections=[coll], bbox=SNOWBIRD, datetime=DATES[coll], max_items=3).items())
    for item in items[:3]:
        for k in keys:
            with rasterio.open(item.assets[k].href) as src:
                t = src.transform
                print(f"{coll:15s} {item.id[:44]:44s} {k:7s} {str(src.crs):11s} res {src.res[0]:.8g} "
                      f"origin ({t.c:.6g}, {t.f:.6g})  mod60 ({t.c % 60:g}, {t.f % 60:g})  mod30 ({t.c % 30:g}, {t.f % 30:g})")
