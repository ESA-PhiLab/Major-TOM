"""Place names for the climate globe: cities and landmarks from Natural Earth (public domain), as one small file.

    Natural Earth 10m (github.com/nvkelso/natural-earth-vector, geojson/):
      ne_10m_populated_places_simple             cities: name, position, population (pop_max), national capital
      ne_10m_geography_regions_elevation_points  mountains, depressions, passes: name, position, elevation
      ne_10m_geography_regions_points            waterfalls and poles (islands and capes are left out)
      -> places.json: {"columns": [...], "rows": [[kind, name, lon, lat, size], ...]}, largest first in one order:
         a city by population, a landmark as 500 people a metre (a 4,000 m peak ranks with a city of 2 million);
         the page labels in that order and skips overlaps
         kind: "capital", "city", "peak", "low" (depression), "pass", "falls", "pole"; size: population or metres

    python apps/climate-globe/build_places.py <folder with the three .geojson files>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "places.json"
LANDMARK = {"mountain": "peak", "spot elevation": "peak", "depression": "low", "pass": "pass",
            "waterfall": "falls", "pole": "pole"}


def features(folder: Path, name: str) -> list[dict]:
    return json.loads((folder / f"{name}.geojson").read_text())["features"]


def build(folder: Path) -> None:
    cities, landmarks = [], []
    for f in features(folder, "ne_10m_populated_places_simple"):
        p = f["properties"]
        kind = "capital" if p.get("adm0cap") == 1 else "city"
        cities.append([kind, p["name"], round(p["longitude"], 4), round(p["latitude"], 4), int(p.get("pop_max") or 0)])
    for name in ("ne_10m_geography_regions_elevation_points", "ne_10m_geography_regions_points"):
        for f in features(folder, name):
            p, (lon, lat) = f["properties"], f["geometry"]["coordinates"][:2]
            kind = LANDMARK.get(p.get("featurecla"))
            if kind and p.get("name"):
                landmarks.append([kind, p["name"], round(lon, 4), round(lat, 4), int(p.get("elevation") or 0)])
    rank = lambda r: r[4] if r[0] in ("capital", "city") else abs(r[4]) * 500
    OUT.write_text(json.dumps({"source": "Natural Earth 10m (public domain)",
                               "columns": ["kind", "name", "lon", "lat", "size"],
                               "rows": sorted(cities + landmarks, key=lambda r: -rank(r))}, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(cities):,} cities, {len(landmarks):,} landmarks -> {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
