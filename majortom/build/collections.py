"""Products the builder knows: bands, cloud masks, how to resample them, and where each archive keeps them.

A product's band names are the same from every archive (Sentinel-2: B01..B12, SCL); a Source maps them to
the archive's collection and asset keys. Collections and cloud masks follow asterisk-labs/taco
examples/workshop.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pystac

MaskFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]


def s2_clouds(scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sentinel-2 scene classes: (cloud shadow, medium and high cloud, cirrus), (no data)."""
    return np.isin(scl, [3, 8, 9, 10]), scl == 0


def landsat_clouds(qa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Landsat QA_PIXEL: (bits 1-4: dilated cloud, cirrus, cloud, shadow), (bit 0: fill)."""
    return (qa & 0b11110) > 0, (qa & 1) > 0


def hls_clouds(fmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HLS Fmask: (bits 1-3: cloud, adjacent to cloud, shadow), (255: fill)."""
    return (fmask & 0b1110) > 0, fmask == 255


def find_asset(item: pystac.Item, name: str) -> str | None:
    """The item's asset key for `name`: the exact key, else the one key ending in "_name" or "/name" (OPERA keys
    carry the item id), preferring an https link over an s3:// copy (readable only inside AWS)."""
    if name in item.assets:
        return name
    keys = [k for k in item.assets if k.endswith(f"_{name}") or k.endswith(f"/{name}")]
    https = [k for k in keys if not item.assets[k].href.startswith("s3://")]
    found = https or keys
    return found[0] if len(found) == 1 else None


def declared_band(item: pystac.Item, key: str) -> dict:
    """The raster:bands entry for an asset key ("ASSET#n": band n of a multi-band asset), or {}."""
    asset, _, n = key.partition(":")[0].partition("#")
    fields = item.assets[find_asset(item, asset)].extra_fields
    bands = fields.get("raster:bands") or fields.get("bands") or []
    index = int(n) - 1 if n else 0
    return bands[index] if index < len(bands) else {}


def declared_scaling(item: pystac.Item, key: str) -> dict | None:
    """Scale and offset the archive declares for an asset (physical = stored x scale + offset), or None."""
    band = declared_band(item, key)
    scale, offset = band.get("scale"), band.get("offset")
    return None if scale is None and offset is None else {"scale": scale, "offset": offset}


def processing_baseline(item: pystac.Item) -> str | None:
    """Sentinel-2 processing baseline, e.g. "05.00": s2:processing_baseline, or processing:version (CDSE)."""
    return item.properties.get("s2:processing_baseline") or item.properties.get("processing:version")


def declared_or_baseline_offset(item: pystac.Item, key: str) -> float | None:
    """Sentinel-2 reflectance offset in stored values: as declared by the archive, else from the baseline.

    Declared: raster:bands offset and scale (Earth Search: offset -0.1 at scale 0.0001, i.e. +1000).
    Not declared (Planetary Computer, CDSE): processing baseline 04.00 or later adds 1000 (from 2022-01-25).
    """
    band = declared_band(item, key)
    offset, scale = band.get("offset"), band.get("scale")
    if offset is not None and scale:
        return float(round(-offset / scale))
    baseline = processing_baseline(item)
    return None if baseline is None else (1000.0 if baseline >= "04.00" else 0.0)


@dataclass(frozen=True)
class Source:
    """One product in one archive: its collection id, the asset key of each band (default: the band name), and
    the time it covers as the archive's catalogue declares it: (first day, last day or None if ongoing)."""
    collection: str
    assets: dict[str, str] = field(default_factory=dict)
    period: tuple[str, str | None] | None = None

    def key(self, band: str) -> str:
        return self.assets.get(band, band)

    def asset(self, band: str) -> str:
        """The item asset holding `band` (a key "VNIR#2" is band 2 of asset VNIR; "otci:OTCI" variable OTCI of
        the netCDF asset otci)."""
        return self.key(band).partition("#")[0].partition(":")[0]


@dataclass(frozen=True)
class Collection:
    """What the builder needs to know about one collection.

    assets      bands to save
    time        "scene": dates select acquisitions; "yearly": dates select the version (the newest if none
                falls in them); "static": one version, dates ignored
    clouds      (mask asset, function giving cloud and no-data pixels) used to rank scenes; not saved
    resampling  method when the data are not already on the cell's pixel grid: "nearest" for classes,
                "bilinear" for continuous values
    res_m       pixel size in metres when the product is in latitude/longitude: one value, or per asset
    clouds_from product whose cloud mask, from the same pass, ranks this product's scenes. Sentinel-2 L1C
                has no mask and borrows the L2A SCL: per candidate scene one extra catalogue search and one
                read of SCL over the window at the coarsest band's pixel size (60 m). Without a matching pass
                the catalogue's whole-scene cloud cover is used. Recorded per sample (cloud_source)
    sources     archive name -> Source; None: Planetary Computer, under the product's own name
    offset      bands whose stored values carry an offset, and the function giving it (item, asset key)
    kind        what the product measures, for grouping in docs and benchmarks: "optical", "radar",
                "elevation", "land cover", "biomass", "water", "temperature" or "night lights"
    optional    bands saved when a scene has them (Landsat: coastal, OLI only)
    native      False: always resample to res_m, even in the cell's CRS (ASTER: pixels neither square nor
                multiples of each other, 29.989 x 29.988 m and 89.904 m; OLCI: a swath on a stand-in grid)
    fill        stored value meaning "no data" in files without a mask (Sentinel-2 L1C: 0 outside the swath)
    period      time covered when `sources` is None (Planetary Computer only); static products: none,
                their one version is used whatever the dates
    """
    assets: tuple[str, ...]
    time: str = "scene"
    clouds: tuple[str, MaskFn] | None = None
    resampling: str = "bilinear"
    res_m: float | dict[str, float] | None = None
    clouds_from: str | None = None
    sources: dict[str, Source] | None = None
    offset: tuple[tuple[str, ...], Callable[[pystac.Item, str], float | None]] | None = None
    kind: str = "optical"
    optional: tuple[str, ...] = ()
    native: bool = True
    fill: float | None = None
    period: tuple[str, str | None] | None = None

    def source(self, name: str, archive: str) -> Source | None:
        """Where `archive` keeps this product (`name` is the product's name), or None."""
        if self.sources is None:
            return Source(name, period=self.period) if archive == "planetary-computer" else None
        return self.sources.get(archive)

    def archives(self) -> list[str]:
        return ["planetary-computer"] if self.sources is None else sorted(self.sources)

    def target_res(self, asset: str) -> float | None:
        return self.res_m.get(asset) if isinstance(self.res_m, dict) else self.res_m


S2 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_L1C = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
S2_EARTH_SEARCH = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
                   "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
                   "B10": "cirrus", "B11": "swir16", "B12": "swir22", "SCL": "scl"}
S2_CDSE_L2A = {"B01": "B01_60m", "B02": "B02_10m", "B03": "B03_10m", "B04": "B04_10m", "B05": "B05_20m",
               "B06": "B06_20m", "B07": "B07_20m", "B08": "B08_10m", "B8A": "B8A_20m", "B09": "B09_60m",
               "B11": "B11_20m", "B12": "B12_20m", "SCL": "SCL_20m"}     # native resolution of each band
S2_PERIOD = ("2015-06-27", None)
DSWX = {"WTR": "B01_WTR", "BWTR": "B02_BWTR", "CONF": "B03_CONF"}   # OPERA DSWx: band numbers in the asset names
LFR = {"GIFAPAR": "gifapar:GIFAPAR", "OTCI": "otci:OTCI", "IWV": "iwv:IWV"}   # netCDF asset : variable


@dataclass(frozen=True)
class EarthEngine:
    """How one Earth Engine image collection names things, for backends/earthengine.py.

    bands      band name we save -> Earth Engine band
    cloud      property with the scene's cloud cover in percent, if any
    platform   property naming the satellite; values are lower-cased with "_" as "-" (LANDSAT_9 -> landsat-9)
    prefix     put before the platform value (Sentinel-1 gives only "A", "B", "C")
    tile       property with the MGRS tile, or the (path, row) properties of a WRS-2 scene
    scaling    band -> (scale, offset): physical = stored x scale + offset; Earth Engine records none per band
    stored     (factor, dtype): Earth Engine values x factor give the integers other archives store (HLS: Earth
               Engine keeps reflectance, the others reflectance x 10 000), so samples match across archives
    zone_index image ids start with the MGRS tile ("T12TVK_..."): the search keeps the window's UTM zone +-1 on
               the server, since some footprints cross 180° and match everywhere (HLS)
    link       (collection, property): copy the property from the image with the same id in that collection, to
               sort by it (Dynamic World takes the cloud cover of its Sentinel-2 image)
    """
    bands: dict[str, str]
    cloud: str | None = None
    platform: str | None = None
    prefix: str = ""
    tile: str | tuple[str, str] | None = None
    scaling: dict[str, tuple[float, float]] = field(default_factory=dict)
    stored: tuple[float, str] | None = None
    zone_index: bool = False
    link: tuple[str, str] | None = None


S2_EE = {b: "B" + b[1:].lstrip("0") if b != "B8A" else "B8A" for b in S2_L1C} | {"SCL": "SCL"}
S2_EE_SCALE = {b: (0.0001, 0.0) for b in S2_L1C}         # harmonized: the +1000 offset is already removed
TM_EE = {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir08": "SR_B4", "swir16": "SR_B5", "swir22": "SR_B7",
         "qa_pixel": "QA_PIXEL"}                           # Landsat 4-7
OLI_EE = {"coastal": "SR_B1", "blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir08": "SR_B5", "swir16": "SR_B6",
          "swir22": "SR_B7", "qa_pixel": "QA_PIXEL"}       # Landsat 8-9
LANDSAT_SR_SCALE = {b: (2.75e-05, -0.2) for b in ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22")}
OLI_L1 = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11")
HLS_EE = {f"B{i:02d}": f"B{i}" for i in range(1, 13)} | {"B8A": "B8A", "Fmask": "Fmask"}
DYNAMIC_WORLD = ("water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare",
                 "snow_and_ice", "label")
MODIS_SR = {f"sur_refl_b{i:02d}": (0.0001, 0.0) for i in range(1, 8)}
NBAR = tuple(f"Nadir_Reflectance_Band{i}" for i in range(1, 8))
ASTER_VNIR, ASTER_SWIR, ASTER_TIR = ("B01", "B02", "B3N"), tuple(f"B{i:02d}" for i in range(4, 10)), \
    tuple(f"B{i}" for i in range(10, 15))
ASTER_PC = {b: f"{asset}#{i}" for asset, bands in (("VNIR", ASTER_VNIR), ("SWIR", ASTER_SWIR), ("TIR", ASTER_TIR))
            for i, b in enumerate(bands, 1)}                    # bands of Planetary Computer's multi-band files
OLCI = tuple(f"Oa{i:02d}_radiance" for i in range(1, 22))
EARTH_ENGINE = {
    "COPERNICUS/S2_SR_HARMONIZED": EarthEngine(S2_EE, "CLOUDY_PIXEL_PERCENTAGE", "SPACECRAFT_NAME", tile="MGRS_TILE",
                                               scaling=S2_EE_SCALE),
    "COPERNICUS/S2_HARMONIZED": EarthEngine(S2_EE, "CLOUDY_PIXEL_PERCENTAGE", "SPACECRAFT_NAME", tile="MGRS_TILE",
                                            scaling=S2_EE_SCALE),
    **{f"LANDSAT/{m}/C02/T1_L2": EarthEngine(TM_EE, "CLOUD_COVER", "SPACECRAFT_ID", tile=("WRS_PATH", "WRS_ROW"),
                                              scaling=LANDSAT_SR_SCALE) for m in ("LT04", "LT05", "LE07")},
    **{f"LANDSAT/{m}/C02/T1_L2": EarthEngine(OLI_EE, "CLOUD_COVER", "SPACECRAFT_ID", tile=("WRS_PATH", "WRS_ROW"),
                                              scaling=LANDSAT_SR_SCALE) for m in ("LC08", "LC09")},
    **{f"LANDSAT/{m}/C02/T1": EarthEngine({b: b for b in OLI_L1} | {"B8": "B8", "qa_pixel": "QA_PIXEL"}, "CLOUD_COVER",
                                           "SPACECRAFT_ID", tile=("WRS_PATH", "WRS_ROW")) for m in ("LC08", "LC09")},
    # L30 B10/B11 are thermal on Earth Engine in another scaling: not mapped (optional in hls2-l30)
    "NASA/HLS/HLSL30/v002": EarthEngine({b: v for b, v in HLS_EE.items() if b not in ("B10", "B11")}, "CLOUD_COVERAGE",
                                        stored=(10000, "int16"), zone_index=True),
    "NASA/HLS/HLSS30/v002": EarthEngine(HLS_EE, "CLOUD_COVERAGE", stored=(10000, "int16"), zone_index=True),
    "GOOGLE/DYNAMICWORLD/V1": EarthEngine({b: b for b in DYNAMIC_WORLD}, "CLOUDY_PIXEL_PERCENTAGE", tile="MGRS_TILE",
                                          link=("COPERNICUS/S2_SR_HARMONIZED", "CLOUDY_PIXEL_PERCENTAGE")),
    "COPERNICUS/S1_GRD": EarthEngine({"vv": "VV", "vh": "VH"}, platform="platform_number", prefix="sentinel-1"),
    **{f"MODIS/061/{p}09A1": EarthEngine({b: b for b in MODIS_SR}, scaling=MODIS_SR) for p in ("MOD", "MYD")},
    **{f"MODIS/061/{p}09Q1": EarthEngine({b: b for b in ("sur_refl_b01", "sur_refl_b02")}, scaling=MODIS_SR)
       for p in ("MOD", "MYD")},
    "MODIS/061/MCD43A4": EarthEngine({b: b for b in NBAR}, scaling={b: (0.0001, 0.0) for b in NBAR}),
    **{f"MODIS/061/{p}11A2": EarthEngine({b: b for b in ("LST_Day_1km", "LST_Night_1km")},
                                         scaling={"LST_Day_1km": (0.02, 0.0), "LST_Night_1km": (0.02, 0.0)})
       for p in ("MOD", "MYD")},
    **{f"MODIS/061/{p}13Q1": EarthEngine({"NDVI": "NDVI", "EVI": "EVI"},
                                         scaling={"NDVI": (0.0001, 0.0), "EVI": (0.0001, 0.0)})
       for p in ("MOD", "MYD")},
    "ASTER/AST_L1T_003": EarthEngine({b: b for b in ASTER_VNIR + ASTER_SWIR + ASTER_TIR}, "CLOUDCOVER"),
    "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG": EarthEngine({"avg_rad": "avg_rad", "cf_cvg": "cf_cvg"}),
    "COPERNICUS/S3/OLCI": EarthEngine({b: b for b in OLCI}),
}
COLLECTIONS = {
    "sentinel-2-l2a": Collection(S2, clouds=("SCL", s2_clouds), offset=(S2, declared_or_baseline_offset),
                                 sources={"planetary-computer": Source("sentinel-2-l2a", period=S2_PERIOD),
                                          "earth-search": Source("sentinel-2-c1-l2a", S2_EARTH_SEARCH, S2_PERIOD),
                                          "cdse": Source("sentinel-2-l2a", S2_CDSE_L2A, S2_PERIOD),
                                          "earth-engine": Source("COPERNICUS/S2_SR_HARMONIZED",
                                                                 period=("2017-03-28", None))}),
    "sentinel-2-l1c": Collection(S2_L1C, clouds_from="sentinel-2-l2a", fill=0,
                                 offset=(S2_L1C, declared_or_baseline_offset),
                                 sources={"earth-search": Source("sentinel-2-l1c", S2_EARTH_SEARCH, S2_PERIOD),
                                          "cdse": Source("sentinel-2-l1c", period=S2_PERIOD),
                                          "earth-engine": Source("COPERNICUS/S2_HARMONIZED", period=S2_PERIOD)}),
    "cop-dem-glo-30": Collection(("DEM",), time="static", res_m=30, kind="elevation",
                                 sources={"planetary-computer": Source("cop-dem-glo-30", {"DEM": "data"}),
                                          "earth-search": Source("cop-dem-glo-30", {"DEM": "data"}),
                                          "cdse": Source("cop-dem-glo-30-dged-cog", {"DEM": "data"})}),
    "landsat-c2-l2": Collection(("blue", "green", "red", "nir08", "swir16", "swir22"), optional=("coastal",),
                                clouds=("qa_pixel", landsat_clouds),                                  # Landsat 4-9
                                sources={"planetary-computer": Source("landsat-c2-l2", period=("1982-08-22", None)),
                                         "earth-engine": Source("+".join(f"LANDSAT/{m}/C02/T1_L2" for m in (
                                             "LT04", "LT05", "LE07", "LC08", "LC09")), period=("1982-08-22", None))}),
    "landsat-c2-l1": Collection(OLI_L1, optional=("B8",), clouds=("qa_pixel", landsat_clouds),       # Landsat 8-9
                                sources={"earth-engine": Source("LANDSAT/LC08/C02/T1+LANDSAT/LC09/C02/T1",
                                                                period=("2013-03-18", None))}),
    "hls2-s30": Collection(("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"),
                           clouds=("Fmask", hls_clouds),
                           sources={"planetary-computer": Source("hls2-s30", period=("2020-01-01", None)),
                                    "earthdata": Source("LPCLOUD/HLSS30_2.0", period=("2015-11-28", None)),
                                    "earth-engine": Source("NASA/HLS/HLSS30/v002", period=("2015-11-28", None))}),
    "hls2-l30": Collection(("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09"), optional=("B10", "B11"),
                           clouds=("Fmask", hls_clouds),                                  # B10, B11: thermal
                           sources={"planetary-computer": Source("hls2-l30", period=("2020-01-01", None)),
                                    "earthdata": Source("LPCLOUD/HLSL30_2.0", period=("2013-04-11", None)),
                                    "earth-engine": Source("NASA/HLS/HLSL30/v002", period=("2013-04-11", None))}),
    "dynamic-world": Collection(DYNAMIC_WORLD, resampling="nearest", kind="land cover",    # per Sentinel-2 scene
                                clouds_from="sentinel-2-l2a",                 # it masks clouds: rank by the L2A SCL
                                sources={"earth-engine": Source("GOOGLE/DYNAMICWORLD/V1",
                                                                period=("2015-06-27", None))}),
    # MODIS: sinusoidal grid, so always resampled; Terra (MOD) and Aqua (MYD) together, as on Planetary Computer
    "modis-09a1": Collection(tuple(MODIS_SR), res_m=500,
                             sources={"planetary-computer": Source("modis-09A1-061", period=("2000-02-18", None)),
                                      "earth-engine": Source("MODIS/061/MOD09A1+MODIS/061/MYD09A1",
                                                             period=("2000-02-18", None))}),
    "modis-09q1": Collection(("sur_refl_b01", "sur_refl_b02"), res_m=250,
                             sources={"planetary-computer": Source("modis-09Q1-061", period=("2000-02-18", None)),
                                      "earth-engine": Source("MODIS/061/MOD09Q1+MODIS/061/MYD09Q1",
                                                             period=("2000-02-18", None))}),
    "modis-43a4": Collection(NBAR, res_m=500,                       # nadir BRDF-adjusted reflectance, daily
                             sources={"planetary-computer": Source("modis-43A4-061", period=("2000-02-16", None)),
                                      "earth-engine": Source("MODIS/061/MCD43A4", period=("2000-02-24", None))}),
    "modis-11a2": Collection(("LST_Day_1km", "LST_Night_1km"), res_m=1000, kind="temperature",   # kelvin x 50
                             sources={"planetary-computer": Source("modis-11A2-061", period=("2000-02-18", None)),
                                      "earth-engine": Source("MODIS/061/MOD11A2+MODIS/061/MYD11A2",
                                                             period=("2000-02-18", None))}),
    "modis-13q1": Collection(("NDVI", "EVI"), res_m=250,
                             sources={"planetary-computer": Source("modis-13Q1-061", {"NDVI": "250m_16_days_NDVI",
                                                                                      "EVI": "250m_16_days_EVI"},
                                                                   period=("2000-02-18", None)),
                                      "earth-engine": Source("MODIS/061/MOD13Q1+MODIS/061/MYD13Q1",
                                                             period=("2000-02-18", None))}),
    # ASTER: digital numbers; SWIR failed in 2008, VNIR missing from some Planetary Computer scenes, so optional
    "aster-l1t": Collection(ASTER_TIR, optional=ASTER_VNIR + ASTER_SWIR, native=False,
                            res_m={**{b: 15 for b in ASTER_VNIR}, **{b: 30 for b in ASTER_SWIR},
                                   **{b: 90 for b in ASTER_TIR}},
                            sources={"planetary-computer": Source("aster-l1t", ASTER_PC,
                                                                  ("2000-03-04", "2006-12-31")),
                                     "earth-engine": Source("ASTER/AST_L1T_003", period=("2000-03-04", None))}),
    "viirs-dnb-monthly": Collection(("avg_rad", "cf_cvg"), res_m=500, kind="night lights",   # nW/cm2/sr, cloud-free
                                    sources={"earth-engine": Source("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
                                                                    period=("2014-01-01", None))}),
    # Sentinel-3 OLCI: swaths; CDSE and Planetary Computer files are netCDF placed by their latitude and longitude
    "sentinel-3-olci-l1": Collection(OLCI, res_m=300, native=False, fill=0,   # top-of-atmosphere radiance
                                     sources={"earth-engine": Source("COPERNICUS/S3/OLCI", period=("2016-10-18", None)),
                                              "cdse": Source("sentinel-3-olci-1-efr-ntc",
                                                             {b: f"{b}Data:{b}" for b in OLCI},
                                                             period=("2020-04-16", None))}),
    "sentinel-3-olci-l2-lfr": Collection(("GIFAPAR", "OTCI", "IWV"), res_m=300, native=False,   # land products
                                         sources={"planetary-computer": Source("sentinel-3-olci-lfr-l2-netcdf", LFR,
                                                                               period=("2016-04-25", None)),
                                                  "cdse": Source("sentinel-3-olci-2-lfr-ntc", LFR,
                                                                 period=("2016-04-25", None))}),
    "sentinel-1-rtc": Collection(("vv", "vh"), kind="radar",       # backscatter (gamma0, linear), UTM 10 m; no clouds
                                 period=("2014-10-10", None)),
    # raw GRD: amplitude digital numbers, placed with the files' ground control points (no terrain correction)
    "sentinel-1-grd": Collection(("vv", "vh"), kind="radar", res_m=10, native=False,
                                 sources={"planetary-computer": Source("sentinel-1-grd", period=("2014-10-10", None)),
                                          "earth-search": Source("sentinel-1-grd", period=("2014-10-10", None)),
                                          "cdse": Source("sentinel-1-grd", period=("2014-10-04", None))}),
    "sentinel-1-grd-tc": Collection(("vv", "vh"), kind="radar",  # terrain-corrected by Earth Engine; backscatter in dB
                                    sources={"earth-engine": Source("COPERNICUS/S1_GRD", period=("2014-10-03", None))}),
    # OPERA (NASA JPL): analysis-ready products from Sentinel-1 and HLS, UTM 30 m; asset keys found by suffix
    "opera-rtc-s1": Collection(("VV", "VH"), kind="radar",          # gamma0 backscatter, linear; one burst per item
                               sources={"earthdata": Source("ASF/OPERA_L2_RTC-S1_V1_1", period=("2016-04-14", None))}),
    "opera-dswx-hls": Collection(("WTR", "BWTR", "CONF"), kind="water", resampling="nearest",   # water classes
                                 sources={"earthdata": Source("POCLOUD/OPERA_L3_DSWX-HLS_V1_1.0", DSWX,
                                                              period=("2016-01-01", None))}),
    "opera-dswx-s1": Collection(("WTR", "BWTR", "CONF"), kind="water", resampling="nearest",
                                sources={"earthdata": Source("POCLOUD/OPERA_L3_DSWX-S1_V1_1.0", DSWX,
                                                             period=("2023-12-01", None))}),
    "opera-dist-alert-hls": Collection(("VEG-DIST-STATUS", "VEG-IND", "VEG-ANOM"), kind="land cover",   # disturbance
                                       resampling="nearest",
                                       sources={"earthdata": Source("LPCLOUD/OPERA_L3_DIST-ALERT-HLS_V1_1",
                                                                    period=("2023-01-01", None))}),
    "alos-palsar-mosaic": Collection(("HH", "HV"), time="yearly", res_m=25, kind="radar",
                                     period=("2015-01-01", "2021-12-31")),
    "nasadem": Collection(("elevation",), time="static", res_m=30, kind="elevation"),
    "alos-dem": Collection(("data",), time="static", res_m=30, kind="elevation"),
    "hgb": Collection(("aboveground", "belowground", "aboveground_uncertainty", "belowground_uncertainty"),
                      time="static", res_m=300, kind="biomass"),
    "alos-fnf-mosaic": Collection(("C",), time="yearly", resampling="nearest", res_m=25, kind="land cover",
                                  period=("2015-01-01", "2020-12-31")),
    "esa-worldcover": Collection(("map", "input_quality"), time="yearly", resampling="nearest",
                                 res_m={"map": 10, "input_quality": 60}, kind="land cover",
                                 period=("2020-01-01", "2021-12-31")),
    "io-lulc-annual-v02": Collection(("data",), time="yearly", resampling="nearest", res_m=10, kind="land cover",
                                     period=("2017-01-01", "2024-01-01")),
    "esa-cci-lc": Collection(("lccs_class", "change_count", "observation_count", "current_pixel_state",
                              "processed_flag"), time="yearly", resampling="nearest", res_m=300, kind="land cover",
                             period=("1992-01-01", "2020-12-31")),
    "jrc-gsw": Collection(("occurrence", "change", "seasonality", "recurrence", "transitions", "extent"),
                          time="static", resampling="nearest", res_m=30, kind="water"),
}
