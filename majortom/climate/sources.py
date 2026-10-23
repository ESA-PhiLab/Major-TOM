"""Climate sources: where each lives, its grid, variables and units (docs/CLIMATE.md section 2).

    SOURCES[name]  (store, Grid, our variable name -> the source's, units, time step, reader, period)
      -> field(source, variable, time or source day)   one global array, row 0 north, NaN where missing
      -> weights.cell_means                            value and valid share per cell

Hourly sources (period "hour": ERA5, CAMS every 3 h, IMERG) feed the hourly and daily tables through
majortom.climate.stream. Daily sources (period "day": ERA5-Land daily statistics, NOAA CPC, Sentinel-5P)
feed the daily table through majortom.climate.days.

Values keep the units the producer published; they are recorded per variable, never converted.
Values are not altered either: ERA5 accumulations hold tiny negative values (about -2e-9 m) from its
storage precision, and they stay as published.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd

from .weights import Grid


@dataclass(frozen=True)
class Source:
    name: str                    # column prefix: era5_t2m
    store: str                   # where the data are
    grid: Grid
    variables: dict[str, str]    # our short name -> the source's name
    units: dict[str, str]        # as stored
    step: str                    # time step, as a pandas frequency: "1h", "3h", "1D"
    accumulated: tuple[str, ...] = ()   # totals over the step ending at the time stamp, not instantaneous values
    land_only: tuple[str, ...] = ()     # meaningful over land only (ERA5 stores 0 over the sea): land-weighted
    land_mask: str | None = None        # the source's land fraction (0..1), a fixed field; land if >= 0.5
    reader: str = "zarr"                # module in majortom/climate/readers
    period: str = "hour"                # "hour": fields per time step; "day": one field per source day
    options: dict = dataclass_field(default_factory=dict)    # reader settings


ERA5_VARIABLES = {
    "t2m": ("2m_temperature", "K"),
    "d2m": ("2m_dewpoint_temperature", "K"),
    "tp": ("total_precipitation", "m"),
    "u10": ("10m_u_component_of_wind", "m s**-1"),
    "v10": ("10m_v_component_of_wind", "m s**-1"),
    "sp": ("surface_pressure", "Pa"),
    "ssrd": ("surface_solar_radiation_downwards", "J m**-2"),
    "sf": ("snowfall", "m of water equivalent"),
    "sd": ("snow_depth", "m of water equivalent"),
    "swvl1": ("volumetric_soil_water_layer_1", "m**3 m**-3"),
    "sst": ("sea_surface_temperature", "K"),             # NaN over land: a cell mean covers its sea part
    "siconc": ("sea_ice_cover", "(0 - 1)"),              # sea ice area fraction; NaN over land, like sst
}

# ERA5-Land daily statistics: our name -> (request name, name in the delivered netCDF, units)
ERA5_LAND_VARIABLES = {
    "t2m": ("2m_temperature", "t2m", "K"),
    "d2m": ("2m_dewpoint_temperature", "d2m", "K"),
    "u10": ("10m_u_component_of_wind", "u10", "m s**-1"),
    "v10": ("10m_v_component_of_wind", "v10", "m s**-1"),
    "sp": ("surface_pressure", "sp", "Pa"),
    "snowc": ("snow_cover", "snowc", "%"),
    "swvl1": ("volumetric_soil_water_layer_1", "swvl1", "m**3 m**-3"),
}

# CAMS EAC4: our name -> (request name, name in the delivered netCDF, units); surface gases are model level 60
CAMS_SINGLE = {
    "tc_no2": ("total_column_nitrogen_dioxide", "tcno2", "kg m**-2"),
    "tc_o3": ("total_column_ozone", "gtco3", "kg m**-2"),
    "tc_co": ("total_column_carbon_monoxide", "tcco", "kg m**-2"),
    "tc_so2": ("total_column_sulphur_dioxide", "tcso2", "kg m**-2"),
    "tc_hcho": ("total_column_formaldehyde", "tchcho", "kg m**-2"),
    "tc_ch4": ("total_column_methane", "tc_ch4", "kg m**-2"),
    "aod550": ("total_aerosol_optical_depth_550nm", "aod550", "~"),
    "aod550_dust": ("dust_aerosol_optical_depth_550nm", "duaod550", "~"),
    "aod550_seasalt": ("sea_salt_aerosol_optical_depth_550nm", "ssaod550", "~"),
    "aod550_bc": ("black_carbon_aerosol_optical_depth_550nm", "bcaod550", "~"),
    "aod550_om": ("organic_matter_aerosol_optical_depth_550nm", "omaod550", "~"),
    "aod550_sulphate": ("sulphate_aerosol_optical_depth_550nm", "suaod550", "~"),
    "pm1": ("particulate_matter_1um", "pm1", "kg m**-3"),
    "pm2p5": ("particulate_matter_2.5um", "pm2p5", "kg m**-3"),
    "pm10": ("particulate_matter_10um", "pm10", "kg m**-3"),
}
CAMS_SURFACE = {
    "no2": ("nitrogen_dioxide", "no2", "kg kg**-1"),
    "o3": ("ozone", "go3", "kg kg**-1"),
    "co": ("carbon_monoxide", "co", "kg kg**-1"),
    "so2": ("sulphur_dioxide", "so2", "kg kg**-1"),
    "hcho": ("formaldehyde", "hcho", "kg kg**-1"),
}

# ERA5 and ERA5-Land monthly means (Climate Data Store): file names in the delivered netCDF
ERA5_FILES = {"t2m": "t2m", "d2m": "d2m", "tp": "tp", "u10": "u10", "v10": "v10", "sp": "sp", "ssrd": "ssrd", "sf": "sf",
              "sd": "sd", "swvl1": "swvl1", "sst": "sst", "siconc": "siconc"}
ERA5_LAND_MONTHLY = {
    "t2m": ("2m_temperature", "t2m", "K"),
    "d2m": ("2m_dewpoint_temperature", "d2m", "K"),
    "tp": ("total_precipitation", "tp", "m day**-1"),            # monthly means of accumulations: per day
    "u10": ("10m_u_component_of_wind", "u10", "m s**-1"),
    "v10": ("10m_v_component_of_wind", "v10", "m s**-1"),
    "sp": ("surface_pressure", "sp", "Pa"),
    "ssrd": ("surface_solar_radiation_downwards", "ssrd", "J m**-2 day**-1"),
    "sf": ("snowfall", "sf", "m of water equivalent day**-1"),
    "snowc": ("snow_cover", "snowc", "%"),
    "sd": ("snow_depth_water_equivalent", "sd", "m of water equivalent"),
    "swvl1": ("volumetric_soil_water_layer_1", "swvl1", "m**3 m**-3"),
}

# Sentinel-5P L3 on Earth Engine: our name -> (collection, band, units)
S5P_PRODUCTS = {
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", "mol m**-2"),
    "co": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density", "mol m**-2"),
    "o3": ("COPERNICUS/S5P/OFFL/L3_O3", "O3_column_number_density", "mol m**-2"),
    "ch4": ("COPERNICUS/S5P/OFFL/L3_CH4", "CH4_column_volume_mixing_ratio_dry_air", "ppb"),
    "so2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", "mol m**-2"),
    "hcho": ("COPERNICUS/S5P/OFFL/L3_HCHO", "tropospheric_HCHO_column_number_density", "mol m**-2"),
    "aer_ai": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index", "1"),
}


def _names(table: dict, i: int) -> dict[str, str]:
    return {k: v[i] for k, v in table.items()}


SOURCES = {
    # ARCO-ERA5 (Google): Zarr format 2, one chunk per variable and hour; latitude 90 -> -90, longitude 0 -> 359.75
    "era5": Source("era5", "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                   Grid(west=-0.125, north=90.125, step=0.25, width=1440, height=721),
                   _names(ERA5_VARIABLES, 0), _names(ERA5_VARIABLES, 1),
                   step="1h", accumulated=("tp", "ssrd", "sf"), land_only=("swvl1", "sd"), land_mask="land_sea_mask"),
    # ERA5-Land post-processed daily statistics (Climate Data Store): daily means on days centred by UTC offset
    "era5-land": Source("era5land", "derived-era5-land-daily-statistics",
                        Grid(west=-180.05, north=90.05, step=0.1, width=3600, height=1801),
                        _names(ERA5_LAND_VARIABLES, 0), _names(ERA5_LAND_VARIABLES, 2), step="1D",
                        reader="queue", period="day",
                        options={"url": "https://cds.climate.copernicus.eu/api", "files": _names(ERA5_LAND_VARIABLES, 1),
                                 "suffix": "_mean", "offset_step": 3}),
    # CAMS global reanalysis EAC4 (Atmosphere Data Store): 3-hourly, 0.75°, 2003-2025
    "cams": Source("cams", "cams-global-reanalysis-eac4",
                   Grid(west=-0.375, north=90.375, step=0.75, width=480, height=241),
                   {**_names(CAMS_SINGLE, 0), **_names(CAMS_SURFACE, 0)},
                   {**_names(CAMS_SINGLE, 2), **_names(CAMS_SURFACE, 2)}, step="3h", reader="queue",
                   options={"url": "https://ads.atmosphere.copernicus.eu/api", "surface": tuple(CAMS_SURFACE),
                            "files": {**_names(CAMS_SINGLE, 1), **_names(CAMS_SURFACE, 1)}, "model_level": "60"}),
    # [Source]-Monthly: ERA5 monthly means (Climate Data Store), one request per year; accumulations are per day
    "era5-monthly": Source("era5", "reanalysis-era5-single-levels-monthly-means",
                           Grid(west=-0.125, north=90.125, step=0.25, width=1440, height=721),
                           _names(ERA5_VARIABLES, 0),
                           {**_names(ERA5_VARIABLES, 1), "tp": "m day**-1", "ssrd": "J m**-2 day**-1",
                            "sf": "m of water equivalent day**-1"},
                           step="1MS", land_only=("swvl1", "sd"), reader="queue", period="month",
                           options={"url": "https://cds.climate.copernicus.eu/api", "files": ERA5_FILES,
                                    "product_type": "monthly_averaged_reanalysis", "land_from": "era5",
                                    "first_month": "1940-01-01"}),
    "era5-land-monthly": Source("era5land", "reanalysis-era5-land-monthly-means",
                                Grid(west=-180.05, north=90.05, step=0.1, width=3600, height=1801),
                                _names(ERA5_LAND_MONTHLY, 0), _names(ERA5_LAND_MONTHLY, 2), step="1MS",
                                reader="queue", period="month",
                                options={"url": "https://cds.climate.copernicus.eu/api",
                                         "files": _names(ERA5_LAND_MONTHLY, 1),
                                         "product_type": "monthly_averaged_reanalysis", "first_month": "1950-01-01"}),
    # NOAA CPC Global Unified (PSL): daily gauge analyses, 0.5°, yearly netCDF files
    "cpc": Source("cpc", "https://downloads.psl.noaa.gov/Datasets",
                  Grid(west=0.0, north=90.0, step=0.5, width=720, height=360),
                  {"precip": "cpc_global_precip/precip", "tmax": "cpc_global_temp/tmax", "tmin": "cpc_global_temp/tmin"},
                  {"precip": "mm", "tmax": "degC", "tmin": "degC"}, step="1D", reader="files", period="day",
                  options={"end_of_day": "https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/DOCU/"
                                         "PRCP_CU_GAUGE_V1.0GLB_0.50deg_EOD.lnx"}),     # day boundaries per country
    # GPM IMERG Final V07 (NASA GES DISC, Earthdata token): half-hourly 0.1°, summed to the hour ending at the stamp
    "imerg": Source("imerg", "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07",
                    Grid(west=-180.0, north=90.0, step=0.1, width=3600, height=1800),
                    {"precip": "precipitation"}, {"precip": "mm"}, step="1h", accumulated=("precip",), reader="files"),
    # Sentinel-5P OFFL L3 (Earth Engine, project from EE_PROJECT): UTC-day mosaics on a 0.05° grid
    "s5p": Source("s5p", "earthengine", Grid(west=-180.0, north=90.0, step=0.05, width=7200, height=3600),
                  _names(S5P_PRODUCTS, 1), _names(S5P_PRODUCTS, 2), step="1D", reader="earthengine", period="day",
                  options={"collections": _names(S5P_PRODUCTS, 0), "overpass_local": 13.5, "valid_per_variable": True,
                           "first_day": "2018-06-28"}),
}


def reader(source: Source):
    return importlib.import_module(f".readers.{source.reader}", __package__)


def field(source: Source, variable: str, time) -> np.ndarray:
    """One global field (height x width) of `variable` at `time` (hourly sources) or for a source day (daily)."""
    return reader(source).field(source, variable, time)


def valid_range(source: Source) -> tuple[pd.Timestamp, pd.Timestamp]:
    return reader(source).valid_range(source)


def land_pixels(source: Source) -> np.ndarray | None:
    """1 where a source pixel counts as land (sources with a land mask), else None."""
    module = reader(source)
    return module.land_pixels(source) if hasattr(module, "land_pixels") else None


def on_step(source: Source, time: pd.Timestamp) -> bool:
    """Whether `time` is one of the source's time steps (CAMS: every 3 hours from 00 UTC)."""
    step = pd.Timedelta(source.step)
    return (pd.Timestamp(time) - pd.Timestamp(time).normalize()) % step == pd.Timedelta(0)
