> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** reviewed (Miko, 2026-11-26), as published at `major-tom/core-weather/README.md`; plain-language opening paragraph, Sources section, licence, source citations and Read from links reviewed (Miko, 2026-11-27). Figures from docs/evidence/core_weather_figures.py; validation from docs/evidence/core_weather_validate.py. 2026-11-28 (text reviewed before editing, Miko): pointer to Climate's Format choices under Tables.

# Core-Weather

For each image in the timestamped Major TOM Core datasets (Core-S2L2A, Core-S2L1C, Core-S1RTC), Core-Weather gives the weather and air quality in the image's 10 km grid cell: every hour from 24 hours before the image to 24 hours after, and every day from 30 days before to 30 days after. It answers questions such as whether it was raining or snowing when the image was taken, whether the ground was frozen, how much rain fell in the weeks before, and how polluted the air was. Values are means over the cell, taken from six public sources. Built with `majortom.climate` (`python -m majortom.climate stream`); design in [docs/CLIMATE.md](https://github.com/ESA-PhiLab/Major-TOM/blob/76b77caf5fbe081e9044e66c8de6a6bd94dc640b/docs/CLIMATE.md).

<img src="https://data.source.coop/major-tom/core-weather/figures/temperature-at-acquisition.png" alt="ERA5 2 m temperature in the hour nearest each acquisition, mean per 12 x 12 cells, 2016-2024" width="100%">

## Sources

Three kinds of source. Reanalyses are weather models fitted to past observations: complete everywhere and at every hour, but modelled. Satellite and rain-gauge products are measurements: closer to what happened, but with gaps and coarser in time.

| Source | Kind | What it contains | Grid, time step | Read from |
|---|---|---|---|---|
| [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) (ECMWF, Copernicus) | reanalysis | the weather worldwide: temperature, humidity, wind, pressure, rain and snow, sunlight, clouds, soil water, snow cover, sea surface and waves | 0.25° (about 28 km), hourly | [ARCO-ERA5](https://github.com/google-research/arco-era5) on Google Cloud |
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) (ECMWF, Copernicus) | reanalysis | ERA5's land surface rerun on a finer grid: temperature, soil water and soil temperature, snow, runoff, evaporation, lakes; land only | 0.1° (about 9 km), hourly | [Earth Data Hub](https://earthdatahub.destine.eu/collections/era5/datasets/era5-land); south of 57.1°S, the [Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) |
| [CAMS EAC4](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4) (ECMWF, Copernicus) | reanalysis | air quality: nitrogen dioxide, ozone, carbon monoxide, sulphur dioxide, formaldehyde, methane, particulate matter (PM1, PM2.5, PM10), aerosol optical depth | 0.75° (about 80 km), every 3 hours | [Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4) |
| [Sentinel-5P TROPOMI](https://sentiwiki.copernicus.eu/web/s5p-mission) (ESA, Copernicus) | satellite | the same gases and an aerosol index, measured from orbit once a day around 13:30 local time; from 2018-06-28; gaps under cloud | 0.05° grid of 5.5 × 3.5 km pixels, daily | [Earth Engine L3](https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p) |
| NOAA CPC Global Unified ([precipitation](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html), [temperature](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html)) | rain gauges | daily rainfall and daily maximum and minimum temperature from weather stations | 0.5° (about 55 km), daily | [NOAA PSL](https://psl.noaa.gov) |
| [GPM IMERG Final V07](https://gpm.nasa.gov/data/imerg) (NASA) | satellite | rainfall estimated from a constellation of satellites | 0.1°, half-hourly, summed to hours | [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07) |

The column prefix of each source (`era5_`, `era5land_`, `cams_`, `s5p_`, `cpc_`, `imerg_`) is listed under Columns.

## Tables

| Table | One row per | Steps |
|---|---|---|
| `hourly/` | acquisition × hour | `step_h` from −24 to +24 |
| `daily/` | acquisition × day | `step_d` from −30 to +30 |
| `acquisitions.parquet` | Core sample | links each sample (`dataset`, `product_id`) to its rows |

Rows are keyed by `grid_cell` and `acquired` (the acquisition time, UTC). Core-S2L2A and Core-S2L1C samples of the same acquisition share rows. Files are split by acquisition year and day: `daily/year=2021/2021-07-14.parquet`.

Format choices for both products (keyed tables, measured duplication of coarse sources, geometry from the `majortom` package): [Major TOM Climate, Format choices](https://source.coop/major-tom/climate).

## Reading

The files are at `https://data.source.coop/major-tom/core-weather/`. Find a sample's acquisition in `acquisitions.parquet`, then read its rows from that day's file:

```python
import fsspec
import pandas as pd

https = fsspec.filesystem("https")
url = "https://data.source.coop/major-tom/core-weather"
links = pd.read_parquet(f"{url}/acquisitions.parquet", filesystem=https, filters=[("grid_cell", "=", "624U_29L")])
pick = links[links.dataset == "Core-S2L2A"].iloc[0]                      # Loch Lomond, 2022-12-11 11:45:01 UTC
day = f"year={pick.acquired:%Y}/{pick.acquired:%Y-%m-%d}.parquet"     # files are split by acquisition day
here = [("grid_cell", "=", pick.grid_cell), ("acquired", "=", pick.acquired)]
hourly = pd.read_parquet(f"{url}/hourly/{day}", filesystem=https, filters=here,
                         columns=["step_h", "era5_t2m", "era5land_t2m", "imerg_precip"])              # 49 hours
daily = pd.read_parquet(f"{url}/daily/{day}", filesystem=https, filters=here,
                        columns=["step_d", "era5land_t2m_mean", "cpc_precip", "cams_pm2p5_mean", "s5p_no2"])  # 61 days
```

## Examples

One Sentinel-2 acquisition at each of three cells, with the same cell's Sentinel-1 and DEM thumbnails from Major TOM Core. The dashed line is the acquisition (step 0). Figures from `docs/evidence/core_weather_figures.py`.

<img src="https://data.source.coop/major-tom/core-weather/figures/place-loch-lomond.png" alt="Loch Lomond, 2022-12-11: thumbnails, hourly and daily temperature and precipitation" width="100%">

<img src="https://data.source.coop/major-tom/core-weather/figures/place-dolomites.png" alt="Dolomites (Corvara), 2018-08-27: thumbnails, hourly and daily temperature and precipitation" width="100%">

<img src="https://data.source.coop/major-tom/core-weather/figures/place-amazon.png" alt="Amazon (Manaus), 2021-07-31: thumbnails, hourly and daily temperature and precipitation" width="100%">

## Steps

- `acquired` is the sample's `timestamp` in the Core metadata: the sensing time of its Sentinel product, the same for every cell of that product.
- A cell can have more than one acquisition on a day (190,372 cell-days have two, e.g. a Sentinel-1 and a Sentinel-2 pass). Each has its own rows, and its steps count from its own time.
- **Hourly:** step 0 is the full hour nearest the acquisition, rounding half up (10:29 → 10:00, 10:30 → 11:00); step −1 the hour before.
- **Daily:** step 0 is the 24 hours centred on the acquisition (−12 h to +12 h); step −1 the 24 hours before that.
- Sources with their own days (NOAA CPC, Sentinel-5P) contribute the source day that holds the window's centre. Their `<source>_start` column gives that day's UTC start, so every value states which 24 hours it covers.

Step 0 is the closest value in time; it is not a measurement at the image's pixels. ERA5 and CAMS are reanalyses (models fitted to observations); all values are cell means from grids of 0.05° to 0.75°.

## Columns

Each source's columns carry its prefix. Units are as published by the producer; values are not converted or altered.

| Source | Hourly | Daily | Grid |
|---|---|---|---|
| ERA5 (`era5_`) | 207 variables: every hourly single-level ERA5 variable except exact duplicates and mean rates ([ERA5_VARIABLES.md](https://github.com/ESA-PhiLab/Major-TOM/blob/76b77caf5fbe081e9044e66c8de6a6bd94dc640b/docs/ERA5_VARIABLES.md)), among them `t2m`, `d2m` (K), `tp` (m, hour ending at the stamp), `u10`, `v10` (m s⁻¹), `sp` (Pa), `ssrd` (J m⁻²), `sf`, `sd` (m of water equivalent), `swvl1` (m³ m⁻³), `sst` (K), `siconc` (0–1); `valid`, `land` | means of states, sums of accumulated variables, min and max of the hourly extremes | 0.25° |
| ERA5-Land (`era5land_`) | all 49 time-varying land variables (among them `t2m`, `d2m` (K), `u10`, `v10` (m s⁻¹), `sp` (Pa), `tp` (m), `sf`, `sd` (m of water equivalent), `ssrd` (J m⁻²), `snowc` (%), `swvl1` (m³ m⁻³)); accumulated variables are amounts over the hour ending at the stamp | means of states, sums of accumulated variables, min and max of `t2m` | 0.1°, land only |
| CAMS EAC4 (`cams_`) | every 3 hours: surface `no2`, `o3`, `co`, `so2`, `hcho` (kg kg⁻¹, lowest model level), `pm1`, `pm2p5`, `pm10` (kg m⁻³); total columns `tc_no2`, `tc_o3`, `tc_co`, `tc_so2`, `tc_hcho`, `tc_ch4` (kg m⁻²); aerosol optical depth at 550 nm, total and by type | means of the 8 three-hourly steps | 0.75° |
| NOAA CPC (`cpc_`) | — | `precip` (mm), `tmax`, `tmin` (°C), `start` | 0.5° |
| Sentinel-5P (`s5p_`) | — | `no2`, `co`, `o3`, `so2`, `hcho` (mol m⁻²), `ch4` (ppb), `aer_ai`, a valid share per product, `start` | 0.05° grid of pixels of 5.5 × 3.5 km at nadir |
| IMERG (`imerg_`) | `precip` (mm, hour ending at the stamp) | sums | 0.1° |

`<source>_valid` is the share of the cell with data; for ERA5 it is counted on 2 m temperature, which covers every cell. `era5_land` is the share of the cell in ERA5 land pixels (land fraction ≥ 0.5); ERA5 soil water and snow depth average over those pixels only, and are empty for cells without land. Sea surface temperature and sea ice cover are empty over land.

Daily columns name their statistic: `_mean` for states, `_sum` for accumulated variables, `_min` and `_max` for 2 m temperature (ERA5, ERA5-Land) and for ERA5's own extremes (`fg10_max`, `mx2t_max`, `mxtpr_max`, `mn2t_min`, `mntpr_min`); `era5_ptype_mode` is the most frequent precipitation type. The `_valid` columns, `era5_land`, NOAA CPC and Sentinel-5P keep their names.

## Notes on the sources

- **ERA5:** hourly accumulations hold tiny negative values (about −2 × 10⁻⁹ m) from storage precision; they are kept.
- **ERA5-Land:** read from Destination Earth's Earth Data Hub copy (zarr), which stores values rounded to 1 part in 16,384 (bitround, 13 mantissa bits; ±0.018 K at 300 K). Checked against Earth Engine's copy: within 4 × 10⁻⁴ relative. Built in six groups of variables ([docs/CLIMATE.md](https://github.com/ESA-PhiLab/Major-TOM/blob/76b77caf5fbe081e9044e66c8de6a6bd94dc640b/docs/CLIMATE.md) 6.1). Empty over the sea. **South of 57.1°S** (Antarctica and nearby islands), where Earth Data Hub's copy stops, the values come from the Climate Data Store's hourly ERA5-Land (`reanalysis-era5-land`), fetched for those cells only and written into the same columns, in the hourly and the daily tables (update of 2026-11-20: 1,225 days, 3,235,318 daily rows on 52,776 cells, 2016 and 2018–2024; Core has no acquisitions there in 2015 or 2017). ERA5 (`era5_*`, 0.25°) has covered these cells since the first release; only the `era5land_*` columns were empty there before 2026-11-20. Checked against ERA5 in the same daily rows: 2 m temperature differs by −0.42 K on average (standard deviation 0.92 K, correlation 0.998). Over glaciers ERA5-Land holds fixed values: snow depth (`sd`) 10 m of water equivalent, snow density (`rsn`) 300 kg m⁻³, leaf area index 0 ([docs/CLIMATE.md](https://github.com/ESA-PhiLab/Major-TOM/blob/76b77caf5fbe081e9044e66c8de6a6bd94dc640b/docs/CLIMATE.md) 6.1).
- **NOAA CPC:** each country's gauges report at their own hour, so a CPC day ends at a different UTC hour per country (CPC's end-of-day map, which CPC describes as based on limited, unchecked information).
- **Sentinel-5P:** data start on 2018-06-28; earlier steps are empty. One overpass a day; gaps under clouds and between swaths.
- **CAMS EAC4:** available 2003–2025.
- Core-S2L2A includes open-sea cells; for them, land-only variables are empty.

## Licences and attribution

| Source | Licence | Required notice |
|---|---|---|
| ERA5, ERA5-Land, CAMS | CC BY 4.0 | "Contains modified Copernicus Climate Change Service information [year]" (ERA5); "Contains modified Copernicus Atmosphere Monitoring Service information [year]" (CAMS) |
| ERA5-Land access | (as ERA5-Land) | credit: "ERA5-Land accessed through the Earth Data Hub of Destination Earth (European Union), https://earthdatahub.destine.eu" |
| ERA5-Land access south of 57.1°S | (as ERA5-Land) | credit: "ERA5-Land south of 57.1°S accessed through the Copernicus Climate Data Store, https://cds.climate.copernicus.eu" |
| Sentinel-5P | Copernicus Sentinel data terms | "Contains modified Copernicus Sentinel data [year]" |
| NOAA CPC | no restriction | acknowledgement of NOAA PSL requested |
| IMERG | NASA open data | cite the IMERG V07 product |

Copernicus data from the Climate Data Store and the Atmosphere Data Store are under CC BY 4.0 since 2 July 2025, when CC BY 4.0 replaced the Licence to use Copernicus Products. ERA5-Land is Copernicus Climate Change Service data; Earth Data Hub, which also serves it under CC BY 4.0, is credited as the access route, not as a licensor. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data this dataset contains.

Source citations:

- Copernicus Climate Change Service (2018), ERA5 hourly data on single levels from 1940 to present, Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.adbb2d47.
- Hersbach, H., et al. (2020), The ERA5 global reanalysis, Q. J. R. Meteorol. Soc., 146, 1999–2049, doi:10.1002/qj.3803.
- Carver, R. W., and A. Merose (2023), ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset, 22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc., 4A.1, https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842.
- Copernicus Climate Change Service (2019), ERA5-Land hourly data from 1950 to present, Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.e2161bac.
- Muñoz-Sabater, J., et al. (2021), ERA5-Land: a state-of-the-art global reanalysis dataset for land applications, Earth Syst. Sci. Data, 13, 4349–4383, doi:10.5194/essd-13-4349-2021.
- Copernicus Atmosphere Monitoring Service (2020), CAMS global reanalysis (EAC4), ECMWF, doi:10.24381/d58bbf47.
- Inness, A., et al. (2019), The CAMS reanalysis of atmospheric composition, Atmos. Chem. Phys., 19, 3515–3556, doi:10.5194/acp-19-3515-2019.
- European Space Agency, TROPOMI Level 2 products, read as Earth Engine's level-3 grids: Nitrogen Dioxide (2021), doi:10.5270/S5P-9bnp8q8; Carbon Monoxide Total Column (2021), doi:10.5270/S5P-bj3nry0; Ozone Total Column (2020), doi:10.5270/S5P-ft13p57; Sulphur Dioxide (2020), doi:10.5270/S5P-74eidii; Formaldehyde (2020), doi:10.5270/S5P-vg1i7t0; Methane Total Column (2021), doi:10.5270/S5P-3lcdqiv; Ultraviolet Aerosol Index, doi:10.5270/S5P-0wafvaf.
- Xie, P., A. Yatagai, M. Chen, T. Hayasaka, Y. Fukushima, C. Liu, and S. Yang (2007), A gauge-based analysis of daily precipitation over East Asia, J. Hydrometeorol., 8, 607–626, doi:10.1175/JHM583.1.
- Chen, M., W. Shi, P. Xie, V. B. S. Silva, V. E. Kousky, R. W. Higgins, and J. E. Janowiak (2008), Assessing objective techniques for gauge-based analyses of global daily precipitation, J. Geophys. Res., 113, D04110, doi:10.1029/2007JD009132.
- Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, J. Tan (2023), GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07, Greenbelt, MD, Goddard Earth Sciences Data and Information Services Center (GES DISC), doi:10.5067/GPM/IMERG/3B-HH/07.

## Size

3,704,013 acquisitions (distinct cell and acquisition time) on 2,976 days, 2015–2024, for 5,961,727 Core samples
(2,245,886 each of Core-S2L2A and Core-S2L1C, 1,469,955 of Core-S1RTC). The hourly table has 181,496,637 rows and
285 columns (137.8 GB of Parquet in 2,976 files), the daily table 225,944,793 rows and 309 columns (258.3 GB in 2,976
files).

## Validation

Before merging, every source folder was checked from its Parquet metadata: all 2,976 days are present for every
source, and on every day all sources have the same number of rows (the merge joins them on cell, acquisition and
step, so no row is left without a partner). Columns that are empty for a whole day are expected: cloud base height
and convective inhibition without cloud or convection, ocean-only variables on days with land acquisitions only (and
the reverse), lake variables where there is no lake, Sentinel-5P before 2018-06-28.

## Author and citation

Built by [Mikolaj Czerkawski](https://github.com/mikonvergence), Asterisk Labs.

To cite this dataset, cite the Major TOM paper below and the sources listed above. A Major TOM v2 paper will replace this reference when it is published.

```bibtex
@inproceedings{Francis2024MajorTOM,
  author={Francis, Alistair and Czerkawski, Mikolaj},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium},
  title={Major TOM: Expandable Datasets for Earth Observation},
  year={2024},
  pages={2935-2940},
  doi={10.1109/IGARSS53475.2024.10640760}
}
```

---

<img src="https://data.source.coop/major-tom/climate/brand/earth-compress.svg" alt="Earth Compress" width="200">

Developed by [Mikolaj Czerkawski](https://github.com/mikonvergence) and Sponsored by Earth Compress. A project funded by the Advanced Research + Invention Agency (ARIA) and co-led by Asterisk Labs and National Oceanography Centre UK. Learn more here: [www.earthcompress.org](https://www.earthcompress.org)
