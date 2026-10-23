> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed draft of the dataset card for Major-TOM/Core-Weather (2026-10-23).

# Core-Weather

Weather and air quality around every acquisition of the timestamped Major TOM Core datasets (Core-S2L2A, Core-S2L1C, Core-S1RTC): an hourly table for the 24 hours either side of each image, and a daily table for the 30 days either side. Values are means over each 10 km grid cell. Built with `majortom.climate` (`python -m majortom.climate stream`); design in `docs/CLIMATE.md`.

## Tables

| Table | One row per | Steps |
|---|---|---|
| `hourly/` | acquisition × hour | `step_h` from −24 to +24 |
| `daily/` | acquisition × day | `step_d` from −30 to +30 |
| `acquisitions.parquet` | Core sample | links each sample (`dataset`, `product_id`) to its rows |

Rows are keyed by `grid_cell` and `acquired` (the acquisition time, UTC). Core-S2L2A and Core-S2L1C samples of the same acquisition share rows. Files are split by acquisition year and day: `daily/year=2021/2021-07-14.parquet`.

## Steps

- **Hourly:** step 0 is the full hour nearest the acquisition; step −1 the hour before.
- **Daily:** step 0 is the 24 hours centred on the acquisition (−12 h to +12 h); step −1 the 24 hours before that.
- Sources with their own days (NOAA CPC, Sentinel-5P) contribute the source day that holds the window's centre. Their `<source>_start` column gives that day's UTC start, so every value states which 24 hours it covers.

Step 0 is the closest value in time; it is not a measurement at the image's pixels. ERA5 and CAMS are reanalyses (models fitted to observations); all values are cell means from grids of 0.05° to 0.75°.

## Columns

Each source's columns carry its prefix. Units are as published by the producer; values are not converted or altered.

| Source | Hourly | Daily | Grid |
|---|---|---|---|
| ERA5 (`era5_`) | `t2m`, `d2m` (K), `tp` (m, hour ending at the stamp), `u10`, `v10` (m s⁻¹), `sp` (Pa), `ssrd` (J m⁻²), `sf`, `sd` (m of water equivalent), `swvl1` (m³ m⁻³), `sst` (K), `siconc` (0–1), `valid`, `land` | means of states, sums of `tp`, `ssrd`, `sf`, min and max of `t2m` | 0.25° |
| CAMS EAC4 (`cams_`) | every 3 hours: surface `no2`, `o3`, `co`, `so2`, `hcho` (kg kg⁻¹, lowest model level), `pm1`, `pm2p5`, `pm10` (kg m⁻³); total columns `tc_no2`, `tc_o3`, `tc_co`, `tc_so2`, `tc_hcho`, `tc_ch4` (kg m⁻²); aerosol optical depth at 550 nm, total and by type | means of the 8 three-hourly steps | 0.75° |
| NOAA CPC (`cpc_`) | — | `precip` (mm), `tmax`, `tmin` (°C), `start` | 0.5° |
| Sentinel-5P (`s5p_`) | — | `no2`, `co`, `o3`, `so2`, `hcho` (mol m⁻²), `ch4` (ppb), `aer_ai`, a valid share per product, `start` | 0.05° grid of ~5.5 × 3.5 km pixels |
| IMERG (`imerg_`) | `precip` (mm, hour ending at the stamp) | sums | 0.1° |

`<source>_valid` is the share of the cell with data. `era5_land` is the share of the cell in ERA5 land pixels (land fraction ≥ 0.5); ERA5 soil water and snow depth average over those pixels only, and are empty for cells without land. Sea surface temperature and sea ice cover are empty over land.

## Notes on the sources

- **ERA5:** hourly accumulations hold tiny negative values (about −2 × 10⁻⁹ m) from storage precision; they are kept.
- **NOAA CPC:** each country's gauges report at their own hour, so a CPC day ends at a different UTC hour per country (CPC's end-of-day map, which CPC describes as based on limited, unchecked information).
- **Sentinel-5P:** data start on 2018-06-28; earlier steps are empty. One overpass a day; gaps under clouds and between swaths.
- **CAMS EAC4:** available 2003–2025.
- Core-S2L2A includes open-sea cells; for them, land-only variables are empty.

## Licences and attribution

| Source | Licence | Required notice |
|---|---|---|
| ERA5, CAMS | Copernicus licence (CC-BY) | "Contains modified Copernicus Climate Change Service information [year]" (ERA5); "Contains modified Copernicus Atmosphere Monitoring Service information [year]" (CAMS) |
| Sentinel-5P | Copernicus Sentinel data terms | "Contains modified Copernicus Sentinel data [year]" |
| NOAA CPC | no restriction | acknowledgement of NOAA PSL requested |
| IMERG | NASA open data | cite the IMERG V07 product |

The exact ERA5 and CAMS notice wording is to be checked against each product's licence page before publishing.

## Size

About 3.8 million acquisitions. ERA5 alone: ~16 GB of parquet. All sources: tens of GB.
