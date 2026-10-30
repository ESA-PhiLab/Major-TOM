> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** reviewed (Miko, 2026-10-24); draft of the dataset cards for Major-TOM/[Source]-Monthly (2026-10-24).

# [Source]-Monthly

Monthly values for every Major TOM grid cell over each source's whole record, and a climatology (each calendar month's mean and spread across years). One dataset per source. Built with `python -m majortom.climate monthly` and `climatology`; design in [docs/CLIMATE.md](../CLIMATE.md) section 4.

| Dataset | Source | Record | Cells | Grid |
|---|---|---|---|---|
| ERA5-Monthly | ERA5 monthly means (Copernicus Climate Data Store) | 1940– | whole globe (5.1 M) | 0.25° |
| ERA5Land-Monthly | ERA5-Land monthly means (Climate Data Store) | 1950– | land (1.5 M) | 0.1° |
| CAMS-Monthly | CAMS EAC4 monthly means (Atmosphere Data Store) | 2003–2025 | whole globe | 0.75° |
| IMERG-Monthly | GPM IMERG V07 monthly (NASA, via Earth Engine) | 1998– | whole globe | 0.1° |
| CPC-Monthly | NOAA CPC Global Unified, from daily analyses | 1979– | land (1.6 M) | 0.5° |

## Tables

- `series/year=YYYY/MM.parquet`: one row per cell and month: `grid_cell`, `year`, `month`, one column per variable (`<source>_<variable>`), `<source>_valid` (share of the cell with data).
- `climatology/band=NN.parquet`: one row per cell and calendar month: `<variable>_mean`, `<variable>_std` (spread across years), `n_years`. Split by bands of grid rows, so a cell's 12 months share a file.
- `cells.parquet`: the cells, in the order every month file uses. A cell is kept if the source had data for it in the first month of its record.

## Variables and units

Units are as published; values are cell means, not converted.

- **ERA5:** `t2m`, `d2m` (K), `tp` (m day⁻¹), `u10`, `v10` (m s⁻¹), `sp` (Pa), `ssrd` (J m⁻² day⁻¹), `sf` (m of water equivalent day⁻¹), `sd` (m of water equivalent), `swvl1` (m³ m⁻³), `sst` (K), `siconc` (0–1). Accumulations in monthly means are averages per day. Soil water and snow depth average over land pixels (land fraction ≥ 0.5); sea surface temperature and sea ice cover are empty over land.
- **ERA5-Land:** as ERA5 over land, with `snowc` (%) and `sd` (snow water equivalent); no sea variables.
- **CAMS:** total columns of NO₂, O₃, CO, SO₂, HCHO, CH₄ (kg m⁻²), aerosol optical depth at 550 nm (total and by type), PM2.5 and PM10 (kg m⁻³), surface NO₂, O₃, CO, SO₂, HCHO (kg kg⁻¹, lowest model level). PM1 is not in the monthly product.
- **IMERG:** `precip`, the month's mean precipitation rate (mm hr⁻¹).
- **CPC:** `precip` (monthly total, mm), `tmax`, `tmin` (monthly mean of daily maximum and minimum, °C). CPC days end at different UTC hours per country.

## Climatology period

1991–2020, the current WMO standard normal, for sources that cover it (ERA5, ERA5-Land, CPC). IMERG and CAMS use their full record (1998–2024, 2003–2025; Miko, 2026-10-24); `n_years` says how many years each value rests on.

## Licences and attribution

As for Core-Weather ([docs/datasets/Core-Weather.md](Core-Weather.md)): Copernicus licence for ERA5, ERA5-Land and CAMS, with the "Contains modified Copernicus … information" notice; NOAA PSL acknowledgement for CPC; IMERG V07 citation.

## Size

ERA5-Monthly ~210 GB of values before compression (globe, 1940–2025); IMERG-Monthly 14 GB of parquet (1998–2024); the others smaller.
