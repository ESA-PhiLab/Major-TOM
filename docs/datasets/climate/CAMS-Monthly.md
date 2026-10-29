> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed draft (2026-10-28) of `major-tom/climate/CAMS-Monthly/README.md`. Remove these notes
> when publishing. Licence and citations checked 2026-10-28 (Atmosphere Data Store catalogue: CC-BY-4.0; DataCite
> and Crossref records of both DOIs). The credit line is ECMWF's recommended wording, not a licence requirement.

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate-banner.svg" alt="Major TOM Climate" width="100%">

# CAMS-Monthly

Monthly air quality for every cell of the Major TOM grid, 2003–2025: trace-gas columns, aerosol optical depth by type, particulate matter and surface gases, with climate normals over the whole record. Part of [Major TOM Climate](https://source.coop/major-tom/climate).

<img src="https://data.source.coop/major-tom/climate/CAMS-Monthly/figures/aerosol-optical-depth.png" alt="Mean aerosol optical depth, 2003-2025" width="100%">

<img src="https://data.source.coop/major-tom/climate/CAMS-Monthly/figures/pm25.png" alt="Mean surface PM2.5, 2003-2025" width="100%">

## Source

| | |
|---|---|
| Product | CAMS global reanalysis (EAC4), monthly averaged fields |
| Provider | Copernicus Atmosphere Monitoring Service (CAMS), implemented by ECMWF |
| Kind | reanalysis: an atmospheric chemistry model fitted to satellite observations; complete, without gaps |
| Accessed | Atmosphere Data Store, `cams-global-reanalysis-eac4-monthly` |
| Licence | CC BY 4.0 (the Atmosphere Data Store's licence since 2 July 2025) |

## Coverage

| | |
|---|---|
| Record | January 2003 – December 2025, 276 months |
| Cells | 5,113,745: the whole globe |
| Source grid | 0.75° (~80 km): one source pixel covers about 60 cells, which then share its value |

## Variables

| Column | Meaning | Units |
|---|---|---|
| `cams_tc_no2`, `cams_tc_o3`, `cams_tc_co`, `cams_tc_so2`, `cams_tc_hcho`, `cams_tc_ch4` | total column of NO₂, O₃, CO, SO₂, formaldehyde, methane | kg m⁻² |
| `cams_aod550` | total aerosol optical depth at 550 nm | dimensionless |
| `cams_aod550_dust`, `_seasalt`, `_bc`, `_om`, `_sulphate` | aerosol optical depth at 550 nm by type: dust, sea salt, black carbon, organic matter, sulphate | dimensionless |
| `cams_pm2p5`, `cams_pm10` | particulate matter below 2.5 µm and 10 µm, at the surface | kg m⁻³ |
| `cams_no2`, `cams_o3`, `cams_co`, `cams_so2`, `cams_hcho` | mass mixing ratio at the lowest model level (level 60, near the surface) | kg kg⁻¹ |
| `cams_valid` | share of the cell covered by source pixels with data (always 1) | 0–1 |

PM1 is not in the monthly product.

## Files

| Path | Contents |
|---|---|
| `cells.parquet` | `grid_cell`, `grid_row_u`, `grid_col_r` |
| `series/year=YYYY/MM.parquet` | one row per cell, along the Hilbert curve: `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, the columns above |
| `climatology/band=NN.parquet` | one row per cell and calendar month over 2003–2025: `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `<column>_mean`, `<column>_std`, `n_years` |

Series values are CAMS's monthly means of each variable. The climatology gives, per cell and calendar month over 2003–2025, the mean (`_mean`) and the year-to-year sample standard deviation (`_std`) of each column, and `n_years`, the years with data (23 everywhere, as CAMS has no gaps).

```python
import pandas as pd
url = "https://data.source.coop/major-tom/climate/CAMS-Monthly"
sep = pd.read_parquet(f"{url}/series/year=2024/09.parquet", columns=["grid_cell", "cams_aod550", "cams_pm2p5"])
```

## Notes

- **Climatology period.** CAMS EAC4 starts in 2003, so its normals use the whole record (2003–2025); `n_years` is 23.
- **Grid.** At 0.75°, a city and its surroundings share one value. Major TOM Core-Weather also has Sentinel-5P at a few kilometres, with gaps under cloud.
- **Reanalysis.** Values are model states fitted to satellite retrievals. Surface quantities (PM, surface gases) depend most on the model.
- **Cell means.** Values are area-weighted means over each cell, without interpolation; see the [top-level README](https://source.coop/major-tom/climate).

## Validation

Every month from 2003-01 to 2025-12 is present; every file has one row per cell; no missing values; the climatology has one row per cell and calendar month.

## Licence and citation

Licensed CC BY 4.0. Contains modified Copernicus Atmosphere Monitoring Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

- Copernicus Atmosphere Monitoring Service (2020), CAMS global reanalysis (EAC4) monthly averaged fields, ECMWF, doi:10.24381/fd75fff2.
- Inness, A., et al. (2019), The CAMS reanalysis of atmospheric composition, Atmos. Chem. Phys., 19, 3515–3556, doi:10.5194/acp-19-3515-2019.

## Author and citation

Mikolaj Czerkawski, Asterisk Labs.

To cite this dataset, cite the Major TOM paper below and the source data as listed above. A Major TOM v2 paper will replace this reference when it is published.

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

<img src="https://data.source.coop/major-tom/climate/brand/earth-compress.svg" alt="Earth Compress" width="180">

Developed by [Asterisk Labs](https://asterisk.coop) as part of [Earth Compress](https://asterisk.coop/project-pages/earth-compress.html), an ARIA-funded project jointly led with the National Oceanography Centre.
