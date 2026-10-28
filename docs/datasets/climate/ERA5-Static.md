> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed draft (2026-10-28) of `major-tom/climate/ERA5-Static/README.md`. Remove these notes
> when publishing. Licence and citations checked 2026-10-28 (Climate Data Store catalogue: CC-BY-4.0; DataCite and
> Crossref records; ARCO-ERA5's "How to cite"). The credit line is ECMWF's recommended wording, not a licence requirement.

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate.svg" alt="Major TOM Climate" width="360">

# ERA5-Static

The fixed surface fields of ERA5 for every cell of the Major TOM grid: terrain and its roughness, land, lake and vegetation cover, soil and vegetation types. One row per cell. Part of [Major TOM Climate](../).

## Source

| | |
|---|---|
| Product | ERA5 reanalysis, time-invariant fields |
| Provider | Copernicus Climate Change Service (C3S), implemented by ECMWF |
| Accessed | ARCO-ERA5 (Google Cloud, analysis-ready Zarr copy of ERA5) |
| Licence | CC BY 4.0 (the Climate Data Store's licence since 2 July 2025) |

## Coverage

| | |
|---|---|
| Cells | 5,113,745: the whole globe |
| Source grid | 0.25° (~28 km): one source pixel covers about 8 cells, which then share its value |

## Variables

| Column | Meaning | Units |
|---|---|---|
| `era5_z` | geopotential at the surface (÷ 9.80665 = height in m) | m² s⁻² |
| `era5_sdor` | standard deviation of orography (terrain roughness) | m |
| `era5_sdfor` | standard deviation of filtered subgrid orography | m |
| `era5_slor`, `era5_anor`, `era5_isor` | slope, angle and anisotropy of subgrid orography | –, radians, – |
| `era5_lsm` | land-sea mask (share of land) | 0–1 |
| `era5_cl`, `era5_dl`, `era5_lshf` | lake cover, lake depth, lake shape factor | 0–1, m, – |
| `era5_cvh`, `era5_cvl` | high and low vegetation cover | 0–1 |
| `era5_tvh`, `era5_tvl` | type of high and low vegetation (ECMWF classes) | class |
| `era5_slt` | soil type (ECMWF classes) | class |
| `era5_wmb` | model bathymetry (empty over land) | m |

**Class fields** (`era5_tvh`, `era5_tvl`, `era5_slt`) hold the class covering most of the cell, not a mean.

## Files

`static.parquet`: one row per cell along the Hilbert curve (see the [top-level README](../)), `grid_cell`, `grid_row_u`, `grid_col_r` and the columns above.

```python
import pandas as pd
static = pd.read_parquet("https://data.source.coop/major-tom/climate/ERA5-Static/static.parquet")
height = static.set_index("grid_cell").era5_z / 9.80665
```

## Things to know

- **Model fields.** These are the surface fields the ERA5 model uses, at its own resolution: a smoothed terrain and generalised land cover, not a survey. For detail at the cell scale, use Major TOM's DEM and land-cover datasets.
- Use them to interpret the climate datasets: for example, a cell's ERA5 temperature refers to the model's terrain height (`era5_z`), not the true height of the place.

## Validation

One row per cell, no duplicates; `era5_wmb` is empty over land (28% of cells), every other field is complete.

## Licence and citation

Licensed CC BY 4.0. Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

- Copernicus Climate Change Service (2018), ERA5 hourly data on single levels from 1940 to present, Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.adbb2d47.
- Hersbach, H., et al. (2020), The ERA5 global reanalysis, Q. J. R. Meteorol. Soc., 146, 1999–2049, doi:10.1002/qj.3803.
- Carver, R. W., and A. Merose (2023), ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset, 22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc., 4A.1.

---

<img src="https://data.source.coop/major-tom/climate/brand/earth-compress.svg" alt="Earth Compress" width="180">

Developed by [Asterisk Labs](https://asterisk.coop) as part of [Earth Compress](https://asterisk.coop/project-pages/earth-compress.html), an ARIA-funded project jointly led with the National Oceanography Centre.
