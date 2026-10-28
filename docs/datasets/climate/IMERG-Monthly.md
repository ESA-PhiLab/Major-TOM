> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** unreviewed draft (2026-10-28) of `major-tom/climate/IMERG-Monthly/README.md`. Remove these notes
> when publishing. Licence and citation checked 2026-10-28 (DataCite record of the DOI; NASA Earthdata data use
> guidance). Artifact counts from docs/evidence/monthly_artifacts.py (2026-10-28).

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate.svg" alt="Major TOM Climate" width="360">

# IMERG-Monthly

Monthly satellite precipitation for every cell of the Major TOM grid, 1998–2024, with climate normals over the whole record. Part of [Major TOM Climate](../).

## Source

| | |
|---|---|
| Product | GPM IMERG Final Precipitation L3, 1 month, 0.1° (V07) |
| Provider | NASA Goddard Space Flight Center |
| Kind | precipitation estimated from passive-microwave and infrared satellite observations, calibrated to rain gauges |
| Accessed | Google Earth Engine collection `NASA/GPM_L3/IMERG_MONTHLY_V07` |
| Licence | NASA open data: no restrictions on use (CC0 unless marked); acknowledge NASA and cite the dataset |

## Coverage

| | |
|---|---|
| Record | January 1998 – December 2024, 324 months |
| Cells | 5,109,586: the whole globe, land and sea |
| Source grid | 0.1° (~11 km): about one source pixel per cell |

## Variables

| Column | Meaning | Units |
|---|---|---|
| `imerg_precip` | mean precipitation rate over the month | mm h⁻¹ (× 24 × days in month = monthly total in mm) |
| `imerg_valid` | share of the cell covered by source pixels with data | 0–1 |

## Files

| Path | Contents |
|---|---|
| `cells.parquet` | `grid_cell`, `grid_row_u`, `grid_col_r` |
| `series/year=YYYY/MM.parquet` | one row per cell, along the Hilbert curve: `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, the columns above |
| `climatology/band=NN.parquet` | one row per cell and calendar month over 1998–2024: `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `imerg_precip_mean`, `imerg_precip_std`, `n_years` |

```python
import pandas as pd
url = "https://data.source.coop/major-tom/climate/IMERG-Monthly"
jan = pd.read_parquet(f"{url}/series/year=2024/01.parquet")
normals = pd.read_parquet(f"{url}/climatology/band=9.parquet", filters=[("grid_cell", "=", "35D_667L")])   # near Manaus
```

The band of a cell is (row + 1002) × 20 // 2004, with row the signed number in its name (`35D_667L`: −35, band 9).

## Things to know

- **Climatology period.** IMERG starts in 1998, so its normals use the whole record (1998–2024) rather than the WMO period 1991–2020; `n_years` gives the years behind each value (24–27).
- **High latitudes over snow are weakest.** Satellite microwave estimates of precipitation are least reliable over snow-covered and freezing ground.
- **Known source artifacts.** Beyond 55° latitude, 1,805 cell-months exceed 1 mm h⁻¹ (about 720 mm in a month), spread over the whole record:
  - **north-eastern Siberia** (around 68°N 145°E): 1,027 cell-months, up to 6.8 mm h⁻¹, mostly in March and September–November, in a region that receives about 250 mm a year: artifacts of the freeze and thaw seasons (e.g. 7.1 mm h⁻¹ in NASA's own file at 63.3°N 143.7°E in March 1999);
  - **the southern Alaska coast** (around 60°N 148°W): 753 cell-months, up to 1.8 mm h⁻¹; a very wet coast, so likely overestimates rather than impossible values.
  Values are kept unchanged.
- Values are cell means, not interpolated: see the [top-level README](../).

## Validation

Every month from 1998-01 to 2024-12 is present; every file has one row per cell; `imerg_valid` lies in [0, 1]; the climatology has one row per cell and calendar month.

## Citation

Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, J. Tan (2023), GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07, Greenbelt, MD, Goddard Earth Sciences Data and Information Services Center (GES DISC), doi:10.5067/GPM/IMERG/3B-MONTH/07.

---

<img src="https://data.source.coop/major-tom/climate/brand/earth-compress.svg" alt="Earth Compress" width="180">

Developed by [Asterisk Labs](https://asterisk.coop) as part of [Earth Compress](https://asterisk.coop/project-pages/earth-compress.html), an ARIA-funded project jointly led with the National Oceanography Centre.
