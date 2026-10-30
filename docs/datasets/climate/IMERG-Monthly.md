> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** by_cell/ and overviews/ sections and working code examples added 2026-10-30 (examples run as written). Earlier status: unreviewed draft (2026-10-28) of `major-tom/climate/IMERG-Monthly/README.md`. Remove these notes
> when publishing. Licence and citation checked 2026-10-28 (DataCite record of the DOI; NASA Earthdata data use
> guidance). Artifact counts from docs/evidence/monthly_artifacts.py (2026-10-28).

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate-banner.svg" alt="Major TOM Climate" width="100%">

# IMERG-Monthly

Monthly satellite precipitation for every cell of the Major TOM grid, 1998–2024, with climate normals over the whole record. Part of [Major TOM Climate](https://source.coop/major-tom/climate).

<img src="https://data.source.coop/major-tom/climate/IMERG-Monthly/figures/annual-precipitation.png" alt="Annual precipitation from satellites, 1998-2024" width="100%">

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
| Cells | 5,113,745: the whole globe, land and sea |
| Source grid | 0.1° (~11 km): about one source pixel per cell |

## Variables

| Column | Meaning | Units |
|---|---|---|
| `imerg_precip` | mean precipitation rate over the month | mm h⁻¹ (× 24 × days in month = monthly total in mm) |
| `imerg_error` | random error of the merged satellite-gauge estimate (NASA's randomError) | mm h⁻¹ |
| `imerg_gauge_weight` | weight of rain gauges relative to the satellite estimate (gaugeRelativeWeighting) | % |
| `imerg_quality` | quality index of the precipitation field (precipitationQualityIndex) | equivalent gauges per 2.5° box |
| `imerg_liquid` | accumulation-weighted probability that the precipitation was liquid (probabilityLiquidPrecipitation) | % |
| `imerg_valid` | share of the cell covered by source pixels with data | 0–1 |

## Files

| Path | Contents |
|---|---|
| `cells.parquet` | `grid_cell`, `grid_row_u`, `grid_col_r` |
| `series/year=YYYY/MM.parquet` | one row per cell, along the Hilbert curve: `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, the columns above |
| `climatology/band=NN.parquet` | one row per cell and calendar month over 1998–2024: `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `<column>_mean`, `<column>_std` for each column above, `n_years` |
| `by_cell/band=NN.parquet` | every month of each cell together: rows by cell (along the Hilbert curve), then `year`, `month`; the same columns as `series/`; a row group holds whole cells (202 cells × 324 months) |
| `overviews/year=YYYY/MM.parquet` | one row per block of 4 × 4 cells with a value, for maps of the whole globe: `block_row` = (`grid_row_u` + 1002) // 4, `block_col` = (`grid_col_r` + 2004) // 4, `n_cells` (cells with a value), each column's mean |

Which files to read: `series/` for one month over many cells (maps, monthly analyses); `by_cell/` for one place over time or training on series (a cell's whole record is in one row group, found from its band and the row-group statistics on `grid_row_u` and `grid_col_r`); `overviews/` for a view of the whole globe from one small file. `series/` and `by_cell/` hold the same values.

Series values are the monthly values defined in Variables: `imerg_precip` is the month's mean precipitation rate (mm/h). The climatology gives, per cell and calendar month over 1998–2024, the mean (`_mean`) and the year-to-year sample standard deviation (`_std`), and `n_years`, the years with data.

```python
import fsspec
import pandas as pd

https = fsspec.filesystem("https")      # reads only the parts of a file a query needs
url = "https://data.source.coop/major-tom/climate/IMERG-Monthly"
jan = pd.read_parquet(f"{url}/series/year=2024/01.parquet", filesystem=https)
normals = pd.read_parquet(f"{url}/climatology/band=09.parquet", filesystem=https, filters=[("grid_row_u", "=", -35), ("grid_col_r", "=", -667)])   # near Manaus
history = pd.read_parquet(f"{url}/by_cell/band=09.parquet", filesystem=https, filters=[("grid_row_u", "=", -35), ("grid_col_r", "=", -667)])   # 324 months
```

The band of a cell is (row + 1002) × 20 // 2004, with row the signed number in its name (`35D_667L`: −35, band 09). Filter on `grid_row_u` and `grid_col_r`, the numbers in a cell's name (U and R positive, D and L negative): the row-group statistics then let a reader skip the rest of the file (a cell's series from `by_cell/`: under a second). A filter on `grid_cell` reads the whole file.

## Notes

- **Climatology period.** IMERG starts in 1998, so its normals use the whole record (1998–2024) rather than the WMO period 1991–2020; `n_years` gives the years behind each value (24–27).
- **Quality columns.** `imerg_error`, `imerg_gauge_weight` and `imerg_quality` are IMERG's own per-pixel quality fields, averaged over the cell like precipitation. Low `imerg_quality` and `imerg_gauge_weight` and high `imerg_error` mark values to treat with caution.
- **High latitudes.** Satellite microwave estimates of precipitation are least reliable over snow-covered and freezing ground.
- **Known source artifacts.** Beyond 55° latitude, 1,805 cell-months exceed 1 mm h⁻¹ (about 720 mm in a month), spread over the whole record:
  - **north-eastern Siberia** (around 68°N 145°E): 1,027 cell-months, up to 6.8 mm h⁻¹, mostly in March and September–November, in a region that receives about 250 mm a year: artifacts of the freeze and thaw seasons (e.g. 7.1 mm h⁻¹ in NASA's own file at 63.3°N 143.7°E in March 1999);
  - **the southern Alaska coast** (around 60°N 148°W): 753 cell-months, up to 1.8 mm h⁻¹; a very wet coast; likely overestimates.
  Values are kept unchanged.
- **Cell means.** Values are area-weighted means over each cell, without interpolation; see the [top-level README](https://source.coop/major-tom/climate).

## Validation

Every month from 1998-01 to 2024-12 is present; every file has one row per cell; `imerg_valid` lies in [0, 1]; the climatology has one row per cell and calendar month. `by_cell/` equals `series/` for 50 random cells per band in three months; the overviews are means of `series/` (checked for one month).

## Citation

Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, J. Tan (2023), GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07, Greenbelt, MD, Goddard Earth Sciences Data and Information Services Center (GES DISC), doi:10.5067/GPM/IMERG/3B-MONTH/07.

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
