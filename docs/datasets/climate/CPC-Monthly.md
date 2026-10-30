> **CLAUDE-GENERATED DOCUMENT** — automated LLM-generated content. Verify before relying on it.

> **Review status:** by_cell/ and overviews/ sections and working code examples added 2026-10-30 (examples run as written). Earlier status: unreviewed draft (2026-10-28) of `major-tom/climate/CPC-Monthly/README.md`. Remove these notes
> when publishing. Licence and citations checked 2026-10-28 (NOAA PSL catalogue, datasets 48 and 49).
> Artifact counts from docs/evidence/monthly_artifacts.py (2026-10-28).

<img src="https://data.source.coop/major-tom/climate/brand/mt-climate-banner.svg" alt="Major TOM Climate" width="100%">

# CPC-Monthly

Monthly precipitation and daily temperature extremes from rain gauges, for every land cell of the Major TOM grid, 1979–2025, with 1991–2020 climate normals. Part of [Major TOM Climate](https://source.coop/major-tom/climate).

<img src="https://data.source.coop/major-tom/climate/CPC-Monthly/figures/annual-precipitation.png" alt="Annual precipitation from rain gauges, 1991-2020" width="100%">

<img src="https://data.source.coop/major-tom/climate/CPC-Monthly/figures/maximum-temperature.png" alt="January and July daily maximum temperature, 1991-2020" width="100%">

## Source

| | |
|---|---|
| Product | NOAA CPC Global Unified Gauge-Based Analysis of Daily Precipitation; CPC Global Unified Temperature |
| Provider | NOAA Physical Sciences Laboratory (PSL), https://psl.noaa.gov |
| Kind | analysis of rain-gauge and weather-station reports, interpolated to a 0.5° grid |
| Accessed | yearly daily files from https://downloads.psl.noaa.gov/Datasets (2026-10-24) |
| Licence | no usage restrictions (NOAA PSL catalogue); acknowledgement of NOAA PSL requested |

## Coverage

| | |
|---|---|
| Record | January 1979 – December 2025, 564 months |
| Cells | 1,657,095: land cells (and coastal cells with any land) |
| Source grid | 0.5° (~55 km): one source pixel covers about 25 cells, which then share its value |

## Variables

| Column | Meaning | Units |
|---|---|---|
| `cpc_precip` | total precipitation in the month (sum of daily totals) | mm |
| `cpc_tmax` | mean of the daily maximum temperature | °C |
| `cpc_tmin` | mean of the daily minimum temperature | °C |
| `cpc_valid` | share of the cell covered by source pixels with data | 0–1 |

## Files

| Path | Contents |
|---|---|
| `cells.parquet` | `grid_cell`, `grid_row_u`, `grid_col_r` |
| `series/year=YYYY/MM.parquet` | one row per cell, along the Hilbert curve: `grid_cell`, `grid_row_u`, `grid_col_r`, `year`, `month`, the columns above |
| `climatology/band=NN.parquet` | one row per cell and calendar month, 1991–2020: `grid_cell`, `grid_row_u`, `grid_col_r`, `month`, `<column>_mean`, `<column>_std`, `n_years` |
| `by_cell/band=NN.parquet` | every month of each cell together: rows by cell (along the Hilbert curve), then `year`, `month`; the same columns as `series/`; a row group holds whole cells (116 cells × 564 months) |
| `overviews/year=YYYY/MM.parquet` | one row per block of 4 × 4 cells with a value, for maps of the whole globe: `block_row` = (`grid_row_u` + 1002) // 4, `block_col` = (`grid_col_r` + 2004) // 4, `n_cells` (cells with a value), each column's mean |

Which files to read: `series/` for one month over many cells (maps, monthly analyses); `by_cell/` for one place over time or training on series (a cell's whole record is in one row group, found from its band and the row-group statistics on `grid_row_u` and `grid_col_r`); `overviews/` for a view of the whole globe from one small file. `series/` and `by_cell/` hold the same values.

Series values are the monthly values defined in Variables: `cpc_precip` is the total of the month's daily values, `cpc_tmax` and `cpc_tmin` the means of the month's daily maxima and minima. The climatology gives, per cell and calendar month over 1991–2020, the mean (`_mean`) and the year-to-year sample standard deviation (`_std`) of each column, and `n_years`, the years with data, counted on `cpc_precip`; temperature can have fewer years where precipitation has data and temperature does not.

```python
import fsspec
import pandas as pd

https = fsspec.filesystem("https")      # reads only the parts of a file a query needs
url = "https://data.source.coop/major-tom/climate/CPC-Monthly"
feb = pd.read_parquet(f"{url}/series/year=2021/02.parquet", filesystem=https)
normals = pd.read_parquet(f"{url}/climatology/band=16.parquet", filesystem=https, filters=[("grid_row_u", "=", 624), ("grid_col_r", "=", -29)])
history = pd.read_parquet(f"{url}/by_cell/band=16.parquet", filesystem=https, filters=[("grid_row_u", "=", 624), ("grid_col_r", "=", -29)])   # 564 months
```

The band of a cell is (row + 1002) × 20 // 2004, with row the number in its name (`624U_29L`: 624, band 16). Filter on `grid_row_u` and `grid_col_r`, the numbers in a cell's name (U and R positive, D and L negative): the row-group statistics then let a reader skip the rest of the file (a cell's series from `by_cell/`: under a second). A filter on `grid_cell` reads the whole file.

## Notes

- **Gauge density.** Quality follows gauge density: highest in Europe, North America and East Asia, lowest over tropical Africa, deserts and Antarctica. In the climatology, `n_years` is below 30 where a cell had no data in some years, and 0 where it had none in 1991–2020.
- **The gauge network changes in 2006.** CPC's retrospective version (1979–2005) uses about 30,000 gauges, the real-time version (2006 onwards) about 17,000 (CPC's README), so values before and after 2006 come from different networks. Trends across 2006 should be treated with caution.
- **End of the CPC day.** Each country reports its daily total for its own observation day, so a CPC day ends at a different UTC hour per region. Monthly totals are unaffected except at month boundaries.
- **Known source artifacts.** Six months in the record have cells above 4,000 mm, at places and in seasons where such totals do not occur: February 1991 in central Mexico (up to 9,578 mm), January 1991 in Yucatán (5,195 mm), September 1999 in São Paulo state (6,375 mm), November and December 1993 and October 2001 in western Amazonas (4,505–6,011 mm); 437 cell-months in all. In these months the daily analysis carries large values on most days. They come from the source: we checked them against NOAA's daily files (equal to 0.1 mm) and kept them unchanged. Users who need clean totals can mask them by month and place.
- **Cell means.** Values are area-weighted means over each cell, without interpolation; see the [top-level README](https://source.coop/major-tom/climate).

## Validation

Every month from 1979-01 to 2025-12 is present; every file has one row per cell; `cpc_valid` lies in [0, 1]; the climatology has one row per cell and calendar month. Monthly totals equal the sum of NOAA's daily values (checked to 0.1 mm on the largest values). `by_cell/` equals `series/` for 50 random cells per band in three months; the overviews are means of `series/` (checked for one month).

## Acknowledgement and citation

CPC Global Unified Precipitation and Temperature data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov.

Precipitation, as NOAA PSL asks:

- Xie, P., A. Yatagai, M. Chen, T. Hayasaka, Y. Fukushima, C. Liu, and S. Yang (2007), A gauge-based analysis of daily precipitation over East Asia, J. Hydrometeorol., 8, 607–626.
- Chen, M., W. Shi, P. Xie, V. B. S. Silva, V. E. Kousky, R. W. Higgins, and J. E. Janowiak (2008), Assessing objective techniques for gauge-based analyses of global daily precipitation, J. Geophys. Res., 113, D04110, doi:10.1029/2007JD009132.

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
