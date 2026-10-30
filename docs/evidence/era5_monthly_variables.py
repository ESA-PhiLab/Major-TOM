"""The variables table of the ERA5-Monthly README, from the source definition, in the order of the data files.

    majortom.climate.sources.SOURCES["era5-monthly-all"] (column = era5_<short name>, CDS name, units as the Climate
    Data Store's monthly means give them: accumulations per day) + the column order of one monthly file
      -> Markdown rows | `era5_<name>` | CDS name | units |, ptype left out (dropped at repack: a code has no mean)

    python docs/evidence/era5_monthly_variables.py /data/Major-TOM/climate-release/ERA5-Monthly/series/year=2024/07.parquet
"""
from __future__ import annotations

import sys

import pyarrow.parquet as pq

from majortom.climate.sources import SOURCES

UNITS = {"**-1": "⁻¹", "**-2": "⁻²", "**-3": "⁻³", "**2": "²", "**3": "³"}


def units(text: str) -> str:
    for a, b in UNITS.items():
        text = text.replace(a, b)
    return text


def rows(path: str) -> list[str]:
    src = SOURCES["era5-monthly-all"]
    columns = [c for c in pq.read_schema(path).names if c.startswith("era5_") and c != "era5_valid"]
    out = []
    for column in columns:
        short = column.removeprefix("era5_")
        out.append(f"| `{column}` | {src.variables[short].replace('_', ' ')} | {units(src.units[short])} |")
    return out


if __name__ == "__main__":
    lines = rows(sys.argv[1])
    print(f"{len(lines)} variables", file=sys.stderr)
    print("\n".join(lines))
