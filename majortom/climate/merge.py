"""Merge per-source Core-Weather folders into one hourly and one daily table, for publishing (docs/CLIMATE.md 8).

    <root>/<source>/hourly|daily/year=YYYY/<date>.parquet   (one folder per source, as built)
      -> per acquisition day: outer join of the sources' files on grid_cell, acquired and the step
      -> <out>/hourly|daily/year=YYYY/<date>.parquet          (a rerun skips days already merged)

Sources keep their own prefixed columns (era5_..., cams_..., cpc_..., s5p_...). Every merged file has every
source's columns: a day missing from one source (e.g. Sentinel-5P before 2018, or a stage not yet built) leaves
them empty, so all files share one schema. A column already given by an earlier source is kept from that one: the
folders of one source (era5 with the basic variables, then era5-g1 ... era5-g8) each carry era5_land and
era5_valid; era5_land is the same in all, era5_valid is counted on each folder's first variable, so era5's (2 m
temperature, everywhere) is the one kept when era5 is listed first.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KEYS = {"hourly": ["grid_cell", "acquired", "step_h"], "daily": ["grid_cell", "acquired", "step_d"]}


def merge(root: Path | str, sources: list[str], out: Path | str, kinds: tuple[str, ...] = ("hourly", "daily"),
          rerun: bool = False) -> int:
    """Merge `sources` (folder names under `root`) into `out`; returns the number of day files written."""
    root, out = Path(root), Path(out)
    written = 0
    for kind in kinds:
        days = sorted({p.relative_to(root / s / kind) for s in sources
                       for p in (root / s / kind).glob("year=*/*.parquet")})
        columns, dtypes = {}, {}                                   # each source's columns and types, from its first file
        for s in sources:
            first = next(iter(sorted((root / s / kind).glob("year=*/*.parquet"))), None)
            if first is not None:
                sample = pd.read_parquet(first)
                seen = {c for names in columns.values() for c in names}
                columns[s] = [c for c in sample.columns if c not in KEYS[kind] and c not in seen]
                dtypes.update(sample.dtypes.to_dict())
        for day in days:
            target = out / kind / day
            if target.exists() and not rerun:
                continue
            parts = [pd.read_parquet(root / s / kind / day) for s in sources if (root / s / kind / day).exists()]
            table = parts[0]
            for part in parts[1:]:
                part = part.drop(columns=[c for c in part.columns if c in table.columns and c not in KEYS[kind]])
                table = table.merge(part, on=KEYS[kind], how="outer")
            for s, names in columns.items():
                for name in names:
                    if name not in table:
                        table[name] = pd.Series(index=table.index, dtype=dtypes[name])     # empty, same type
            table = table[KEYS[kind] + [n for s in columns for n in columns[s]]]
            target.parent.mkdir(parents=True, exist_ok=True)
            table.sort_values(KEYS[kind]).to_parquet(target, index=False)
            written += 1
    return written
