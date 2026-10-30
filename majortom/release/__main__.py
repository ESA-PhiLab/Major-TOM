"""Command line for majortom.release (docs/RELEASE.md).

    python -m majortom.release build  --samples <build folder> --out <out.zip> --id <id> --description <text>
                                      --license CC-BY-4.0 --provider "Major TOM" [--no-time] [--no-geoenrich] ...
        built samples (*.npz + .json, one product) -> TACO dataset: partitions + TACOCAT, validated
    python -m majortom.release publish --folder <release folder> --to source.coop://<account>/<repository>[/<prefix>]
        upload partitions + TACOCAT (resumable; files already there are skipped); --dry-run lists the plan
    python -m majortom.release to-cog --dataset <folder, .zip or URL> --out <folder> [--where "<SQL condition>"]
        rumi files of the selected samples -> one COG per sample and file, with the Major TOM profile
    python -m majortom.release tables repack --src <dataset folder> --out <folder> [--workers N]
        tables keyed by grid_cell (cells, series, climatology, or a static table) -> rows along a Hilbert curve
    python -m majortom.release tables overviews --src <dataset folder> --out <dataset folder> [--level 2]
        monthly files -> overviews/year=YYYY/MM.parquet: block means for globe-wide maps
    python -m majortom.release tables by-cell --src <dataset folder> --out <dataset folder> [--workers 4]
        monthly files -> by_cell/band=NN.parquet: every cell's whole series together
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build(a: argparse.Namespace) -> None:
    from .taco import MAX_BYTES, release

    samples = sorted(Path(a.samples).rglob("*.npz"))
    if not samples:
        sys.exit(f"no built samples (*.npz) under {a.samples}")
    path = release(samples, a.out, id=a.id, description=a.description, licenses=a.license, providers=a.provider,
                   title=a.title, keywords=a.keyword, profile=a.profile, max_bytes=a.max_bytes or MAX_BYTES,
                   with_time=not a.no_time, geoenrich=not a.no_geoenrich, overwrite=a.overwrite, progress=True)
    print(f"{len(samples)} samples -> {path} (validated)")


def _to_cog(a: argparse.Namespace) -> None:
    import taco

    from .read import to_cog

    ds = taco.open_dataset(a.dataset)
    rows = ds.sql(f"SELECT * FROM dataset{' WHERE ' + a.where if a.where else ''}").to_pylist()
    items = []
    for row in rows:
        for column in row:
            if column.endswith(".rumi::location"):
                name = column.removesuffix("::location").removesuffix(".rumi")
                items.append((f"{row['id']}_{name}", row[column], row[f"{name}.rumi::header"]))
    written = to_cog(items, a.out)
    print(f"{len(rows)} samples -> {len(written)} COGs in {a.out}")


def _publish(a: argparse.Namespace) -> None:
    from .publish import publish

    report = publish(a.folder, a.to, dry_run=a.dry_run)
    print(report)
    for rel in report.uploaded:
        print(f"  {'would upload' if a.dry_run else 'uploaded'} {rel}")


def _tables(a: argparse.Namespace) -> None:
    if a.layout == "by-cell":
        from .tables import by_cell

        print(f"{by_cell(a.src, a.out, a.workers):,} cells written and checked", flush=True)
        return
    if a.layout == "overviews":
        from .tables import overviews

        print(f"{overviews(a.src, a.out, a.level, a.workers, a.columns):,} overview files written", flush=True)
        return
    from .tables import repack

    print(f"{repack(a.src, a.out, a.workers, drop=tuple(a.drop or ())):,} files written and checked", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m majortom.release", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="built samples -> a TACO dataset")
    b.add_argument("--samples", required=True, help="folder with built samples (*.npz + .json), one product")
    b.add_argument("--out", required=True, help="output .zip path (partitions and TACOCAT go beside it)")
    b.add_argument("--id", required=True)
    b.add_argument("--description", required=True)
    b.add_argument("--title")
    b.add_argument("--license", action="append", required=True, help="repeatable, e.g. CC-BY-4.0")
    b.add_argument("--provider", action="append", required=True, help="repeatable")
    b.add_argument("--keyword", action="append")
    b.add_argument("--profile", default="lossless", help="rumi profile (majortom.release.profiles)")
    b.add_argument("--max-bytes", type=int, help="partition size limit in bytes (default 3 GB)")
    b.add_argument("--no-time", action="store_true", help="one sample per cell: ids are the cell code alone")
    b.add_argument("--no-geoenrich", action="store_true", help="skip GeoEnrich attributes")
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(run=_build)
    u = sub.add_parser("publish", help="upload a release folder (source.coop by default, hf:// for Hugging Face)")
    u.add_argument("--folder", required=True, help="the folder holding the partitions and .tacocat")
    u.add_argument("--to", required=True, help="source.coop://<account>/<repository>[/<prefix>] or hf://<owner>/<dataset>")
    u.add_argument("--dry-run", action="store_true")
    u.set_defaults(run=_publish)
    c = sub.add_parser("to-cog", help="rumi files of a TACO dataset -> COG files")
    c.add_argument("--dataset", required=True, help="dataset folder, .zip or URL")
    c.add_argument("--out", required=True)
    c.add_argument("--where", help='SQL condition on the dataset table, e.g. "\\"majortom:code\\" = \'MT10km_0624U_0029L\'"')
    c.set_defaults(run=_to_cog)
    t = sub.add_parser("tables", help="publishing layouts for tables keyed by grid_cell (climate datasets)")
    tsub = t.add_subparsers(dest="layout", required=True)
    r = tsub.add_parser("repack", help="rows along a Hilbert curve, small row groups, checked against the source")
    r.add_argument("--src", type=Path, required=True, help="a dataset folder (cells, series, climatology) or static table")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--drop", nargs="+", help="columns left out, with their climatology _mean and _std (e.g. era5_ptype)")
    r.set_defaults(run=_tables)
    o = tsub.add_parser("overviews", help="monthly files -> block means for globe-wide maps")
    o.add_argument("--src", type=Path, required=True, help="a dataset folder with series/year=YYYY/MM.parquet")
    o.add_argument("--out", type=Path, required=True, help="dataset folder to write overviews/ into")
    o.add_argument("--level", type=int, default=2, help="blocks of 2^level x 2^level cells (default 2: 4 x 4)")
    o.add_argument("--workers", type=int, default=None)
    o.add_argument("--columns", nargs="+", help="only these columns (default: every float column)")
    o.set_defaults(run=_tables)
    c = tsub.add_parser("by-cell", help="monthly files -> every cell's whole series together, by band")
    c.add_argument("--src", type=Path, required=True, help="a dataset folder (cells.parquet, series/)")
    c.add_argument("--out", type=Path, required=True, help="dataset folder to write by_cell/ into")
    c.add_argument("--workers", type=int, default=4, help="bands at once (memory: ~1/20 of the dataset each)")
    c.set_defaults(run=_tables)
    a = p.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
