> **CLAUDE-GENERATED REPORT** — automated record for tracking software development.

# 2026-10-27 — Release path for every modality, and TACO

All 28 collections buildable without Earth Engine pass the release path (groups -> rumi -> COG -> partition). `majortom.release.release()` writes TACO datasets through taco-eo in Core-DEM's layout; checked end to end on Sentinel-2, a DEM, ESA CCI and Sentinel-1 RTC.

## Release matrix: docs/evidence/release_matrix.py

One sample per collection, one cell (St Louis, inside OPERA's North America), summer 2023, first non-Earth Engine archive. Then groups -> `write_rumi` (lossless) -> read back -> `read(as_="cog")` -> partition.

    python docs/evidence/release_matrix.py build --out /data/Major-TOM/release-bench/matrix     # .venv311
    python docs/evidence/release_matrix.py check --out /data/Major-TOM/release-bench/matrix     # release env

First run found three faults; all fixed (`85beb18`):

- rumi records whole seconds. Every product with sub-second times failed (Sentinel-1/2/3, Landsat, HLS, ASTER, OPERA). `majortom/release/rumi.py:44` rounds down; the exact time stays in `source:acquired` and STAC.
- ESA CCI has uint16 and uint8 bands at 300 m; `groups()` stacked them as uint16. Groups are now keyed by pixel size and dtype (`300m-uint16`, `300m-uint8`), `majortom/release/groups.py:70`.
- 12 px MODIS 1 km images got 176 px COG tiles. `block_size()` in `majortom/release/cog.py` gives min(176, size rounded up to 16).

Also added: categorical data (collections resampled with `nearest`, 11 of 28) refuse lossy profiles in `write_rumi` and `write_cog`. The build now records `resampling` in each sidecar (`majortom/build/__init__.py:298`); older samples fall back to the collection definitions.

Second run: 28 collections, 36 raster groups, all lossless round trips exact, COG pixels and georeferencing identical, categorical groups refuse lossy profiles. Covered: uint8/int16/uint16/float32, 10 m to 1 km, static and yearly products, native and resampled sensors.

Not covered: 4 Earth Engine-only collections (landsat-c2-l1, dynamic-world, viirs-dnb-monthly, sentinel-1-grd-tc) until the quota resets on 1 November; modis-09a1, which Planetary Computer does not hold for tile h10v05 in 2023 (`modis-09Q1-061` does; an archive gap, not a code fault).

## TACO: majortom/release/taco.py

Reuses taco-eo 0.14.2 for the container, staging, partition ZIPs, TACOCAT, validation, the STAC extension (centroid), the MajorTOM extension (`majortom:code`) and the Rumi extension (headers, statistics). Our code adds raster groups, rumi payloads and the partition id.

    path = release(samples, "out/s2.zip", id=..., description=..., licenses=[...], providers=[...],
                   profile="lossless", max_bytes=3 * 10**9, with_time=True)

Sample id is Core-DEM's cell code plus time (`sample_id`, `majortom/release/taco.py:60`): `MT10km_0624U_0029L_20210714T112119`; `with_time=False` gives the cell code alone, as in Core-DEM. taco-eo partitions on `partition:id`, set from `partition()`.

Checks (`docs/evidence/taco_release_check.py`):

| Data | Result |
|---|---|
| 95 Sentinel-2 samples (8 cells x 12 months) | 8 partitions + TACOCAT in 21 s, validated, 285 files read back from the ZIPs identical, `majortom:code` = our cell for all |
| Copernicus DEM, `with_time=False` | validated, exact, id = code |
| ESA CCI land cover (categorical, two dtypes) | validated, both files exact |
| Sentinel-1 RTC (float32, product zone 16, cell zone 15) | validated, exact, code matches |

Offline test: `tests/test_release_taco.py` (synthetic samples, partitions, read back, validate). Release tests: 23 pass in the release env; 108 pass, 3 skipped in `.venv311`.

## Earlier today

- `groups.py`, `partition.py` with the time fallback, `cells.cell_from_name` (`d86cc27`). Core-S2L2A simulated: 4,650 partitions of 2.99 GB, ~260 km across.
- rumi profiles search all GeoZL predictors (`fbff805`): med wins, ~3% smaller.
- Time-series benchmark (RELEASE.md 10a): per-acquisition files as small and fast as per-cell cubes.

## Decisions

- Partition field `partition:id`, chosen: size-targeted id needs its own namespace; rejected `majortom:partition` (owned by the MajorTOM extension's fixed `extra` grid). Open for Miko and Cesar; one constant, `PARTITION_FIELD`.
- One GeoZL graph per raster group for a dataset, chosen on the first sample: rejected per-file graphs (up to 3% smaller, more metadata).
- Categorical = resampled with `nearest`: chosen as today's signal; an explicit `categorical` flag per collection would be clearer (Dynamic World probabilities are continuous but `nearest`, so stored lossless).
- Physical scaling only when the archive declares it: rejected inventing Sentinel-2's 1e-4 scale; `content:stored_offset` records the +1000.
- Release env: Python 3.12 (rasterio 1.5 for tile interleave) at `/data/Major-TOM/release-bench/.venv`, with taco-eo[writer] 0.14.2, rumi-eo 0.26.1, geozl 0.18.0 and the build packages. Project env stays 3.11 until the climate builds finish.

## Open for Miko

1. Partition field name (above).
2. Whether every sample should carry GeoEnrich attributes (needs network at release time; Core-DEM has them).
3. ERA5-Land group 1 (job 1808, at block 431 of ~453, ~199k of ~330k requests) validated on its 2016-2017 overlap with the Earth Engine build (`docs/evidence/era5land_edh_validate.py`; docs/CLIMATE.md 6.1): agrees within ~4 x 10^-4 except (a) 728 cells south of 57.1°S, outside Earth Data Hub (empty in the new build: fill them from the Climate Data Store later, or leave empty and document?), (b) sea cells where Earth Engine fills `ssrd`, (c) days where Earth Engine's derived hourly `ssrd` is wrong (ERA5 agrees with Earth Data Hub). A final pass is queued after the build (job 1829).

## Look at first

- majortom/release/taco.py:86 (release)
- docs/RELEASE.md section 6 (contract as built)
- docs/evidence/release_matrix.py
