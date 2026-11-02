#!/usr/bin/env bash
# ERA5-Land fill south of 57.1°S for Core-Weather (docs/CLIMATE.md 6.1): one Slurm job per variable group, then the
# fill of the merged folder; one acquisition year per run. Prints the commands; with --submit, submits them (the fill
# after all six groups).
#
#   samples of <year> (docs/evidence/era5land_south_samples.py) -> stream --source era5-land-cds-south, one group per job
#     -> <Core-Weather>/era5land-cds-south-<year>-g1 ... g6   (requests: weeks cut at month ends, per group)
#   -> merge --fill: merged day files updated in place where the fill covers more of the cell
#
#   bash docs/evidence/era5land_south.sh <year> [--submit]      (years with samples: 2016, 2018-2024)
set -euo pipefail
REPO=/home/miko/Projects/asterisk/Major-TOM
PY=$REPO/.venv311/bin/python
CW=/data/Major-TOM/Core-Weather
YEAR=${1:?year, e.g. 2016}
SAMPLES=$(ls $CW/tests/era5land-south/south/year=$YEAR/Core-*/metadata.parquet)
GROUPS_=(
  "t2m d2m u10 v10 sp tp sf ssrd snowc sd swvl1"
  "skt e ro ssr str strd slhf sshf"
  "swvl2 swvl3 swvl4 stl1 stl2 stl3 stl4"
  "pev evabs evatc evavt evaow sro ssro"
  "sde rsn tsn asn smlt es fal src"
  "lblt licd lict lmld lmlt ltlt lai_hv lai_lv"
)
submit=false; [[ "${2:-}" == "--submit" ]] && submit=true
ids=()
for g in 1 2 3 4 5 6; do
  cmd=(sbatch --parsable --cpus-per-task=4 --mem=32G --time=7-00:00:00 -J "era5land-south-$YEAR-g$g" ~/slurm-templates/cpu.sbatch
       "$PY" -W ignore -u -m majortom.climate stream --samples $SAMPLES --source era5-land-cds-south
       --variables ${GROUPS_[$((g - 1))]} --out "$CW/era5land-cds-south-$YEAR-g$g" --cache "$CW/cache" --workers 8 --progress 50000)
  echo "${cmd[*]}"
  if $submit; then ids+=("$("${cmd[@]}")"); fi
done
dep=$(IFS=:; echo "${ids[*]:-}")
fill=(sbatch --parsable --cpus-per-task=4 --mem=32G --time=1-00:00:00 -J era5land-south-$YEAR-fill
      ${dep:+--dependency=afterok:$dep} ~/slurm-templates/cpu.sbatch
      "$PY" -W ignore -u -m majortom.climate merge --fill --root "$CW"
      --sources $(printf "era5land-cds-south-$YEAR-g%s " 1 2 3 4 5 6)--out "$CW/merged")
echo "${fill[*]}"
if $submit; then "${fill[@]}"; fi
