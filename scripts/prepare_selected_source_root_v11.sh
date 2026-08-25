#!/usr/bin/env bash
set -euo pipefail

BASE="/home/TARDIS/TARDIS_SOURCE_EXPERIMENTS"
ROOT="/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/experiments/seedance_source_strength_selection_20260817"
SELECTED="${ROOT}/selected_test/tardis/seedance"
DEST="/home/TARDIS/TARDIS_SOURCE_EXPERIMENTS_V11"

if [[ -e "${DEST}" ]]; then
  echo "refusing to overwrite existing source root: ${DEST}" >&2
  exit 1
fi

for SEED in 3407 3413 3433 3469 3491; do
  test -f "${SELECTED}/seed_${SEED}/metrics.json"
done

install -d "${DEST}/main/tardis"
for METHOD in controlvideo_canny rerender_flow stablevideo_propagation streamdiffusion_img2img tokenflow_core vid2vid_zero_core; do
  cp -a "${BASE}/main/${METHOD}" "${DEST}/main/"
done
cp -a "${BASE}/main/tardis/dataverse" "${DEST}/main/tardis/"
cp -a "${BASE}/main/tardis/openvid" "${DEST}/main/tardis/"
cp -a "${SELECTED}" "${DEST}/main/tardis/seedance"
cp -a "${BASE}/supplement" "${DEST}/"
cp -a "${BASE}/diagnostics" "${DEST}/"
