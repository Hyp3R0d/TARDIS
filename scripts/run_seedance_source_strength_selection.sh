#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/experiments/seedance_source_strength_selection_20260817"
MANIFEST="${ROOT}/seedance_validation50.json"
SEEDS=(3407 2024 17 73 991)

for STRENGTH in 0.30 0.35 0.40 0.45; do
  TAG="s${STRENGTH/./}"
  python -m tardis.experiments.queue \
    --output-root "${ROOT}/runs_${TAG}" \
    --methods tardis \
    --datasets seedance \
    --seeds "${SEEDS[@]}" \
    --protocol source50 \
    --data-split validation \
    --record-ids-file "${MANIFEST}" \
    --source-strength "${STRENGTH}" \
    --metrics primary \
    --num-workers 4 \
    --checkpoint-every 10 \
    --no-refresh-package
done
