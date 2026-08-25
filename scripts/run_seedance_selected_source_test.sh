#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/experiments/seedance_source_strength_selection_20260817"
SELECTION="${ROOT}/selection_result.json"
if [[ ! -f "${SELECTION}" ]]; then
  echo "missing completed validation selection: ${SELECTION}" >&2
  exit 1
fi

STRENGTH="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["source_strength"])' "${SELECTION}")"

python -m tardis.experiments.queue \
  --output-root "${ROOT}/selected_test" \
  --methods tardis \
  --datasets seedance \
  --seeds 3407 3413 3433 3469 3491 \
  --protocol source50 \
  --source-strength "${STRENGTH}" \
  --metrics full \
  --num-workers 4 \
  --checkpoint-every 10 \
  --no-refresh-package
