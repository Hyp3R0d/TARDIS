#!/usr/bin/env bash
set -euo pipefail

cd /home/TARDIS

LOG=/home/TARDIS/TARDIS_SOURCE_EXPERIMENTS/delivery_pipeline.log
exec > >(tee -a "$LOG") 2>&1

verify_queue() {
  local manifest=$1
  local expected=$2
  python - "$manifest" "$expected" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
results = payload.get("results", [])
completed = sum(
    item.get("status") in {"completed", "already_completed"} for item in results
)
failed = sum(item.get("status") == "failed" for item in results)
assert payload.get("status") == "completed", payload
assert int(payload.get("unit_count", -1)) == expected, payload
assert completed == expected and failed == 0, payload
print(f"verified {path}: {completed}/{expected}, failed={failed}")
PY
}

printf '[%s] verify source core queue\n' "$(date -u +%FT%TZ)"
verify_queue TARDIS_SOURCE_EXPERIMENTS/queue_manifest.json 105

printf '[%s] run source prompt-baseline supplement\n' "$(date -u +%FT%TZ)"
bash scripts/run_source_prompt_baselines.sh
verify_queue TARDIS_SOURCE_EXPERIMENTS/supplement/queue_manifest.json 45

printf '[%s] run source A0-A10 ablation\n' "$(date -u +%FT%TZ)"
bash scripts/run_source_ablations.sh
verify_queue TARDIS_SOURCE_ABLATION_EXPERIMENTS/queue_manifest.json 11

printf '[%s] run unified source diagnostics\n' "$(date -u +%FT%TZ)"
bash scripts/run_source_diagnostics.sh
verify_queue TARDIS_SOURCE_EXPERIMENTS/diagnostics/queue_manifest.json 150

printf '[%s] rebuild and verify delivery package\n' "$(date -u +%FT%TZ)"
python -m tardis.experiments.package --refresh
python RTVD-TC-DataPackage-v1.0/05_scripts/verify_package.py
python -m tardis.experiments.audit

printf '[%s] delivery pipeline completed\n' "$(date -u +%FT%TZ)"
