#!/usr/bin/env bash
set -euo pipefail

cd /home/TARDIS

export HF_HOME=/root/autodl-tmp/TARDIS/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TORCH_HOME=/root/autodl-tmp/TARDIS/cache/torch
export PYTHONUNBUFFERED=1

python -m tardis.experiments.queue \
  --output-root /home/TARDIS/TARDIS_ABLATION_EXPERIMENTS/main \
  --methods tardis_a0 tardis_a1 tardis_a2 tardis_a3 tardis_a4 tardis_a5 \
            tardis_a6 tardis_a7 tardis_a8 tardis_a9 tardis_a10 \
  --datasets dataverse \
  --seeds 3407 \
  --no-refresh-package
