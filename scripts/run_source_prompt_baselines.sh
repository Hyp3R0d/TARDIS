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
  --output-root /home/TARDIS/TARDIS_SOURCE_EXPERIMENTS/supplement/main \
  --protocol source50 \
  --source-strength 0.45 \
  --methods sd_turbo_independent animatediff_lightning text2video_zero \
  --datasets dataverse seedance openvid \
  --seeds 3407 3413 3433 3469 3491 \
  --no-refresh-package
