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
  --output-root /home/TARDIS/TARDIS_SOURCE_EXPERIMENTS/diagnostics/main \
  --protocol source50_diagnostics \
  --source-strength 0.45 \
  --methods streamdiffusion_img2img rerender_flow tokenflow_core \
            vid2vid_zero_core controlvideo_canny stablevideo_propagation \
            animatediff_lightning text2video_zero sd_turbo_independent tardis \
  --datasets dataverse seedance openvid \
  --seeds 3407 3413 3433 3469 3491 \
  --temporal-diagnostics \
  --flow-cache-root /root/autodl-tmp/TARDIS/metric_cache/raft_small_backward \
  --flow-batch-size 2 \
  --stop-on-error \
  --no-refresh-package
