#!/usr/bin/env bash
set -euo pipefail

cd /home/TARDIS

export HF_HOME=/root/autodl-tmp/TARDIS/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TORCH_HOME=/root/autodl-tmp/TARDIS/cache/torch
export PYTHONUNBUFFERED=1

python -m tardis.experiments.queue
