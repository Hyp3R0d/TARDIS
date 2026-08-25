#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STORAGE_ROOT="${TARDIS_STORAGE_ROOT:-/root/autodl-tmp/TARDIS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache/xdg}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]] || export OMP_NUM_THREADS=1
[[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]] || export MKL_NUM_THREADS=1
CHECKPOINT_ARGS=()
if [[ -n "${TARDIS_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${TARDIS_CHECKPOINT}")
fi
EMA_ARGS=(--use-ema)
[[ "${TARDIS_USE_EMA:-1}" == "0" ]] && EMA_ARGS=(--no-use-ema)
GRADIENT_CHECKPOINTING_ARGS=(--gradient-checkpointing)
[[ "${TARDIS_GRADIENT_CHECKPOINTING:-1}" == "0" ]] && GRADIENT_CHECKPOINTING_ARGS=(--no-gradient-checkpointing)
TRANSPORT_QUOTIENT_ARGS=(--transport-quotient)
[[ "${TARDIS_TRANSPORT_QUOTIENT:-1}" == "0" ]] && TRANSPORT_QUOTIENT_ARGS=(--no-transport-quotient)
PROPER_TIME_ARGS=(--innovation-proper-time)
[[ "${TARDIS_INNOVATION_PROPER_TIME:-1}" == "0" ]] && PROPER_TIME_ARGS=(--no-innovation-proper-time)
KEYFRAME_ALIGNMENT_ARGS=(--keyframe-lite-alignment)
[[ "${TARDIS_KEYFRAME_LITE_ALIGNMENT:-1}" == "0" ]] && KEYFRAME_ALIGNMENT_ARGS=(--no-keyframe-lite-alignment)
KEYFRAME_RESIDUAL_ARGS=(--keyframe-residual-generation)
[[ "${TARDIS_KEYFRAME_RESIDUAL_GENERATION:-1}" == "0" ]] && KEYFRAME_RESIDUAL_ARGS=(--no-keyframe-residual-generation)
SAMPLER_ALIGNMENT_ARGS=(--sampler-trajectory-alignment)
[[ "${TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT:-1}" == "0" ]] && SAMPLER_ALIGNMENT_ARGS=(--no-sampler-trajectory-alignment)
COMPILE_ARGS=(--no-compile-model)
[[ "${TARDIS_COMPILE_MODEL:-0}" == "1" ]] && COMPILE_ARGS=(--compile-model)
DETERMINISTIC_ARGS=(--no-deterministic)
[[ "${TARDIS_DETERMINISTIC:-0}" == "1" ]] && DETERMINISTIC_ARGS=(--deterministic)

torchrun --standalone --nproc_per_node="${TARDIS_NPROC:-1}" -m tardis.cli.apply \
  --dataset "${TARDIS_DATASET:-dataverse}" \
  --pretrained-model "${TARDIS_PRETRAINED_MODEL:-stabilityai/sd-turbo}" \
  --prompt "${TARDIS_PROMPT:-A robot running in the forest}" \
  --style "${TARDIS_STYLE:-cinematic}" \
  --duration "${TARDIS_DURATION:-2}" \
  --height "${TARDIS_HEIGHT:-512}" \
  --width "${TARDIS_WIDTH:-512}" \
  --fps "${TARDIS_FPS:-30}" \
  --seed "${TARDIS_SEED:-3407}" \
  --latent-channels "${TARDIS_LATENT_CHANNELS:-4}" \
  --patch-size "${TARDIS_PATCH_SIZE:-2}" \
  --hidden-size "${TARDIS_HIDDEN_SIZE:-512}" \
  --num-layers "${TARDIS_NUM_LAYERS:-8}" \
  --num-heads "${TARDIS_NUM_HEADS:-8}" \
  --active-ratio "${TARDIS_ACTIVE_RATIO:-0.35}" \
  --motion-max-flow-pixels "${TARDIS_MOTION_MAX_FLOW_PIXELS:-8.0}" \
  --transport-max-correction-pixels "${TARDIS_TRANSPORT_MAX_CORRECTION_PIXELS:-0.25}" \
  --transport-history-fallback-weight "${TARDIS_TRANSPORT_HISTORY_FALLBACK_WEIGHT:-1.0}" \
  --router-threshold "${TARDIS_ROUTER_THRESHOLD:-0.1}" \
  --router-halo-radius "${TARDIS_ROUTER_HALO_RADIUS:-1}" \
  --state-anchor-decay "${TARDIS_STATE_ANCHOR_DECAY:-0.95}" \
  --scene-cut-threshold "${TARDIS_SCENE_CUT_THRESHOLD:-0.98}" \
  --oracle-temperature "${TARDIS_ORACLE_TEMPERATURE:-0.25}" \
  --training-noise-scale "${TARDIS_TRAINING_NOISE_SCALE:-0.1}" \
  --lite-max-magnitude "${TARDIS_LITE_MAX_MAGNITUDE:-0.75}" \
  --quotient-regularization "${TARDIS_QUOTIENT_REGULARIZATION:-1e-4}" \
  --quotient-rank-threshold "${TARDIS_QUOTIENT_RANK_THRESHOLD:-1e-5}" \
  --proper-time-maximum-hazard "${TARDIS_PROPER_TIME_MAXIMUM_HAZARD:-20.0}" \
  --diffusion-steps "${TARDIS_DIFFUSION_STEPS:-2}" \
  --diffusion-time-sampling "${TARDIS_DIFFUSION_TIME_SAMPLING:-endpoint}" \
  --precision "${TARDIS_PRECISION:-bf16}" \
  --output-root "${TARDIS_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}" \
  --checkpoint-root "${TARDIS_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${EMA_ARGS[@]}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  "${TRANSPORT_QUOTIENT_ARGS[@]}" \
  "${PROPER_TIME_ARGS[@]}" \
  "${KEYFRAME_ALIGNMENT_ARGS[@]}" \
  "${KEYFRAME_RESIDUAL_ARGS[@]}" \
  "${SAMPLER_ALIGNMENT_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "${DETERMINISTIC_ARGS[@]}"
