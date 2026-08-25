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

CATALOG_ARGS=()
[[ -n "${TARDIS_CATALOG_RECORD_LIMIT:-}" ]] && CATALOG_ARGS+=(--catalog-record-limit "${TARDIS_CATALOG_RECORD_LIMIT}")
[[ -n "${TARDIS_OPENVID_ARCHIVE_LIMIT:-}" ]] && CATALOG_ARGS+=(--openvid-archive-limit "${TARDIS_OPENVID_ARCHIVE_LIMIT}")
RESUME_ARGS=()
[[ -n "${TARDIS_RESUME:-}" ]] && RESUME_ARGS+=(--resume "${TARDIS_RESUME}")
WARM_START_ARGS=()
if [[ -n "${TARDIS_WARM_START:-}" ]]; then
  WARM_START_ARGS+=(--warm-start "${TARDIS_WARM_START}")
  if [[ "${TARDIS_WARM_START_USE_EMA:-1}" == "0" ]]; then
    WARM_START_ARGS+=(--no-warm-start-use-ema)
  else
    WARM_START_ARGS+=(--warm-start-use-ema)
  fi
fi
GRADIENT_CHECKPOINTING_ARGS=(--gradient-checkpointing)
[[ "${TARDIS_GRADIENT_CHECKPOINTING:-1}" == "0" ]] && GRADIENT_CHECKPOINTING_ARGS=(--no-gradient-checkpointing)
TRANSPORT_QUOTIENT_ARGS=(--transport-quotient)
[[ "${TARDIS_TRANSPORT_QUOTIENT:-1}" == "0" ]] && TRANSPORT_QUOTIENT_ARGS=(--no-transport-quotient)
PROPER_TIME_ARGS=(--innovation-proper-time)
[[ "${TARDIS_INNOVATION_PROPER_TIME:-1}" == "0" ]] && PROPER_TIME_ARGS=(--no-innovation-proper-time)
COMPILE_ARGS=(--no-compile-model)
[[ "${TARDIS_COMPILE_MODEL:-0}" == "1" ]] && COMPILE_ARGS=(--compile-model)
DETERMINISTIC_ARGS=(--no-deterministic)
[[ "${TARDIS_DETERMINISTIC:-0}" == "1" ]] && DETERMINISTIC_ARGS=(--deterministic)
KEYFRAME_ALIGNMENT_ARGS=(--keyframe-lite-alignment)
[[ "${TARDIS_KEYFRAME_LITE_ALIGNMENT:-1}" == "0" ]] && KEYFRAME_ALIGNMENT_ARGS=(--no-keyframe-lite-alignment)
KEYFRAME_RESIDUAL_ARGS=(--keyframe-residual-generation)
[[ "${TARDIS_KEYFRAME_RESIDUAL_GENERATION:-1}" == "0" ]] && KEYFRAME_RESIDUAL_ARGS=(--no-keyframe-residual-generation)
SAMPLER_ALIGNMENT_ARGS=(--sampler-trajectory-alignment)
[[ "${TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT:-1}" == "0" ]] && SAMPLER_ALIGNMENT_ARGS=(--no-sampler-trajectory-alignment)

torchrun --standalone --nproc_per_node="${TARDIS_NPROC:-1}" -m tardis.cli.train \
  --dataset "${TARDIS_DATASET:-dataverse}" \
  --datasets-file "${TARDIS_DATASETS_FILE:-${REPO_ROOT}/datasets.txt}" \
  --mirror-endpoint "${TARDIS_MIRROR_ENDPOINT:-https://hf-mirror.com}" \
  --pretrained-model "${TARDIS_PRETRAINED_MODEL:-stabilityai/sd-turbo}" \
  --train-mode "${TARDIS_TRAIN_MODE:-full_temporal}" \
  --epochs "${TARDIS_EPOCHS:-20}" \
  --steps-per-epoch "${TARDIS_STEPS_PER_EPOCH:-64}" \
  --micro-batch-size "${TARDIS_MICRO_BATCH_SIZE:-2}" \
  --gradient-accumulation-steps "${TARDIS_GRADIENT_ACCUMULATION_STEPS:-2}" \
  --learning-rate "${TARDIS_LEARNING_RATE:-1e-4}" \
  --weight-decay "${TARDIS_WEIGHT_DECAY:-1e-2}" \
  --warmup-steps "${TARDIS_WARMUP_STEPS:-64}" \
  --checkpoint-interval-steps "${TARDIS_CHECKPOINT_INTERVAL_STEPS:-256}" \
  --gradient-clip-norm "${TARDIS_GRADIENT_CLIP_NORM:-1.0}" \
  --ema-decay "${TARDIS_EMA_DECAY:-0.999}" \
  --tc-loss-weight "${TARDIS_TC_LOSS_WEIGHT:-5.0}" \
  --lpips-loss-weight "${TARDIS_LPIPS_LOSS_WEIGHT:-3.0}" \
  --diffusion-loss-weight "${TARDIS_DIFFUSION_LOSS_WEIGHT:-1.0}" \
  --keyframe-loss-weight "${TARDIS_KEYFRAME_LOSS_WEIGHT:-1.0}" \
  --residual-loss-weight "${TARDIS_RESIDUAL_LOSS_WEIGHT:-1.0}" \
  --transport-loss-weight "${TARDIS_TRANSPORT_LOSS_WEIGHT:-1.0}" \
  --flow-loss-weight "${TARDIS_FLOW_LOSS_WEIGHT:-0.1}" \
  --visibility-loss-weight "${TARDIS_VISIBILITY_LOSS_WEIGHT:-0.1}" \
  --router-loss-weight "${TARDIS_ROUTER_LOSS_WEIGHT:-0.2}" \
  --survival-loss-weight "${TARDIS_SURVIVAL_LOSS_WEIGHT:-0.2}" \
  --lite-loss-weight "${TARDIS_LITE_LOSS_WEIGHT:-0.2}" \
  --budget-loss-weight "${TARDIS_BUDGET_LOSS_WEIGHT:-0.05}" \
  --warp-loss-weight "${TARDIS_WARP_LOSS_WEIGHT:-0.2}" \
  --drift-loss-weight "${TARDIS_DRIFT_LOSS_WEIGHT:-0.1}" \
  --crcd-loss-weight "${TARDIS_CRCD_LOSS_WEIGHT:-1.0}" \
  --text-loss-weight "${TARDIS_TEXT_LOSS_WEIGHT:-0.1}" \
  --lpips-frame-chunk-size "${TARDIS_LPIPS_FRAME_CHUNK_SIZE:-4}" \
  --curriculum-profile "${TARDIS_CURRICULUM_PROFILE:-full}" \
  --validation-interval "${TARDIS_VALIDATION_INTERVAL:-1}" \
  --validation-batch-size "${TARDIS_VALIDATION_BATCH_SIZE:-8}" \
  --validation-size "${TARDIS_VALIDATION_SIZE:-256}" \
  --test-size "${TARDIS_TEST_SIZE:-512}" \
  --split-seed "${TARDIS_SPLIT_SEED:-3407}" \
  --seed "${TARDIS_SEED:-3407}" \
  --num-workers "${TARDIS_NUM_WORKERS:-8}" \
  --prefetch-factor "${TARDIS_PREFETCH_FACTOR:-4}" \
  --height "${TARDIS_HEIGHT:-512}" \
  --width "${TARDIS_WIDTH:-512}" \
  --num-frames "${TARDIS_NUM_FRAMES:-16}" \
  --fps "${TARDIS_FPS:-30}" \
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
  "${CATALOG_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${WARM_START_ARGS[@]}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  "${TRANSPORT_QUOTIENT_ARGS[@]}" \
  "${PROPER_TIME_ARGS[@]}" \
  "${KEYFRAME_ALIGNMENT_ARGS[@]}" \
  "${KEYFRAME_RESIDUAL_ARGS[@]}" \
  "${SAMPLER_ALIGNMENT_ARGS[@]}" \
  "${COMPILE_ARGS[@]}" \
  "${DETERMINISTIC_ARGS[@]}"
