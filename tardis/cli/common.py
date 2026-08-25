"""Typed command-line defaults shared by all production interfaces."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tardis.training.modes import TRAIN_MODES
from tardis.utils.defaults import DEFAULT_TARDIS_ARCHITECTURE

DATASET_CHOICES = ("dataverse", "openvid", "seedance")


@dataclass(frozen=True, slots=True)
class DataOptions:
    dataset: str = "dataverse"
    datasets_file: Path = Path("datasets.txt")
    mirror_endpoint: str = "https://hf-mirror.com"
    validation_size: int = 256
    test_size: int = 512
    split_seed: int = 3407
    num_workers: int = 8
    prefetch_factor: int = 4
    request_timeout_seconds: float = 60.0
    max_retries: int = 3
    # Production defaults consume every record available in the pinned local
    # subsets. Optional limits remain available for explicit diagnostics only.
    catalog_record_limit: int | None = None
    openvid_archive_limit: int | None = None
    dataverse_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelOptions:
    pretrained_model: str = "stabilityai/sd-turbo"
    height: int = DEFAULT_TARDIS_ARCHITECTURE.height
    width: int = DEFAULT_TARDIS_ARCHITECTURE.width
    num_frames: int = 16
    fps: int = 30
    latent_channels: int = DEFAULT_TARDIS_ARCHITECTURE.motion_noise_channels
    patch_size: int = DEFAULT_TARDIS_ARCHITECTURE.patch_size
    hidden_size: int = DEFAULT_TARDIS_ARCHITECTURE.residual_hidden_size
    num_layers: int = DEFAULT_TARDIS_ARCHITECTURE.residual_layers
    num_heads: int = DEFAULT_TARDIS_ARCHITECTURE.residual_heads
    active_ratio: float = DEFAULT_TARDIS_ARCHITECTURE.active_ratio
    motion_max_flow_pixels: float = DEFAULT_TARDIS_ARCHITECTURE.motion_max_flow_pixels
    transport_max_correction_pixels: float = (
        DEFAULT_TARDIS_ARCHITECTURE.transport_max_correction_pixels
    )
    transport_history_fallback_weight: float = (
        DEFAULT_TARDIS_ARCHITECTURE.transport_history_fallback_weight
    )
    router_threshold: float = DEFAULT_TARDIS_ARCHITECTURE.router_threshold
    router_halo_radius: int = DEFAULT_TARDIS_ARCHITECTURE.router_halo_radius
    state_anchor_decay: float = DEFAULT_TARDIS_ARCHITECTURE.state_anchor_decay
    scene_cut_threshold: float = DEFAULT_TARDIS_ARCHITECTURE.scene_cut_threshold
    oracle_temperature: float = DEFAULT_TARDIS_ARCHITECTURE.oracle_temperature
    training_noise_scale: float = DEFAULT_TARDIS_ARCHITECTURE.training_noise_scale
    lite_max_magnitude: float = DEFAULT_TARDIS_ARCHITECTURE.lite_max_magnitude
    keyframe_lite_alignment: bool = DEFAULT_TARDIS_ARCHITECTURE.keyframe_lite_alignment
    keyframe_residual_generation: bool = (
        DEFAULT_TARDIS_ARCHITECTURE.keyframe_residual_generation
    )
    transport_quotient: bool = DEFAULT_TARDIS_ARCHITECTURE.transport_quotient
    quotient_regularization: float = DEFAULT_TARDIS_ARCHITECTURE.quotient_regularization
    quotient_rank_threshold: float = DEFAULT_TARDIS_ARCHITECTURE.quotient_rank_threshold
    innovation_proper_time: bool = DEFAULT_TARDIS_ARCHITECTURE.innovation_proper_time
    proper_time_maximum_hazard: float = DEFAULT_TARDIS_ARCHITECTURE.proper_time_maximum_hazard
    diffusion_steps: int = DEFAULT_TARDIS_ARCHITECTURE.diffusion_steps
    diffusion_time_sampling: str = DEFAULT_TARDIS_ARCHITECTURE.diffusion_time_sampling
    sampler_trajectory_alignment: bool = (
        DEFAULT_TARDIS_ARCHITECTURE.sampler_trajectory_alignment
    )
    gradient_checkpointing: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    seed: int = 3407
    output_root: Path = Path("outputs")
    checkpoint_root: Path = Path("checkpoints")
    precision: str = "bf16"
    device: str = "cuda"
    compile_model: bool = False
    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class TrainOptions:
    train_mode: str = "full_temporal"
    epochs: int = 20
    # A budgeted optimization epoch keeps the 20-epoch first-pass campaign bounded.
    # The loader still samples only the selected dataset and remains resumable.
    steps_per_epoch: int | None = 64
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 64
    validation_interval: int = 1
    checkpoint_interval_steps: int = 256
    validation_batch_size: int = 8
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    tc_loss_weight: float = 5.0
    lpips_loss_weight: float = 3.0
    diffusion_loss_weight: float = 1.0
    keyframe_loss_weight: float = 1.0
    residual_loss_weight: float = 1.0
    transport_loss_weight: float = 1.0
    flow_loss_weight: float = 0.1
    visibility_loss_weight: float = 0.1
    router_loss_weight: float = 0.2
    survival_loss_weight: float = 0.2
    lite_loss_weight: float = 0.2
    budget_loss_weight: float = 0.05
    warp_loss_weight: float = 0.2
    drift_loss_weight: float = 0.1
    crcd_loss_weight: float = 1.0
    text_loss_weight: float = 0.1
    lpips_frame_chunk_size: int = 4
    curriculum_profile: str = "full"
    resume: Path | None = None
    warm_start: Path | None = None
    warm_start_use_ema: bool = True
    allow_cross_dataset_warm_start: bool = False


@dataclass(frozen=True, slots=True)
class InferOptions:
    checkpoint: Path | None = None
    showcase_count: int = 5
    resume_metrics: bool = True


@dataclass(frozen=True, slots=True)
class ApplyOptions:
    prompt: str = "A robot running in the forest"
    negative_prompt: str = ""
    duration_seconds: float = 2.0


def build_parser() -> argparse.ArgumentParser:
    """Build a light parser without importing model or CUDA modules."""
    data = DataOptions()
    model = ModelOptions()
    runtime = RuntimeOptions()
    train = TrainOptions()
    infer = InferOptions()
    apply = ApplyOptions()

    parser = argparse.ArgumentParser(description="TARDIS prompt-to-video runtime")
    data_group = parser.add_argument_group("local data")
    data_group.add_argument("--dataset", choices=DATASET_CHOICES, default=data.dataset)
    data_group.add_argument("--datasets-file", type=Path, default=data.datasets_file)
    data_group.add_argument("--mirror-endpoint", default=data.mirror_endpoint)
    data_group.add_argument("--validation-size", type=int, default=data.validation_size)
    data_group.add_argument("--test-size", type=int, default=data.test_size)
    data_group.add_argument("--split-seed", type=int, default=data.split_seed)
    data_group.add_argument("--num-workers", type=int, default=data.num_workers)
    data_group.add_argument("--prefetch-factor", type=int, default=data.prefetch_factor)
    data_group.add_argument(
        "--request-timeout-seconds", type=float, default=data.request_timeout_seconds
    )
    data_group.add_argument("--max-retries", type=int, default=data.max_retries)
    data_group.add_argument("--catalog-record-limit", type=int, default=data.catalog_record_limit)
    data_group.add_argument("--openvid-archive-limit", type=int, default=data.openvid_archive_limit)
    data_group.add_argument(
        "--dataverse-record-ids",
        type=parse_record_ids_csv,
        default=data.dataverse_record_ids,
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--pretrained-model", default=model.pretrained_model)
    model_group.add_argument("--height", type=int, default=model.height)
    model_group.add_argument("--width", type=int, default=model.width)
    model_group.add_argument("--num-frames", type=int, default=model.num_frames)
    model_group.add_argument("--fps", type=int, default=model.fps)
    model_group.add_argument("--latent-channels", type=int, default=model.latent_channels)
    model_group.add_argument("--patch-size", type=int, default=model.patch_size)
    model_group.add_argument("--hidden-size", type=int, default=model.hidden_size)
    model_group.add_argument("--num-layers", type=int, default=model.num_layers)
    model_group.add_argument("--num-heads", type=int, default=model.num_heads)
    model_group.add_argument("--active-ratio", type=float, default=model.active_ratio)
    model_group.add_argument(
        "--motion-max-flow-pixels",
        type=float,
        default=model.motion_max_flow_pixels,
    )
    model_group.add_argument(
        "--transport-max-correction-pixels",
        type=float,
        default=model.transport_max_correction_pixels,
    )
    model_group.add_argument(
        "--transport-history-fallback-weight",
        type=float,
        default=model.transport_history_fallback_weight,
    )
    model_group.add_argument(
        "--router-threshold",
        type=float,
        default=model.router_threshold,
    )
    model_group.add_argument(
        "--router-halo-radius",
        type=int,
        default=model.router_halo_radius,
    )
    model_group.add_argument(
        "--state-anchor-decay",
        type=float,
        default=model.state_anchor_decay,
    )
    model_group.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=model.scene_cut_threshold,
    )
    model_group.add_argument(
        "--oracle-temperature",
        type=float,
        default=model.oracle_temperature,
    )
    model_group.add_argument(
        "--training-noise-scale",
        type=float,
        default=model.training_noise_scale,
    )
    model_group.add_argument(
        "--lite-max-magnitude",
        type=float,
        default=model.lite_max_magnitude,
    )
    model_group.add_argument(
        "--keyframe-lite-alignment",
        action=argparse.BooleanOptionalAction,
        default=model.keyframe_lite_alignment,
    )
    model_group.add_argument(
        "--keyframe-residual-generation",
        action=argparse.BooleanOptionalAction,
        default=model.keyframe_residual_generation,
    )
    model_group.add_argument(
        "--transport-quotient",
        action=argparse.BooleanOptionalAction,
        default=model.transport_quotient,
    )
    model_group.add_argument(
        "--quotient-regularization",
        type=float,
        default=model.quotient_regularization,
    )
    model_group.add_argument(
        "--quotient-rank-threshold",
        type=float,
        default=model.quotient_rank_threshold,
    )
    model_group.add_argument(
        "--innovation-proper-time",
        action=argparse.BooleanOptionalAction,
        default=model.innovation_proper_time,
    )
    model_group.add_argument(
        "--proper-time-maximum-hazard",
        type=float,
        default=model.proper_time_maximum_hazard,
    )
    model_group.add_argument("--diffusion-steps", type=int, default=model.diffusion_steps)
    model_group.add_argument(
        "--diffusion-time-sampling",
        choices=("uniform", "high_noise", "endpoint"),
        default=model.diffusion_time_sampling,
    )
    model_group.add_argument(
        "--sampler-trajectory-alignment",
        action=argparse.BooleanOptionalAction,
        default=model.sampler_trajectory_alignment,
    )
    model_group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=model.gradient_checkpointing,
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--seed", type=int, default=runtime.seed)
    runtime_group.add_argument("--output-root", type=Path, default=runtime.output_root)
    runtime_group.add_argument("--checkpoint-root", type=Path, default=runtime.checkpoint_root)
    runtime_group.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default=runtime.precision
    )
    runtime_group.add_argument("--device", default=runtime.device)
    runtime_group.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=runtime.compile_model,
    )
    runtime_group.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=runtime.deterministic,
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--train-mode", choices=TRAIN_MODES, default=train.train_mode)
    train_group.add_argument("--epochs", type=int, default=train.epochs)
    train_group.add_argument("--steps-per-epoch", type=int, default=train.steps_per_epoch)
    train_group.add_argument("--micro-batch-size", type=int, default=train.micro_batch_size)
    train_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=train.gradient_accumulation_steps,
    )
    train_group.add_argument("--learning-rate", type=float, default=train.learning_rate)
    train_group.add_argument("--weight-decay", type=float, default=train.weight_decay)
    train_group.add_argument("--warmup-steps", type=int, default=train.warmup_steps)
    train_group.add_argument("--validation-interval", type=int, default=train.validation_interval)
    train_group.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=train.checkpoint_interval_steps,
    )
    train_group.add_argument(
        "--validation-batch-size",
        type=int,
        default=train.validation_batch_size,
    )
    train_group.add_argument("--gradient-clip-norm", type=float, default=train.gradient_clip_norm)
    train_group.add_argument("--ema-decay", type=float, default=train.ema_decay)
    train_group.add_argument("--tc-loss-weight", type=float, default=train.tc_loss_weight)
    train_group.add_argument("--lpips-loss-weight", type=float, default=train.lpips_loss_weight)
    train_group.add_argument(
        "--diffusion-loss-weight",
        type=float,
        default=train.diffusion_loss_weight,
    )
    train_group.add_argument(
        "--keyframe-loss-weight",
        type=float,
        default=train.keyframe_loss_weight,
    )
    train_group.add_argument(
        "--residual-loss-weight",
        type=float,
        default=train.residual_loss_weight,
    )
    train_group.add_argument(
        "--transport-loss-weight",
        type=float,
        default=train.transport_loss_weight,
    )
    train_group.add_argument("--flow-loss-weight", type=float, default=train.flow_loss_weight)
    train_group.add_argument(
        "--visibility-loss-weight",
        type=float,
        default=train.visibility_loss_weight,
    )
    train_group.add_argument("--router-loss-weight", type=float, default=train.router_loss_weight)
    train_group.add_argument(
        "--survival-loss-weight",
        type=float,
        default=train.survival_loss_weight,
    )
    train_group.add_argument("--lite-loss-weight", type=float, default=train.lite_loss_weight)
    train_group.add_argument("--budget-loss-weight", type=float, default=train.budget_loss_weight)
    train_group.add_argument("--warp-loss-weight", type=float, default=train.warp_loss_weight)
    train_group.add_argument("--drift-loss-weight", type=float, default=train.drift_loss_weight)
    train_group.add_argument("--crcd-loss-weight", type=float, default=train.crcd_loss_weight)
    train_group.add_argument("--text-loss-weight", type=float, default=train.text_loss_weight)
    train_group.add_argument(
        "--lpips-frame-chunk-size",
        type=int,
        default=train.lpips_frame_chunk_size,
    )
    train_group.add_argument(
        "--curriculum-profile",
        choices=("full", "transport", "closed_loop_motion", "metric_alignment"),
        default=train.curriculum_profile,
    )
    train_group.add_argument("--resume", type=Path, default=train.resume)
    train_group.add_argument("--warm-start", type=Path, default=train.warm_start)
    train_group.add_argument(
        "--warm-start-use-ema",
        action=argparse.BooleanOptionalAction,
        default=train.warm_start_use_ema,
    )
    train_group.add_argument(
        "--allow-cross-dataset-warm-start",
        action=argparse.BooleanOptionalAction,
        default=train.allow_cross_dataset_warm_start,
        help="Allow an explicit weights-only warm start from another dataset checkpoint.",
    )

    infer_group = parser.add_argument_group("inference")
    infer_group.add_argument("--checkpoint", type=Path, default=infer.checkpoint)
    infer_group.add_argument("--showcase-count", type=int, default=infer.showcase_count)
    infer_group.add_argument(
        "--resume-metrics",
        action=argparse.BooleanOptionalAction,
        default=infer.resume_metrics,
    )

    apply_group = parser.add_argument_group("application")
    apply_group.add_argument("--prompt", default=apply.prompt)
    apply_group.add_argument("--negative-prompt", default=apply.negative_prompt)
    apply_group.add_argument("--duration-seconds", type=float, default=apply.duration_seconds)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse shared arguments and reject invalid scientific settings."""
    args = build_parser().parse_args(argv)
    if args.height <= 0 or args.width <= 0 or args.num_frames <= 0 or args.fps <= 0:
        raise ValueError("height, width, num_frames, and fps must be positive")
    if not 0.0 < args.active_ratio <= 1.0:
        raise ValueError("active_ratio must be in (0, 1]")
    if args.motion_max_flow_pixels <= 0:
        raise ValueError("motion_max_flow_pixels must be positive")
    if args.transport_max_correction_pixels < 0:
        raise ValueError("transport_max_correction_pixels cannot be negative")
    if not 0 <= args.transport_history_fallback_weight <= 1:
        raise ValueError("transport_history_fallback_weight must be in [0, 1]")
    if args.router_threshold < 0 or args.router_halo_radius < 0:
        raise ValueError("router threshold and halo radius cannot be negative")
    if not 0 <= args.state_anchor_decay < 1:
        raise ValueError("state_anchor_decay must be in [0, 1)")
    if not 0 < args.scene_cut_threshold <= 1:
        raise ValueError("scene_cut_threshold must be in (0, 1]")
    if args.oracle_temperature <= 0 or args.lite_max_magnitude <= 0:
        raise ValueError("oracle_temperature and lite_max_magnitude must be positive")
    if args.training_noise_scale < 0:
        raise ValueError("training_noise_scale cannot be negative")
    if args.quotient_regularization <= 0 or args.quotient_rank_threshold <= 0:
        raise ValueError("quotient regularization and rank threshold must be positive")
    if args.proper_time_maximum_hazard <= 0:
        raise ValueError("proper_time_maximum_hazard must be positive")
    if args.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive")
    if args.checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")
    loss_weights = (
        args.tc_loss_weight,
        args.lpips_loss_weight,
        args.diffusion_loss_weight,
        args.keyframe_loss_weight,
        args.residual_loss_weight,
        args.transport_loss_weight,
        args.flow_loss_weight,
        args.visibility_loss_weight,
        args.router_loss_weight,
        args.survival_loss_weight,
        args.lite_loss_weight,
        args.budget_loss_weight,
        args.warp_loss_weight,
        args.drift_loss_weight,
        args.crcd_loss_weight,
        args.text_loss_weight,
    )
    if any(weight < 0 for weight in loss_weights):
        raise ValueError("training loss weights must be non-negative")
    if args.lpips_frame_chunk_size <= 0:
        raise ValueError("lpips_frame_chunk_size must be positive")
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")
    return args


def parse_record_ids_csv(value: str) -> tuple[str, ...]:
    record_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise argparse.ArgumentTypeError("record IDs must be a non-empty, unique CSV list")
    return record_ids
