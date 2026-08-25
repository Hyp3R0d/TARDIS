"""Import-light argument helpers shared by generation commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tardis.cli.common import ModelOptions, RuntimeOptions


def add_model_arguments(parser: argparse.ArgumentParser, *, include_num_frames: bool) -> None:
    model = ModelOptions()
    group = parser.add_argument_group("model")
    group.add_argument("--pretrained-model", default=model.pretrained_model)
    group.add_argument("--height", type=int, default=model.height)
    group.add_argument("--width", type=int, default=model.width)
    if include_num_frames:
        group.add_argument("--num-frames", type=int, default=model.num_frames)
    group.add_argument("--fps", type=int, default=model.fps)
    group.add_argument("--latent-channels", type=int, default=model.latent_channels)
    group.add_argument("--patch-size", type=int, default=model.patch_size)
    group.add_argument("--hidden-size", type=int, default=model.hidden_size)
    group.add_argument("--num-layers", type=int, default=model.num_layers)
    group.add_argument("--num-heads", type=int, default=model.num_heads)
    group.add_argument("--active-ratio", type=float, default=model.active_ratio)
    group.add_argument(
        "--motion-max-flow-pixels",
        type=float,
        default=model.motion_max_flow_pixels,
    )
    group.add_argument(
        "--transport-max-correction-pixels",
        type=float,
        default=model.transport_max_correction_pixels,
    )
    group.add_argument(
        "--transport-history-fallback-weight",
        type=float,
        default=model.transport_history_fallback_weight,
    )
    group.add_argument("--router-threshold", type=float, default=model.router_threshold)
    group.add_argument("--router-halo-radius", type=int, default=model.router_halo_radius)
    group.add_argument("--state-anchor-decay", type=float, default=model.state_anchor_decay)
    group.add_argument("--scene-cut-threshold", type=float, default=model.scene_cut_threshold)
    group.add_argument("--oracle-temperature", type=float, default=model.oracle_temperature)
    group.add_argument("--training-noise-scale", type=float, default=model.training_noise_scale)
    group.add_argument("--lite-max-magnitude", type=float, default=model.lite_max_magnitude)
    group.add_argument(
        "--keyframe-lite-alignment",
        action=argparse.BooleanOptionalAction,
        default=model.keyframe_lite_alignment,
    )
    group.add_argument(
        "--keyframe-residual-generation",
        action=argparse.BooleanOptionalAction,
        default=model.keyframe_residual_generation,
    )
    group.add_argument(
        "--transport-quotient",
        action=argparse.BooleanOptionalAction,
        default=model.transport_quotient,
    )
    group.add_argument(
        "--quotient-regularization",
        type=float,
        default=model.quotient_regularization,
    )
    group.add_argument(
        "--quotient-rank-threshold",
        type=float,
        default=model.quotient_rank_threshold,
    )
    group.add_argument(
        "--innovation-proper-time",
        action=argparse.BooleanOptionalAction,
        default=model.innovation_proper_time,
    )
    group.add_argument(
        "--proper-time-maximum-hazard",
        type=float,
        default=model.proper_time_maximum_hazard,
    )
    group.add_argument("--diffusion-steps", type=int, default=model.diffusion_steps)
    group.add_argument(
        "--diffusion-time-sampling",
        choices=("uniform", "high_noise", "endpoint"),
        default=model.diffusion_time_sampling,
    )
    group.add_argument(
        "--sampler-trajectory-alignment",
        action=argparse.BooleanOptionalAction,
        default=model.sampler_trajectory_alignment,
    )
    group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=model.gradient_checkpointing,
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    runtime = RuntimeOptions()
    group = parser.add_argument_group("runtime")
    group.add_argument("--seed", type=int, default=runtime.seed)
    group.add_argument("--output-root", type=Path, default=runtime.output_root)
    group.add_argument("--checkpoint-root", type=Path, default=runtime.checkpoint_root)
    group.add_argument("--checkpoint", type=Path, default=None)
    group.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default=runtime.precision,
    )
    group.add_argument("--device", default=runtime.device)
    group.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=runtime.compile_model,
    )
    group.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=runtime.deterministic,
    )


def validate_model_arguments(args: argparse.Namespace) -> None:
    dimensions = [args.height, args.width, args.fps, args.latent_channels, args.patch_size]
    dimensions.extend((args.hidden_size, args.num_layers, args.num_heads, args.diffusion_steps))
    if hasattr(args, "num_frames"):
        dimensions.append(args.num_frames)
    if min(int(value) for value in dimensions) <= 0:
        raise ValueError("model dimensions, frame rate, and diffusion steps must be positive")
    if not 0.0 < float(args.active_ratio) <= 1.0:
        raise ValueError("active_ratio must be in (0, 1]")
    if float(args.motion_max_flow_pixels) <= 0 or float(args.lite_max_magnitude) <= 0:
        raise ValueError("motion and lite bounds must be positive")
    if float(args.transport_max_correction_pixels) < 0:
        raise ValueError("transport correction bound cannot be negative")
    if not 0.0 <= float(args.transport_history_fallback_weight) <= 1.0:
        raise ValueError("transport history fallback weight must be in [0, 1]")
    if float(args.router_threshold) < 0 or int(args.router_halo_radius) < 0:
        raise ValueError("router threshold and halo radius cannot be negative")
    if not 0.0 <= float(args.state_anchor_decay) < 1.0:
        raise ValueError("state_anchor_decay must be in [0, 1)")
    if not 0.0 < float(args.scene_cut_threshold) <= 1.0:
        raise ValueError("scene_cut_threshold must be in (0, 1]")
    if float(args.oracle_temperature) <= 0 or float(args.training_noise_scale) < 0:
        raise ValueError("oracle temperature and training noise scale are invalid")
    if float(args.quotient_regularization) <= 0 or float(args.quotient_rank_threshold) <= 0:
        raise ValueError("quotient regularization and rank threshold must be positive")
    if float(args.proper_time_maximum_hazard) <= 0:
        raise ValueError("proper_time_maximum_hazard must be positive")
    if bool(args.sampler_trajectory_alignment) and args.diffusion_time_sampling != "endpoint":
        raise ValueError("sampler trajectory alignment requires endpoint diffusion time sampling")


def validate_generated_video(
    video: Any,
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
) -> Any:
    """Require generated pixels to match the requested production geometry."""

    import torch

    expected = (batch_size, num_frames, 3, height, width)
    if not isinstance(video, torch.Tensor) or tuple(video.shape) != expected:
        actual = None if not isinstance(video, torch.Tensor) else tuple(video.shape)
        raise RuntimeError(f"generated video has shape {actual}; expected {expected}")
    if not video.is_floating_point() or not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("generated video must contain finite floating-point pixels")
    return video
