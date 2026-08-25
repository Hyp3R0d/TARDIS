"""Temporal diagnostic metrics required by the paper experiment protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as functional

from tardis.metrics.base import validate_video_pair


def temporal_diagnostic_details(
    generated: torch.Tensor,
    reference: torch.Tensor,
    backward_flow: torch.Tensor,
    lpips_feature: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    lag_values: Sequence[int] = (1, 2, 4, 8),
    brightness_threshold: float = 0.1,
) -> dict[str, Any]:
    validate_video_pair(generated, reference, min_frames=2)
    expected_flow_shape = (
        generated.shape[0] - 1,
        2,
        generated.shape[-2],
        generated.shape[-1],
    )
    if tuple(backward_flow.shape) != expected_flow_shape:
        raise ValueError(
            "backward flow must have shape "
            f"{expected_flow_shape}; received {tuple(backward_flow.shape)}"
        )
    if not 0 <= brightness_threshold <= 2:
        raise ValueError("brightness threshold must lie in [0, 2]")

    flow = backward_flow.to(device=generated.device, dtype=generated.dtype)
    if not bool(torch.isfinite(flow).all().item()):
        raise ValueError("backward flow must contain only finite values")
    warped, valid = _warp_with_backward_flow(generated[:-1], flow)
    target = generated[1:]
    valid_rgb = valid.expand_as(target)
    squared_error = (warped - target).square() * valid_rgb
    valid_counts = valid.flatten(1).sum(dim=1) * target.shape[1]
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("backward flow leaves a transition without valid pixels")
    flow_warp_values = squared_error.flatten(1).sum(dim=1) / valid_counts

    lpips_input = torch.where(valid_rgb, warped, target)
    tlpips_values = lpips_feature(lpips_input, target).detach().reshape(-1)
    if (
        tlpips_values.numel() != target.shape[0]
        or not bool(torch.isfinite(tlpips_values).all().item())
        or bool((tlpips_values < 0).any().item())
    ):
        raise ValueError("tLPIPS feature must return one finite non-negative value per transition")

    generated_delta = generated[1:] - generated[:-1]
    reference_delta = reference[1:] - reference[:-1]
    tc_values = (generated_delta - reference_delta).abs().mean(dim=(1, 2, 3))
    brightness = generated.add(1).mul(0.5).mean(dim=(1, 2, 3))
    brightness_delta = (brightness[1:] - brightness[:-1]).abs()
    flicker_flags = brightness_delta > brightness_threshold
    motion_values = flow.square().sum(dim=1).sqrt().mean(dim=(1, 2))

    tc_by_lag: dict[str, float] = {}
    for lag in sorted({int(value) for value in lag_values if int(value) > 0}):
        if lag >= generated.shape[0]:
            continue
        generated_lag_delta = generated[lag:] - generated[:-lag]
        reference_lag_delta = reference[lag:] - reference[:-lag]
        value = (generated_lag_delta - reference_lag_delta).abs().mean()
        tc_by_lag[str(lag)] = float(value.item())

    return {
        "flow_warp_error": float(flow_warp_values.mean().item()),
        "tlpips": float(tlpips_values.mean().item()),
        "flicker_rate": float(flicker_flags.to(torch.float32).mean().item()),
        "drift_slope": _linear_slope(tc_values),
        "motion_magnitude": float(motion_values.mean().item()),
        "tc_by_lag": tc_by_lag,
        "flow_warp_error_per_transition": _float_list(flow_warp_values),
        "tlpips_per_transition": _float_list(tlpips_values),
        "motion_magnitude_per_transition": _float_list(motion_values),
        "brightness_per_frame": _float_list(brightness),
        "brightness_delta_per_transition": _float_list(brightness_delta),
        "flicker_flags": [bool(value) for value in flicker_flags.cpu().tolist()],
    }


def _warp_with_backward_flow(
    source: torch.Tensor,
    backward_flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count, _, height, width = source.shape
    if height < 2 or width < 2:
        raise ValueError("flow warping requires spatial dimensions of at least two pixels")
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0).expand(count, -1, -1) + backward_flow[:, 0]
    y = y.unsqueeze(0).expand(count, -1, -1) + backward_flow[:, 1]
    valid = ((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)).unsqueeze(1)
    grid = torch.stack(
        (
            x.mul(2 / (width - 1)).sub(1),
            y.mul(2 / (height - 1)).sub(1),
        ),
        dim=-1,
    )
    warped = functional.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, valid


def _linear_slope(values: torch.Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    x = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    centered = x - x.mean()
    denominator = centered.square().sum()
    slope = (centered * (values - values.mean())).sum() / denominator
    return float(slope.item())


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]
