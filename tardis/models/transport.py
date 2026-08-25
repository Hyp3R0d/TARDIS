"""TAR transport alignment for generated latents and finite causal state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True, slots=True)
class TransportOutput:
    prior: torch.Tensor
    warped_latent: torch.Tensor
    warped_state: dict[str, torch.Tensor]
    corrected_flow: torch.Tensor
    effective_visibility: torch.Tensor
    valid_mask: torch.Tensor


def flow_to_sampling_grid(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert backward pixel flow to an align-corners-false sampling grid."""

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("backward flow must have shape [B,2,H,W]")
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = x[None].expand(batch, -1, -1) + flow[:, 0]
    sample_y = y[None].expand(batch, -1, -1) + flow[:, 1]
    grid_x = 2 * (sample_x + 0.5) / width - 1
    grid_y = 2 * (sample_y + 0.5) / height - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    valid = (
        (sample_x >= 0) & (sample_x <= width - 1) & (sample_y >= 0) & (sample_y <= height - 1)
    ).unsqueeze(1)
    return grid, valid


def warp_tensor(
    tensor: torch.Tensor, backward_flow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp ``tensor`` into current coordinates using backward flow in pixel units."""

    if tensor.ndim != 4:
        raise ValueError("tensor to warp must have shape [B,C,H,W]")
    if tensor.shape[0] != backward_flow.shape[0] or tensor.shape[-2:] != backward_flow.shape[-2:]:
        raise ValueError("tensor and flow must share batch and spatial dimensions")
    grid, valid = flow_to_sampling_grid(backward_flow)
    warped = functional.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped, valid


class MotionStateTransport(nn.Module):
    """Construct a visibility-weighted reusable prior from generated history."""

    def __init__(
        self,
        *,
        channels: int,
        max_correction_pixels: float,
        history_fallback_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or max_correction_pixels < 0:
            raise ValueError("channels must be positive and correction bound non-negative")
        if not 0 <= history_fallback_weight <= 1:
            raise ValueError("history_fallback_weight must be in [0, 1]")
        self.max_correction_pixels = max_correction_pixels
        self.history_fallback_weight = history_fallback_weight
        self.null_latent = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        previous_latent: torch.Tensor,
        backward_flow: torch.Tensor,
        visibility: torch.Tensor,
        *,
        raw_correction: torch.Tensor | None = None,
        state: Mapping[str, torch.Tensor] | None = None,
    ) -> TransportOutput:
        if previous_latent.ndim != 4 or previous_latent.shape[1] != self.null_latent.shape[1]:
            raise ValueError("previous_latent has incompatible [B,C,H,W] geometry")
        if visibility.shape != (previous_latent.shape[0], 1, *previous_latent.shape[-2:]):
            raise ValueError("visibility must have shape [B,1,H,W]")
        if raw_correction is None:
            correction = torch.zeros_like(backward_flow)
        else:
            if raw_correction.shape != backward_flow.shape:
                raise ValueError("raw_correction must match backward_flow")
            correction = torch.tanh(raw_correction) * self.max_correction_pixels
        corrected_flow = backward_flow + correction
        warped_latent, valid = warp_tensor(previous_latent, corrected_flow)
        effective_visibility = visibility.clamp(0, 1) * valid.to(visibility.dtype)
        null = self.null_latent.to(previous_latent.dtype).expand_as(previous_latent)
        fallback = torch.lerp(null, previous_latent, self.history_fallback_weight)
        prior = effective_visibility * warped_latent + (1 - effective_visibility) * fallback

        warped_state: dict[str, torch.Tensor] = {}
        for name, value in (state or {}).items():
            warped, _ = warp_tensor(value, corrected_flow)
            warped_state[name] = warped
        return TransportOutput(
            prior=prior,
            warped_latent=warped_latent,
            warped_state=warped_state,
            corrected_flow=corrected_flow,
            effective_visibility=effective_visibility,
            valid_mask=valid,
        )
