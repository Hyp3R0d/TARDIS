"""Finite causal state with innovation-aware short and anchor memories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True, slots=True)
class CausalState:
    latent: torch.Tensor
    short: torch.Tensor
    anchor: torch.Tensor
    innovation_hazard: torch.Tensor
    frame_index: torch.Tensor

    def spatial_condition(self) -> torch.Tensor:
        return self.short

    def anchor_tokens(self, *, stride: int) -> torch.Tensor:
        if stride <= 0:
            raise ValueError("anchor token stride must be positive")
        pooled = functional.avg_pool2d(self.anchor, kernel_size=stride, stride=stride)
        return pooled.flatten(2).transpose(1, 2)


class CausalStateUpdater(nn.Module):
    """Maintain constant-memory short state and confidence-weighted EMA anchor."""

    def __init__(
        self,
        *,
        latent_channels: int,
        state_channels: int,
        anchor_decay: float,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or state_channels <= 0:
            raise ValueError("state channel counts must be positive")
        if not 0 <= anchor_decay < 1:
            raise ValueError("anchor_decay must be in [0, 1)")
        self.anchor_decay = anchor_decay
        self.encoder = nn.Conv2d(latent_channels, state_channels, 1)

    def initialize(self, latent: torch.Tensor, *, detach: bool) -> CausalState:
        self._validate_latent(latent)
        features = self.encoder(latent)
        state = CausalState(
            latent=latent,
            short=features,
            anchor=features,
            innovation_hazard=torch.zeros(
                latent.shape[0],
                1,
                *latent.shape[-2:],
                device=latent.device,
                dtype=latent.dtype,
            ),
            frame_index=torch.zeros(latent.shape[0], device=latent.device, dtype=torch.int64),
        )
        return _detach_state(state) if detach else state

    def update(
        self,
        previous: CausalState,
        current_latent: torch.Tensor,
        *,
        innovation_probability: torch.Tensor,
        innovation_hazard: torch.Tensor | None = None,
        reset_mask: torch.Tensor,
        detach: bool,
    ) -> CausalState:
        self._validate_latent(current_latent)
        if previous.latent.shape != current_latent.shape:
            raise ValueError("causal state and current latent must share shape")
        if innovation_probability.shape != (
            current_latent.shape[0],
            1,
            *current_latent.shape[-2:],
        ):
            raise ValueError("innovation_probability must have shape [B,1,H,W]")
        if reset_mask.shape != (current_latent.shape[0],) or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be bool [B]")
        if (
            innovation_hazard is not None
            and innovation_hazard.shape != innovation_probability.shape
        ):
            raise ValueError("innovation_hazard must have shape [B,1,H,W]")

        current_features = self.encoder(current_latent)
        innovation = innovation_probability.clamp(0, 1)
        short = (1 - innovation) * previous.short + innovation * current_features
        anchor_rate = (1 - self.anchor_decay) * (1 - innovation)
        anchor = (1 - anchor_rate) * previous.anchor + anchor_rate * current_features

        reset = reset_mask[:, None, None, None]
        short = torch.where(reset, current_features, short)
        anchor = torch.where(reset, current_features, anchor)
        next_hazard = previous.innovation_hazard if innovation_hazard is None else innovation_hazard
        next_hazard = torch.where(reset, torch.zeros_like(next_hazard), next_hazard)
        frame_index = torch.where(
            reset_mask,
            torch.zeros_like(previous.frame_index),
            previous.frame_index + 1,
        )
        state = CausalState(
            latent=current_latent,
            short=short,
            anchor=anchor,
            innovation_hazard=next_hazard,
            frame_index=frame_index,
        )
        return _detach_state(state) if detach else state

    def _validate_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim != 4 or latent.shape[1] != self.encoder.in_channels:
            raise ValueError("latent must have shape [B,C,H,W] matching updater channels")


def _detach_state(state: CausalState) -> CausalState:
    return CausalState(
        latent=state.latent.detach(),
        short=state.short.detach(),
        anchor=state.anchor.detach(),
        innovation_hazard=state.innovation_hazard.detach(),
        frame_index=state.frame_index.detach(),
    )
