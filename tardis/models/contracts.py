"""Typed boundaries between frozen priors and the trainable temporal network."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LatentCodec(Protocol):
    latent_channels: int
    spatial_scale: int

    def encode(self, video: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class TextConditioner(Protocol):
    embedding_dim: int
    max_length: int

    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class FirstFrameGenerator(Protocol):
    def generate_first_latent(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        generator: torch.Generator | Sequence[torch.Generator],
        height: int,
        width: int,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class MotionTargets:
    """Adjacent-frame motion supervision in output-grid pixel units."""

    forward_flow: torch.Tensor
    backward_flow: torch.Tensor
    visibility: torch.Tensor


@runtime_checkable
class MotionTeacher(Protocol):
    def estimate(
        self,
        video: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> MotionTargets: ...
