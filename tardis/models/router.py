"""Visibility-calibrated innovation probability and hard compute routing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True, slots=True)
class InnovationSelection:
    indices: torch.Tensor
    valid_tokens: torch.Tensor
    active_counts: torch.Tensor
    active_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RouterOutput:
    logits: torch.Tensor
    pixel_probability: torch.Tensor
    patch_probability: torch.Tensor
    selection: InnovationSelection


def oracle_innovation(
    target_latent: torch.Tensor,
    transported_prior: torch.Tensor,
    visibility: torch.Tensor,
    *,
    residual_temperature: float,
    detach_transport: bool = True,
    quotient_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct soft innovation risk from quotient-normal failure energy."""

    if target_latent.shape != transported_prior.shape or target_latent.ndim != 4:
        raise ValueError("target and transported latent must share [B,C,H,W]")
    if visibility.shape != (target_latent.shape[0], 1, *target_latent.shape[-2:]):
        raise ValueError("visibility must have shape [B,1,H,W]")
    if residual_temperature <= 0:
        raise ValueError("residual_temperature must be positive")
    if quotient_residual is None:
        residual_base = transported_prior.detach() if detach_transport else transported_prior
        residual = target_latent - residual_base
    else:
        if quotient_residual.shape != target_latent.shape:
            raise ValueError("quotient_residual must match target_latent")
        residual = quotient_residual.detach() if detach_transport else quotient_residual
    residual_energy = residual.abs().mean(dim=1, keepdim=True)
    confidence = visibility.clamp(0, 1) * torch.exp(-residual_energy / residual_temperature)
    result: torch.Tensor = 1 - confidence
    return result


def select_innovation_budget(
    patch_scores: torch.Tensor,
    *,
    active_ratio: float,
    threshold: float,
    halo_radius: int,
) -> InnovationSelection:
    """Deterministically choose thresholded Top-K patches with halo inside K."""

    if patch_scores.ndim != 4 or patch_scores.shape[1] != 1:
        raise ValueError("patch_scores must have shape [B,1,H_p,W_p]")
    if not 0 < active_ratio <= 1:
        raise ValueError("active_ratio must be in (0, 1]")
    if halo_radius < 0:
        raise ValueError("halo_radius cannot be negative")
    batch, _, height, width = patch_scores.shape
    token_count = height * width
    budget = math.ceil(active_ratio * token_count)
    seed_scores = patch_scores.masked_fill(patch_scores <= threshold, -torch.inf)
    if halo_radius > 0:
        propagated_scores = functional.max_pool2d(
            seed_scores,
            kernel_size=2 * halo_radius + 1,
            stride=1,
            padding=halo_radius,
        )
    else:
        propagated_scores = seed_scores
    flat_scores = propagated_scores.flatten(1)
    ranked = torch.argsort(flat_scores, dim=1, descending=True, stable=True)
    indices = ranked[:, :budget]
    selected_scores = flat_scores.gather(1, indices)
    valid_tokens = selected_scores.isfinite()
    masks = torch.zeros(batch, token_count, device=patch_scores.device, dtype=torch.bool)
    masks.scatter_(1, indices, valid_tokens)
    return InnovationSelection(
        indices=indices,
        valid_tokens=valid_tokens,
        active_counts=valid_tokens.sum(dim=1),
        active_mask=masks.reshape(batch, 1, height, width),
    )


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must share shape")
    return (probabilities - targets).square().mean()


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int,
) -> torch.Tensor:
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must share shape")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    probabilities = probabilities.flatten().clamp(0, 1)
    targets = targets.flatten().to(probabilities.dtype)
    boundaries = torch.linspace(0, 1, num_bins + 1, device=probabilities.device)
    ece = torch.zeros((), device=probabilities.device, dtype=probabilities.dtype)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if index == num_bins - 1 else probabilities < upper
        )
        if in_bin.any():
            weight = in_bin.float().mean()
            ece = ece + weight * (probabilities[in_bin].mean() - targets[in_bin].mean()).abs()
    return ece


class VisibilityCalibratedInnovationRouter(nn.Module):
    """Predict one innovation probability controlling reuse and sparse compute."""

    def __init__(
        self,
        *,
        latent_channels: int,
        motion_channels: int,
        state_channels: int,
        text_dim: int,
        hidden_size: int,
        patch_size: int,
        active_ratio: float,
        threshold: float,
        halo_radius: int,
    ) -> None:
        super().__init__()
        if min(latent_channels, motion_channels, state_channels, text_dim, hidden_size) <= 0:
            raise ValueError("router dimensions must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.active_ratio = active_ratio
        self.threshold = threshold
        self.halo_radius = halo_radius
        input_channels = latent_channels + 1 + motion_channels + state_channels
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_size), hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, 1, 1),
        )
        self.text_to_logit = nn.Linear(hidden_size, 1)

    def forward(
        self,
        transported_prior: torch.Tensor,
        visibility: torch.Tensor,
        motion_features: torch.Tensor,
        state_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> RouterOutput:
        spatial = transported_prior.shape[-2:]
        tensors = (visibility, motion_features, state_features)
        if transported_prior.ndim != 4 or any(
            tensor.ndim != 4
            or tensor.shape[0] != transported_prior.shape[0]
            or tensor.shape[-2:] != spatial
            for tensor in tensors
        ):
            raise ValueError("router spatial conditions must share [B,H,W]")
        if text_embeddings.ndim != 3 or text_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("text inputs must be [B,L,D] and bool mask [B,L]")
        weights = text_mask.unsqueeze(-1).to(text_embeddings.dtype)
        pooled = (text_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
        text_condition = self.text_projection(pooled)
        features = torch.cat(
            (transported_prior, visibility, motion_features, state_features), dim=1
        )
        logits = self.network(features) + self.text_to_logit(text_condition)[:, :, None, None]
        probability = logits.sigmoid()
        patch_probability = functional.avg_pool2d(
            probability,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        selection = select_innovation_budget(
            patch_probability.detach(),
            active_ratio=self.active_ratio,
            threshold=self.threshold,
            halo_radius=self.halo_radius,
        )
        return RouterOutput(
            logits=logits,
            pixel_probability=probability,
            patch_probability=patch_probability,
            selection=selection,
        )


def _group_count(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1
