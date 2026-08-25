"""Mechanism-aligned objectives for transport-innovation residual diffusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint


class PerceptualMetric(Protocol):
    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class LossWeights:
    diffusion: float = 1.0
    keyframe: float = 1.0
    residual: float = 1.0
    transport: float = 1.0
    flow: float = 0.1
    visibility: float = 0.1
    router: float = 0.2
    survival: float = 0.2
    lite: float = 0.2
    lpips: float = 0.3
    tc: float = 0.5
    warp: float = 0.2
    text: float = 0.1
    budget: float = 0.05
    drift: float = 0.1
    crcd: float = 1.0

    def __post_init__(self) -> None:
        if any(weight < 0 for weight in asdict(self).values()):
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True, slots=True)
class CRCDTarget:
    teacher_residual: torch.Tensor
    causal_history: torch.Tensor


class EmaLossNormalizer:
    """Normalize one loss by a checkpointable detached EMA magnitude."""

    def __init__(self, *, decay: float = 0.99, epsilon: float = 1.0e-6) -> None:
        if not 0 <= decay < 1 or epsilon <= 0:
            raise ValueError("EMA decay and epsilon are invalid")
        self.decay = decay
        self.epsilon = epsilon
        self.value = 1.0

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        magnitude = max(float(loss.detach().abs().mean().item()), self.epsilon)
        self.value = self.decay * self.value + (1 - self.decay) * magnitude
        return loss / max(self.value, self.epsilon)

    def state_dict(self) -> dict[str, float]:
        return {"decay": self.decay, "epsilon": self.epsilon, "value": self.value}

    def load_state_dict(self, state: Mapping[str, float]) -> None:
        required = {"decay", "epsilon", "value"}
        if set(state) != required:
            raise ValueError("EMA state has an incompatible schema")
        if state["decay"] != self.decay or state["epsilon"] != self.epsilon:
            raise ValueError("EMA state hyperparameters do not match")
        if state["value"] <= 0:
            raise ValueError("EMA value must be positive")
        self.value = float(state["value"])


def diffusion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked flow-matching/diffusion MSE in residual space."""

    _validate_pair(prediction, target)
    return _masked_mean((prediction - target).square(), mask)


def residual_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_pair(prediction, target)
    return _masked_mean((prediction - target).abs(), mask)


def transport_loss(
    transported_prior: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    *,
    flow_correction: torch.Tensor | None = None,
    correction_weight: float = 0.01,
    smoothness_weight: float = 0.01,
) -> torch.Tensor:
    """Reliable-region reconstruction plus bounded-flow regularization."""

    _validate_pair(transported_prior, target)
    reconstruction = _masked_mean((transported_prior - target).abs(), visibility)
    if flow_correction is None:
        return reconstruction
    magnitude = flow_correction.abs().mean()
    horizontal = flow_correction[..., :, 1:] - flow_correction[..., :, :-1]
    vertical = flow_correction[..., 1:, :] - flow_correction[..., :-1, :]
    smoothness = horizontal.abs().mean() + vertical.abs().mean()
    return reconstruction + correction_weight * magnitude + smoothness_weight * smoothness


def flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_pair(prediction, target)
    return _masked_mean((prediction - target).abs(), visibility)


def visibility_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_pair(logits, target)
    return functional.binary_cross_entropy_with_logits(logits, target.clamp(0, 1))


def router_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    brier_weight: float = 1.0,
) -> torch.Tensor:
    _validate_pair(probability, target)
    with torch.autocast(probability.device.type, enabled=False):
        bounded = probability.float().clamp(1.0e-7, 1 - 1.0e-7)
        calibration_target = target.float().clamp(0, 1)
        binary = functional.binary_cross_entropy(bounded, calibration_target)
        brier = (bounded - calibration_target).square().mean()
    return binary + brier_weight * brier


def survival_calibration_loss(
    predicted_probability: torch.Tensor,
    oracle_probability: torch.Tensor,
) -> torch.Tensor:
    """Calibrate cumulative renewal probability, not only one-frame risk."""

    _validate_pair(predicted_probability, oracle_probability)
    with torch.autocast(predicted_probability.device.type, enabled=False):
        predicted = predicted_probability.float().clamp(1.0e-7, 1 - 1.0e-7)
        target = oracle_probability.detach().float().clamp(0, 1)
        binary = functional.binary_cross_entropy(predicted, target)
        brier = (predicted - target).square().mean()
    return binary + brier


def lite_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    _validate_pair(prediction, target)
    return _masked_mean((prediction - target).abs(), 1 - active_mask.to(prediction.dtype))


def budget_loss(probability: torch.Tensor, maximum_ratio: float) -> torch.Tensor:
    if not 0 < maximum_ratio <= 1:
        raise ValueError("maximum_ratio must be in (0, 1]")
    return (probability.mean() - maximum_ratio).clamp_min(0).square()


def official_temporal_consistency_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Official source-motion difference loss averaged over all transitions."""

    _validate_video_pair(generated, reference)
    generated_delta = generated[:, 1:] - generated[:, :-1]
    reference_delta = reference[:, 1:] - reference[:, :-1]
    return (generated_delta - reference_delta).abs().mean()


def multi_scale_temporal_consistency_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    levels: int = 3,
) -> torch.Tensor:
    if levels <= 0:
        raise ValueError("levels must be positive")
    _validate_video_pair(generated, reference)
    losses: list[torch.Tensor] = []
    current_generated = generated
    current_reference = reference
    for level in range(levels):
        losses.append(official_temporal_consistency_loss(current_generated, current_reference))
        if level + 1 == levels:
            break
        if min(current_generated.shape[-2:]) < 2:
            break
        current_generated = _downsample_video(current_generated)
        current_reference = _downsample_video(current_reference)
    return torch.stack(losses).mean()


def lpips_loss(
    metric: PerceptualMetric,
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    frame_chunk_size: int = 1,
) -> torch.Tensor:
    _validate_pair(generated, reference)
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive")
    total = generated.new_zeros(())
    count = 0
    for start in range(0, generated.shape[0], frame_chunk_size):
        generated_chunk = generated[start : start + frame_chunk_size]
        reference_chunk = reference[start : start + frame_chunk_size]
        if torch.is_grad_enabled() and generated_chunk.requires_grad:
            values = checkpoint(
                metric,
                generated_chunk,
                reference_chunk,
                use_reentrant=False,
            )
        else:
            values = metric(generated_chunk, reference_chunk)
        if not isinstance(values, torch.Tensor) or values.numel() == 0:
            raise RuntimeError("perceptual metric must return a non-empty tensor")
        total = total + values.sum()
        count += values.numel()
    return total / count


def warp_loss(
    generated: torch.Tensor,
    warped_previous: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    _validate_pair(generated, warped_previous)
    return _masked_mean((generated - warped_previous).abs(), valid_mask)


def text_alignment_loss(
    visual_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    if visual_embeddings.shape != text_embeddings.shape or visual_embeddings.ndim != 2:
        raise ValueError("visual and text embeddings must share [B,D]")
    similarity = functional.cosine_similarity(visual_embeddings, text_embeddings, dim=-1)
    return torch.ones((), device=similarity.device, dtype=similarity.dtype) - similarity.mean()


def drift_loss(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 5 or latents.shape[1] < 3:
        raise ValueError("drift loss requires [B,T,C,H,W] with at least three frames")
    velocity = latents[:, 1:] - latents[:, :-1]
    return (velocity[:, 1:] - velocity[:, :-1]).abs().mean()


def build_crcd_target(
    teacher_residual: torch.Tensor,
    student_causal_history: torch.Tensor,
) -> CRCDTarget:
    if teacher_residual.ndim != 4 or student_causal_history.ndim != 4:
        raise ValueError("CRCD tensors must use batched spatial geometry")
    if teacher_residual.shape[0::3] != student_causal_history.shape[0::3]:
        raise ValueError("teacher residual and student history must share batch and spatial shape")
    return CRCDTarget(
        teacher_residual=teacher_residual.detach(),
        causal_history=student_causal_history.detach(),
    )


def crcd_loss(student_residual: torch.Tensor, target: CRCDTarget) -> torch.Tensor:
    _validate_pair(student_residual, target.teacher_residual)
    return functional.mse_loss(student_residual, target.teacher_residual)


def weighted_loss(losses: Mapping[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    if not losses:
        raise ValueError("at least one loss is required")
    available = asdict(weights)
    unknown = set(losses) - set(available)
    if unknown:
        raise ValueError(f"unknown loss names: {sorted(unknown)}")
    total: torch.Tensor | None = None
    for name, loss in losses.items():
        if loss.ndim != 0:
            raise ValueError(f"loss {name!r} must be scalar")
        weighted = available[name] * loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise RuntimeError("unreachable empty loss mapping")
    return total


def _validate_pair(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError("paired loss tensors must share shape")


def _validate_video_pair(generated: torch.Tensor, reference: torch.Tensor) -> None:
    _validate_pair(generated, reference)
    if generated.ndim != 5 or generated.shape[1] < 2:
        raise ValueError("temporal loss requires [B,T,C,H,W] with at least two frames")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    try:
        expanded = torch.broadcast_to(mask.to(value.dtype), value.shape)
    except RuntimeError as error:
        raise ValueError("loss mask is not broadcastable to the value tensor") from error
    denominator = expanded.sum().clamp_min(1)
    return (value * expanded).sum() / denominator


def _downsample_video(video: torch.Tensor) -> torch.Tensor:
    batch, frames, channels, height, width = video.shape
    flattened = video.reshape(batch * frames, channels, height, width)
    downsampled = functional.avg_pool2d(flattened, kernel_size=2, stride=2)
    return downsampled.reshape(
        batch,
        frames,
        channels,
        downsampled.shape[-2],
        downsampled.shape[-1],
    )
