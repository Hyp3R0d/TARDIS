"""Streaming paired-reference metrics for one normalized video at a time."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

import torch
import torch.nn.functional as functional

from tardis.metrics.base import (
    Aggregation,
    FramePairFeature,
    MacroMicroAccumulator,
    VideoTextFeature,
    validate_video,
    validate_video_pair,
)


class _StreamingPairMetric:
    provenance_id: str

    def __init__(self) -> None:
        self._values = MacroMicroAccumulator()

    def compute(self, aggregation: Aggregation = "macro") -> float:
        return self._values.compute(aggregation)

    def reset(self) -> None:
        self._values.reset()

    def merge(self, other: Self) -> None:
        if type(self) is not type(other) or self.provenance_id != other.provenance_id:
            raise ValueError("only matching metric implementations can be merged")
        self._values.merge(other._values)

    def state_dict(self) -> dict[str, object]:
        return {"provenance_id": self.provenance_id, "values": self._values.state_dict()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"provenance_id", "values"}:
            raise ValueError("paired metric state has an incompatible schema")
        if state["provenance_id"] != self.provenance_id:
            raise ValueError("paired metric provenance does not match")
        values = state["values"]
        if not isinstance(values, Mapping):
            raise ValueError("paired metric values state must be a mapping")
        self._values.load_state_dict(cast(Mapping[str, float | int], values))

    def all_reduce(self) -> None:
        self._values.all_reduce()

    def _record(self, values: torch.Tensor) -> None:
        detached = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if detached.numel() == 0 or not bool(torch.isfinite(detached).all().item()):
            raise ValueError("metric values must be finite and non-empty")
        numerator = float(detached.sum().item())
        self._values.update(numerator, detached.numel())


class TemporalConsistencyMetric(_StreamingPairMetric):
    """Official TC: L1 difference between generated and reference frame deltas."""

    provenance_id = "tardis/official-temporal-consistency:v1"

    @torch.no_grad()
    def update(self, generated: torch.Tensor, reference: torch.Tensor) -> None:
        validate_video_pair(generated, reference, min_frames=2)
        generated_delta = generated[1:] - generated[:-1]
        reference_delta = reference[1:] - reference[:-1]
        self._record((generated_delta - reference_delta).abs())


class LPIPSMetric(_StreamingPairMetric):
    """Framewise LPIPS with per-video and per-frame streaming aggregation."""

    def __init__(self, feature: FramePairFeature | None = None) -> None:
        super().__init__()
        if feature is None:
            from tardis.metrics.features import AlexNetLPIPS

            feature = AlexNetLPIPS()
        if not feature.provenance_id.strip():
            raise ValueError("LPIPS feature provenance_id must be non-empty")
        self.feature = feature
        self.provenance_id = feature.provenance_id

    @torch.no_grad()
    def update(self, generated: torch.Tensor, reference: torch.Tensor) -> None:
        validate_video_pair(generated, reference)
        scores = self.feature(generated, reference)
        if not isinstance(scores, torch.Tensor):
            raise ValueError("LPIPS must return one finite non-negative value per frame")
        flattened = scores.detach().reshape(-1)
        if (
            flattened.numel() != generated.shape[0]
            or not bool(torch.isfinite(flattened).all().item())
            or bool((flattened < 0).any().item())
        ):
            raise ValueError("LPIPS must return one finite non-negative value per frame")
        self._record(flattened)


class SSIMMetric(_StreamingPairMetric):
    """Gaussian-window multichannel SSIM over normalized RGB frames."""

    def __init__(self, *, window_size: int = 11, sigma: float = 1.5) -> None:
        if window_size <= 0 or window_size % 2 == 0 or sigma <= 0:
            raise ValueError("SSIM window_size must be positive and odd and sigma must be positive")
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.provenance_id = f"tardis/ssim:gaussian-{window_size}-sigma-{sigma:g}-data-range-2"

    @torch.no_grad()
    def update(self, generated: torch.Tensor, reference: torch.Tensor) -> None:
        validate_video_pair(generated, reference)
        self._record(_multichannel_ssim(generated, reference, self.window_size, self.sigma))


class CLIPScoreMetric(_StreamingPairMetric):
    """Average raw cosine similarity between each generated frame and its prompt."""

    def __init__(self, feature: VideoTextFeature | None = None) -> None:
        super().__init__()
        if feature is None:
            from tardis.metrics.features import OpenCLIPFeatures

            feature = OpenCLIPFeatures()
        if feature.feature_dim <= 0 or not feature.provenance_id.strip():
            raise ValueError("CLIP feature dimension and provenance must be valid")
        self.feature = feature
        self.provenance_id = feature.provenance_id

    @torch.no_grad()
    def update(self, generated: torch.Tensor, prompt: str) -> None:
        validate_video(generated, name="generated")
        if not prompt.strip():
            raise ValueError("CLIPScore prompt must be non-empty")
        visual = self.feature.encode_video(generated)
        text = self.feature.encode_text(prompt)
        if text.ndim == 1:
            text = text.unsqueeze(0)
        if (
            visual.ndim != 2
            or visual.shape != (generated.shape[0], self.feature.feature_dim)
            or text.shape != (1, self.feature.feature_dim)
            or not bool(torch.isfinite(visual).all().item())
            or not bool(torch.isfinite(text).all().item())
        ):
            raise ValueError("CLIP encoder outputs must be finite frame [T,D] and text [1,D]")
        visual = visual.detach().to(device="cpu", dtype=torch.float64)
        text = text.detach().to(device="cpu", dtype=torch.float64)
        scores = functional.cosine_similarity(visual, text.expand_as(visual), dim=1, eps=1.0e-12)
        self._record(scores)


TCMetric = TemporalConsistencyMetric


def _multichannel_ssim(
    generated: torch.Tensor,
    reference: torch.Tensor,
    window_size: int,
    sigma: float,
) -> torch.Tensor:
    height, width = generated.shape[-2:]
    size = min(window_size, height, width)
    if size % 2 == 0:
        size -= 1
    working_generated = generated.detach().to(torch.float64)
    working_reference = reference.detach().to(torch.float64)
    coordinate = torch.arange(size, device=generated.device, dtype=torch.float64)
    coordinate -= (size - 1) / 2
    gaussian = torch.exp(-(coordinate.square()) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(3, 1, size, size)

    mean_generated = functional.conv2d(working_generated, window, groups=3)
    mean_reference = functional.conv2d(working_reference, window, groups=3)
    generated_variance = (
        functional.conv2d(working_generated.square(), window, groups=3) - mean_generated.square()
    ).clamp_min(0)
    reference_variance = (
        functional.conv2d(working_reference.square(), window, groups=3) - mean_reference.square()
    ).clamp_min(0)
    covariance = (
        functional.conv2d(working_generated * working_reference, window, groups=3)
        - mean_generated * mean_reference
    )

    c1 = (0.01 * 2.0) ** 2
    c2 = (0.03 * 2.0) ** 2
    numerator = (2 * mean_generated * mean_reference + c1) * (2 * covariance + c2)
    denominator = (mean_generated.square() + mean_reference.square() + c1) * (
        generated_variance + reference_variance + c2
    )
    scores = numerator / denominator.clamp_min(torch.finfo(torch.float64).eps)
    return scores.mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)
