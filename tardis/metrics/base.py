"""Shared contracts and additive state for streaming video metrics."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Literal, Protocol, Self

import torch
import torch.distributed as dist

Aggregation = Literal["macro", "micro"]
ScalarState = dict[str, float | int]


class FramePairFeature(Protocol):
    """A frozen frame-pair model that returns one value per frame."""

    provenance_id: str

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor: ...


class VideoTextFeature(Protocol):
    """A frozen frame/text encoder used for video-prompt cosine scores."""

    provenance_id: str
    feature_dim: int

    def encode_video(self, video: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, prompt: str) -> torch.Tensor: ...


class MacroMicroAccumulator:
    """Track a per-sample mean and its underlying weighted mean."""

    def __init__(self) -> None:
        self.macro_sum = 0.0
        self.macro_count = 0
        self.micro_sum = 0.0
        self.micro_count = 0

    def update(self, numerator: float, denominator: int) -> None:
        if not isfinite(numerator):
            raise ValueError("metric numerator must be finite")
        if denominator <= 0:
            raise ValueError("metric denominator must be positive")
        self.macro_sum += numerator / denominator
        self.macro_count += 1
        self.micro_sum += numerator
        self.micro_count += denominator

    def compute(self, aggregation: Aggregation = "macro") -> float:
        if aggregation == "macro":
            if self.macro_count == 0:
                raise RuntimeError("metric has no samples")
            return self.macro_sum / self.macro_count
        if aggregation == "micro":
            if self.micro_count == 0:
                raise RuntimeError("metric has no observations")
            return self.micro_sum / self.micro_count
        raise ValueError(f"unknown aggregation {aggregation!r}")

    def merge(self, other: Self) -> None:
        self.macro_sum += other.macro_sum
        self.macro_count += other.macro_count
        self.micro_sum += other.micro_sum
        self.micro_count += other.micro_count

    def reset(self) -> None:
        self.macro_sum = 0.0
        self.macro_count = 0
        self.micro_sum = 0.0
        self.micro_count = 0

    def state_dict(self) -> ScalarState:
        return {
            "macro_sum": self.macro_sum,
            "macro_count": self.macro_count,
            "micro_sum": self.micro_sum,
            "micro_count": self.micro_count,
        }

    def load_state_dict(self, state: Mapping[str, float | int]) -> None:
        required = {"macro_sum", "macro_count", "micro_sum", "micro_count"}
        if set(state) != required:
            raise ValueError("metric accumulator state has an incompatible schema")
        macro_sum = float(state["macro_sum"])
        macro_count = int(state["macro_count"])
        micro_sum = float(state["micro_sum"])
        micro_count = int(state["micro_count"])
        if not isfinite(macro_sum) or not isfinite(micro_sum) or macro_count < 0 or micro_count < 0:
            raise ValueError("metric accumulator state contains invalid values")
        self.macro_sum = macro_sum
        self.macro_count = macro_count
        self.micro_sum = micro_sum
        self.micro_count = micro_count

    def all_reduce(self) -> None:
        reduced = _distributed_sum(
            torch.tensor(
                [self.macro_sum, self.macro_count, self.micro_sum, self.micro_count],
                dtype=torch.float64,
            )
        )
        self.macro_sum = float(reduced[0].item())
        self.macro_count = int(reduced[1].item())
        self.micro_sum = float(reduced[2].item())
        self.micro_count = int(reduced[3].item())


def validate_video(video: torch.Tensor, *, name: str, min_frames: int = 1) -> None:
    """Validate the repository's normalized unbatched video contract."""

    if video.ndim != 4:
        raise ValueError(f"{name} video must have shape [T,3,H,W]")
    if video.shape[1] != 3:
        raise ValueError(f"{name} video must have three channels")
    if video.shape[0] < min_frames or min(video.shape[-2:]) <= 0:
        raise ValueError(f"{name} video has invalid frame or spatial dimensions")
    if not video.is_floating_point():
        raise ValueError(f"{name} video must use floating point values")
    if not bool(torch.isfinite(video).all().item()):
        raise ValueError(f"{name} video must contain finite values")
    minimum = float(video.detach().min().item())
    maximum = float(video.detach().max().item())
    if minimum < -1.0 or maximum > 1.0:
        raise ValueError(f"{name} video values must be in [-1, 1]")


def validate_video_pair(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    min_frames: int = 1,
) -> None:
    validate_video(generated, name="generated", min_frames=min_frames)
    validate_video(reference, name="reference", min_frames=min_frames)
    if generated.shape != reference.shape:
        raise ValueError("generated and reference videos must share shape")


def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    original_device = value.device
    reduction = value
    if dist.get_backend() == "nccl" and value.device.type != "cuda":
        reduction = value.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    return reduction.to(original_device)
