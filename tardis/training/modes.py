"""Parameter ownership for full and focused TARDIS optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

TRAIN_MODES = ("full_temporal", "keyframe_only")


@dataclass(frozen=True, slots=True)
class TrainModeSummary:
    mode: str
    trainable_groups: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int


def configure_train_mode(model: nn.Module, mode: str) -> TrainModeSummary:
    """Freeze a stable parameter subset before optimizer and DDP construction."""

    from torch import nn

    if mode not in TRAIN_MODES:
        raise ValueError(f"train mode must be one of {TRAIN_MODES}; got {mode!r}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("priors."))

    if mode == "keyframe_only":
        for name, parameter in model.named_parameters():
            if not name.startswith("priors."):
                parameter.requires_grad_(False)
        selected: tuple[str, ...] = ("keyframe_residual_dit", "lite_corrector")
        for group in selected:
            module = getattr(model, group, None)
            if not isinstance(module, nn.Module):
                raise TypeError(f"keyframe-only training requires model.{group}")
            module.requires_grad_(True)
        trainable_groups = selected
    else:
        trainable_groups = tuple(
            sorted(
                {
                    name.split(".", 1)[0]
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
            )
        )

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise ValueError(f"train mode {mode!r} selected no parameters")
    return TrainModeSummary(
        mode=mode,
        trainable_groups=trainable_groups,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
    )


__all__ = ["TRAIN_MODES", "TrainModeSummary", "configure_train_mode"]
