"""Motion-advected innovation proper time for budgeted renewal diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from tardis.models.router import InnovationSelection, select_innovation_budget


@dataclass(frozen=True, slots=True)
class InnovationClockOutput:
    """One predict-service-reset cycle of the quotient innovation clock."""

    instantaneous_hazard: torch.Tensor
    accrued_hazard: torch.Tensor
    event_probability: torch.Tensor
    patch_probability: torch.Tensor
    selection: InnovationSelection
    service_mask: torch.Tensor
    settled_hazard: torch.Tensor


class InnovationProperTime(nn.Module):
    """Accumulate innovation hazard along motion trajectories until service.

    The router predicts a conditional event probability for one transition. Its
    negative log-survival is additive over time, so the transported cumulative
    hazard is an exact sufficient statistic under the conditional survival model.
    A sparse diffusion update is a renewal event: selected patches are serviced
    and their hazard clock is reset, while unserved patches keep accumulating.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        active_ratio: float,
        threshold: float,
        halo_radius: int,
        maximum_hazard: float = 20.0,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("proper-time patch_size must be positive")
        if not 0 < active_ratio <= 1:
            raise ValueError("proper-time active_ratio must be in (0, 1]")
        if not 0 <= threshold <= 1:
            raise ValueError("proper-time threshold must be in [0, 1]")
        if halo_radius < 0 or maximum_hazard <= 0:
            raise ValueError("proper-time halo and maximum hazard are invalid")
        self.patch_size = patch_size
        self.active_ratio = active_ratio
        self.threshold = threshold
        self.halo_radius = halo_radius
        self.maximum_hazard = maximum_hazard

    def forward(
        self,
        transported_hazard: torch.Tensor,
        instantaneous_risk: torch.Tensor,
        visibility: torch.Tensor,
        *,
        active_ratio: float | None = None,
    ) -> InnovationClockOutput:
        if transported_hazard.ndim != 4 or transported_hazard.shape[1] != 1:
            raise ValueError("transported_hazard must have shape [B,1,H,W]")
        if instantaneous_risk.shape != transported_hazard.shape:
            raise ValueError("instantaneous_risk must match transported_hazard")
        if visibility.shape != transported_hazard.shape:
            raise ValueError("visibility must match transported_hazard")
        height, width = transported_hazard.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("proper-time spatial dimensions must be divisible by patch_size")
        budget_ratio = self.active_ratio if active_ratio is None else active_ratio
        if not 0 < budget_ratio <= 1:
            raise ValueError("proper-time active_ratio override must be in (0, 1]")

        original_dtype = transported_hazard.dtype
        compute_dtype = (
            torch.float32 if original_dtype in {torch.float16, torch.bfloat16} else original_dtype
        )
        previous = transported_hazard.to(compute_dtype).clamp(0, self.maximum_hazard)
        risk = instantaneous_risk.to(compute_dtype).clamp(0, 1)
        confidence = visibility.to(compute_dtype).clamp(0, 1)
        epsilon = torch.finfo(compute_dtype).eps
        instantaneous_hazard = -torch.log1p(-risk.clamp_max(1 - epsilon))
        accrued_hazard = (previous * confidence + instantaneous_hazard).clamp_max(
            self.maximum_hazard
        )
        event_probability = -torch.expm1(-accrued_hazard)

        patch_hazard = functional.avg_pool2d(
            accrued_hazard,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch_probability = -torch.expm1(-patch_hazard)
        selection = select_innovation_budget(
            patch_probability.detach(),
            active_ratio=budget_ratio,
            threshold=self.threshold,
            halo_radius=self.halo_radius,
        )
        service_mask = functional.interpolate(
            selection.active_mask.to(compute_dtype),
            size=(height, width),
            mode="nearest",
        ).to(torch.bool)
        settled_hazard = accrued_hazard * (~service_mask).to(compute_dtype)

        return InnovationClockOutput(
            instantaneous_hazard=instantaneous_hazard.to(original_dtype),
            accrued_hazard=accrued_hazard.to(original_dtype),
            event_probability=event_probability.to(original_dtype),
            patch_probability=patch_probability.to(original_dtype),
            selection=selection,
            service_mask=service_mask,
            settled_hazard=settled_hazard.to(original_dtype),
        )


__all__ = ["InnovationClockOutput", "InnovationProperTime"]
