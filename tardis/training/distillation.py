"""Causal residual consistency distillation over shared transport conditions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import cast

import torch

from tardis.models.quotient import TransportOrbitBasis, TransportOrbitProjector
from tardis.models.residual import ResidualDenoisingContext, SparseResidualDiT
from tardis.training.losses import CRCDTarget, build_crcd_target


class CausalResidualDistiller:
    """Maintain an EMA target that denoises only transport innovation residuals."""

    def __init__(self, *, ema_decay: float = 0.999, transition_ratio: float = 0.5) -> None:
        if not 0 <= ema_decay < 1:
            raise ValueError("CRCD EMA decay must be in [0, 1)")
        if not 0 < transition_ratio < 1:
            raise ValueError("CRCD transition_ratio must be in (0, 1)")
        self.ema_decay = ema_decay
        self.transition_ratio = transition_ratio
        self.teacher: SparseResidualDiT | None = None
        self._pending_teacher_state: dict[str, torch.Tensor] | None = None

    def build_target(
        self,
        student: SparseResidualDiT,
        context: ResidualDenoisingContext,
        *,
        quotient_projector: TransportOrbitProjector | None = None,
        quotient_basis: TransportOrbitBasis | None = None,
    ) -> CRCDTarget:
        if (quotient_projector is None) != (quotient_basis is None):
            raise ValueError("quotient projector and basis must be provided together")
        teacher = self._ensure_teacher(student)
        lower_time = context.diffusion_time * self.transition_ratio
        interpolation = lower_time[:, None, None, None]
        with torch.no_grad():
            first_prediction = context.predict(teacher).residual
            if quotient_projector is not None and quotient_basis is not None:
                first_prediction = quotient_projector.decompose(
                    first_prediction,
                    quotient_basis,
                ).innovation
            intermediate = (1 - interpolation) * first_prediction + (
                interpolation * context.diffusion_noise
            )
            target = context.predict(
                teacher,
                noisy_residual=intermediate,
                diffusion_time=lower_time,
            ).residual
            if quotient_projector is not None and quotient_basis is not None:
                target = quotient_projector.decompose(target, quotient_basis).innovation
        return build_crcd_target(target, context.transported_prior)

    @torch.no_grad()
    def update(self, student: SparseResidualDiT) -> None:
        if self.teacher is None and self._pending_teacher_state is None:
            return
        teacher = self._ensure_teacher(student)
        teacher_parameters = dict(teacher.named_parameters())
        student_parameters = dict(student.named_parameters())
        if set(teacher_parameters) != set(student_parameters):
            raise ValueError("CRCD teacher/student parameter names do not match")
        for name, target in teacher_parameters.items():
            source = student_parameters[name].detach().to(device=target.device, dtype=target.dtype)
            target.lerp_(source, 1 - self.ema_decay)
        teacher_buffers = dict(teacher.named_buffers())
        student_buffers = dict(student.named_buffers())
        if set(teacher_buffers) != set(student_buffers):
            raise ValueError("CRCD teacher/student buffer names do not match")
        for name, target_buffer in teacher_buffers.items():
            target_buffer.copy_(
                student_buffers[name]
                .detach()
                .to(device=target_buffer.device, dtype=target_buffer.dtype)
            )

    def state_dict(self) -> dict[str, object]:
        if self.teacher is not None:
            teacher_state: dict[str, torch.Tensor] | None = {
                name: value.detach().clone() for name, value in self.teacher.state_dict().items()
            }
        elif self._pending_teacher_state is not None:
            teacher_state = {
                name: value.detach().clone() for name, value in self._pending_teacher_state.items()
            }
        else:
            teacher_state = None
        return {
            "version": 1,
            "ema_decay": self.ema_decay,
            "transition_ratio": self.transition_ratio,
            "teacher": teacher_state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"version", "ema_decay", "transition_ratio", "teacher"}
        if set(state) != required or state["version"] != 1:
            raise ValueError("CRCD state has an incompatible schema")
        if float(cast(float, state["ema_decay"])) != self.ema_decay:
            raise ValueError("CRCD EMA decay does not match")
        if float(cast(float, state["transition_ratio"])) != self.transition_ratio:
            raise ValueError("CRCD transition ratio does not match")
        raw_teacher = state["teacher"]
        if raw_teacher is None:
            self.teacher = None
            self._pending_teacher_state = None
            return
        if not isinstance(raw_teacher, Mapping):
            raise ValueError("CRCD teacher state must be a tensor mapping or None")
        pending: dict[str, torch.Tensor] = {}
        for name, value in raw_teacher.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise ValueError("CRCD teacher state entries must be named tensors")
            pending[name] = value.detach().clone()
        self.teacher = None
        self._pending_teacher_state = pending

    def _ensure_teacher(self, student: SparseResidualDiT) -> SparseResidualDiT:
        if self.teacher is None:
            teacher = deepcopy(student)
            teacher.requires_grad_(False)
            teacher.eval()
            if self._pending_teacher_state is not None:
                teacher.load_state_dict(self._pending_teacher_state, strict=True)
                self._pending_teacher_state = None
            self.teacher = teacher
        return self.teacher
