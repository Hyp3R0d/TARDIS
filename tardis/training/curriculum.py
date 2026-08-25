"""Optimizer-step curriculum for staged TARDIS mechanism training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class CurriculumStage(StrEnum):
    TRANSPORT_WARMUP = "transport_warmup"
    ROUTER_CALIBRATION = "router_calibration"
    RESIDUAL_TEACHER = "residual_teacher"
    CLOSED_LOOP = "closed_loop"
    CRCD = "crcd"
    METRIC_ALIGNMENT = "metric_alignment"


_STAGES: Final[tuple[CurriculumStage, ...]] = tuple(CurriculumStage)


@dataclass(frozen=True, slots=True)
class CurriculumPoint:
    stage: CurriculumStage
    global_step: int
    stage_step: int
    stage_progress: float
    teacher_forcing_ratio: float
    residual_steps: int


class CurriculumSchedule:
    """Map optimizer steps to six cumulative training stages."""

    def __init__(self, *, durations: tuple[int, int, int, int, int, int]) -> None:
        if len(durations) != len(_STAGES) or any(duration <= 0 for duration in durations):
            raise ValueError("durations must contain six positive stage lengths")
        self.durations = durations
        cumulative: list[int] = []
        total = 0
        for duration in durations:
            total += duration
            cumulative.append(total)
        self.cumulative_ends = tuple(cumulative)

    def at_step(self, optimizer_step: int) -> CurriculumPoint:
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        stage_index = len(_STAGES) - 1
        stage_start = self.cumulative_ends[-2]
        for index, end in enumerate(self.cumulative_ends):
            if optimizer_step < end:
                stage_index = index
                stage_start = 0 if index == 0 else self.cumulative_ends[index - 1]
                break
        stage = _STAGES[stage_index]
        stage_step = max(optimizer_step - stage_start, 0)
        duration = self.durations[stage_index]
        progress = min(stage_step / max(duration - 1, 1), 1.0)
        return CurriculumPoint(
            stage=stage,
            global_step=optimizer_step,
            stage_step=stage_step,
            stage_progress=progress,
            teacher_forcing_ratio=_teacher_forcing_ratio(stage, progress),
            residual_steps=_residual_steps(stage),
        )


def active_loss_names(stage: CurriculumStage) -> set[str]:
    base = {"diffusion", "transport", "flow", "visibility", "lite"}
    if stage is CurriculumStage.TRANSPORT_WARMUP:
        return base
    if stage is CurriculumStage.ROUTER_CALIBRATION:
        return base | {"router", "survival", "budget"}
    if stage is CurriculumStage.RESIDUAL_TEACHER:
        return base | {"router", "survival", "budget", "residual"}
    if stage is CurriculumStage.CLOSED_LOOP:
        return base | {"router", "survival", "budget", "residual", "warp", "drift"}
    if stage is CurriculumStage.CRCD:
        return base | {
            "router",
            "survival",
            "budget",
            "residual",
            "warp",
            "drift",
            "crcd",
        }
    if stage is CurriculumStage.METRIC_ALIGNMENT:
        return base | {
            "router",
            "survival",
            "budget",
            "residual",
            "warp",
            "drift",
            "crcd",
            "lpips",
            "tc",
            "text",
        }
    raise ValueError(f"unsupported curriculum stage: {stage}")


def active_parameter_groups(stage: CurriculumStage) -> set[str]:
    groups = {"motion", "transport", "state", "lite"}
    if stage in {
        CurriculumStage.ROUTER_CALIBRATION,
        CurriculumStage.RESIDUAL_TEACHER,
        CurriculumStage.CLOSED_LOOP,
        CurriculumStage.CRCD,
        CurriculumStage.METRIC_ALIGNMENT,
    }:
        groups.add("router")
    if stage in {
        CurriculumStage.RESIDUAL_TEACHER,
        CurriculumStage.CLOSED_LOOP,
        CurriculumStage.CRCD,
        CurriculumStage.METRIC_ALIGNMENT,
    }:
        groups.add("residual_teacher")
    if stage in {CurriculumStage.CRCD, CurriculumStage.METRIC_ALIGNMENT}:
        groups.add("residual_student")
    if stage is CurriculumStage.METRIC_ALIGNMENT:
        groups.add("metric_adapter")
    return groups


def _teacher_forcing_ratio(stage: CurriculumStage, progress: float) -> float:
    if stage in {
        CurriculumStage.TRANSPORT_WARMUP,
        CurriculumStage.ROUTER_CALIBRATION,
        CurriculumStage.RESIDUAL_TEACHER,
    }:
        return 1.0
    if stage is CurriculumStage.CLOSED_LOOP:
        return 1.0 - 0.75 * progress
    if stage is CurriculumStage.CRCD:
        return 0.25 * (1 - progress)
    return 0.0


def _residual_steps(stage: CurriculumStage) -> int:
    if stage in {CurriculumStage.TRANSPORT_WARMUP, CurriculumStage.ROUTER_CALIBRATION}:
        return 0
    if stage in {CurriculumStage.RESIDUAL_TEACHER, CurriculumStage.CLOSED_LOOP}:
        return 4
    return 1
