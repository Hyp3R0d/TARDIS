from __future__ import annotations

import pytest

from tardis.training.curriculum import (
    CurriculumSchedule,
    CurriculumStage,
    active_loss_names,
)


def test_curriculum_is_keyed_to_optimizer_steps_and_clamps_after_final_stage() -> None:
    schedule = CurriculumSchedule(durations=(2, 2, 2, 2, 2, 2))

    assert schedule.at_step(0).stage is CurriculumStage.TRANSPORT_WARMUP
    assert schedule.at_step(1).stage is CurriculumStage.TRANSPORT_WARMUP
    assert schedule.at_step(2).stage is CurriculumStage.ROUTER_CALIBRATION
    assert schedule.at_step(10).stage is CurriculumStage.METRIC_ALIGNMENT
    assert schedule.at_step(10_000).stage is CurriculumStage.METRIC_ALIGNMENT


def test_stage_loss_activation_matches_tardis_mechanism_order() -> None:
    assert active_loss_names(CurriculumStage.TRANSPORT_WARMUP) == {
        "diffusion",
        "transport",
        "flow",
        "visibility",
        "lite",
    }
    assert "router" in active_loss_names(CurriculumStage.ROUTER_CALIBRATION)
    assert "survival" in active_loss_names(CurriculumStage.ROUTER_CALIBRATION)
    assert "crcd" in active_loss_names(CurriculumStage.CRCD)
    assert "tc" in active_loss_names(CurriculumStage.METRIC_ALIGNMENT)


def test_schedule_rejects_invalid_durations_and_negative_steps() -> None:
    with pytest.raises(ValueError):
        CurriculumSchedule(durations=(1, 2))
    schedule = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        schedule.at_step(-1)
