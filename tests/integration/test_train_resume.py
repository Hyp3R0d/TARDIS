from __future__ import annotations

import pytest
import torch

from tardis.models.factory import tardis_temporal_state_dict
from tardis.models.tardis import TARDISTrainingBatch
from tardis.training.curriculum import CurriculumSchedule
from tardis.training.engine import TrainEngine, TrainEngineOptions
from tardis.training.objective import TARDISObjective
from tests.helpers.tardis_model import build_tiny_tardis


class TinyPerceptualMetric:
    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3), keepdim=True)


def _build_engine() -> tuple[TrainEngine, TARDISObjective]:
    model = build_tiny_tardis().model
    objective = TARDISObjective(
        perceptual_metric=TinyPerceptualMetric(),
        temporal_levels=1,
    )
    engine = TrainEngine(
        model,
        objective=objective,
        options=TrainEngineOptions(
            learning_rate=1.0e-3,
            weight_decay=0,
            gradient_accumulation_steps=2,
            gradient_clip_norm=1,
            warmup_steps=0,
            total_optimizer_steps=6,
            precision="fp32",
            ema_decay=0.9,
        ),
        curriculum=CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)),
        generator=torch.Generator().manual_seed(29),
    )
    return engine, objective


def _batch() -> TARDISTrainingBatch:
    return TARDISTrainingBatch(
        prompts=["a luminous cube moving slowly"],
        video=torch.randn(1, 3, 3, 16, 16, generator=torch.Generator().manual_seed(31)),
    )


@pytest.mark.integration
def test_tardis_resume_mid_accumulation_matches_uninterrupted_next_update() -> None:
    uninterrupted, uninterrupted_objective = _build_engine()
    training_batch = _batch()
    first = uninterrupted.train_microbatch(training_batch, batch_ids=("first",))
    assert not first.optimizer_updated
    checkpoint = uninterrupted.state_dict(epoch=7, next_batch_index=11)

    expected = uninterrupted.train_microbatch(training_batch, batch_ids=("second",))
    expected_temporal = tardis_temporal_state_dict(uninterrupted.unwrapped_model)
    expected_ema = uninterrupted.ema.state_dict()
    expected_objective = uninterrupted_objective.state_dict()

    resumed, resumed_objective = _build_engine()
    assert resumed.load_state_dict(checkpoint) == (7, 11)
    actual = resumed.train_microbatch(training_batch, batch_ids=("second",))

    assert actual.optimizer_updated and expected.optimizer_updated
    assert actual.total_loss == pytest.approx(expected.total_loss)
    assert resumed.micro_step == uninterrupted.micro_step
    assert resumed.optimizer_step == uninterrupted.optimizer_step
    for name, value in tardis_temporal_state_dict(resumed.unwrapped_model).items():
        assert torch.equal(value, expected_temporal[name])
    resumed_shadow = resumed.ema.state_dict()["shadow"]
    assert isinstance(resumed_shadow, dict)
    expected_shadow = expected_ema["shadow"]
    assert isinstance(expected_shadow, dict)
    for name, value in resumed_shadow.items():
        assert isinstance(value, torch.Tensor)
        assert torch.equal(value, expected_shadow[name])
    assert resumed_objective.state_dict() == expected_objective
