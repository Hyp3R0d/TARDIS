from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from tardis.training.curriculum import CurriculumPoint, CurriculumSchedule
from tardis.training.engine import (
    ObjectiveOutput,
    TrainEngine,
    TrainEngineOptions,
    TrainingObjective,
)
from tardis.training.validation import METRIC_BASELINES, ValidationMetric
from tardis.utils.checkpoint import load_checkpoint
from tests.helpers.tardis_model import build_tiny_tardis


class ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def stochastic_objective(
    model: nn.Module,
    batch: object,
    _point: object,
    generator: torch.Generator,
) -> ObjectiveOutput:
    assert isinstance(batch, tuple)
    value, target = batch
    prediction = model(value)
    noise = torch.rand((), generator=generator, device=prediction.device) * 0.01
    loss = (prediction - target + noise).square().mean()
    return ObjectiveOutput(total=loss, losses={"diffusion": loss})


def build_engine(
    *,
    accumulation: int = 2,
    objective: TrainingObjective = stochastic_objective,
) -> TrainEngine:
    torch.manual_seed(5)
    model = ScalarModel()
    return TrainEngine(
        model,
        objective=objective,
        options=TrainEngineOptions(
            learning_rate=0.1,
            weight_decay=0.0,
            gradient_accumulation_steps=accumulation,
            gradient_clip_norm=1.0,
            warmup_steps=1,
            total_optimizer_steps=12,
            precision="fp32",
            ema_decay=0.9,
        ),
        curriculum=CurriculumSchedule(durations=(2, 2, 2, 2, 2, 2)),
        generator=torch.Generator().manual_seed(77),
    )


def batch(value: float, target: float) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([value]), torch.tensor([target])


def baseline_metrics(*, tc_scale: float = 1.0) -> dict[str, dict[str, float]]:
    result = {
        "dataverse_validation": {
            metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric
        }
    }
    for metrics in result.values():
        metrics["tc"] *= tc_scale
    return result


class StatefulObjective:
    def __init__(self) -> None:
        self.running_scale = 1.0

    def __call__(
        self,
        model: nn.Module,
        batch: object,
        _point: CurriculumPoint,
        _generator: torch.Generator,
    ) -> ObjectiveOutput:
        assert isinstance(batch, tuple)
        value, target = cast(tuple[torch.Tensor, torch.Tensor], batch)
        self.running_scale += 0.25
        loss = (model(value) - target).square().mean() / self.running_scale
        return ObjectiveOutput(total=loss, losses={"stateful": loss})

    def state_dict(self) -> dict[str, object]:
        return {"running_scale": self.running_scale}

    def load_state_dict(self, state: object) -> None:
        assert isinstance(state, dict)
        self.running_scale = float(state["running_scale"])


class TeacherTrackingObjective(StatefulObjective):
    def __init__(self) -> None:
        super().__init__()
        self.teacher_updates = 0

    def update_teacher(self, model: nn.Module) -> None:
        assert isinstance(model, ScalarModel)
        self.teacher_updates += 1


def test_train_engine_accumulates_then_updates_optimizer_scheduler_and_ema() -> None:
    engine = build_engine(accumulation=2)
    initial_weight = engine.unwrapped_model.weight.detach().clone()
    initial_ema = engine.ema.shadow["weight"].clone()

    first = engine.train_microbatch(batch(2.0, 1.0), batch_ids=("first",))
    second = engine.train_microbatch(batch(2.0, 1.0), batch_ids=("second",))

    assert not first.optimizer_updated
    assert second.optimizer_updated
    assert engine.optimizer_step == 1
    assert engine.micro_step == 2
    assert not torch.equal(engine.unwrapped_model.weight, initial_weight)
    assert not torch.equal(engine.ema.shadow["weight"], initial_ema)
    assert second.gradient_norm is not None
    assert second.learning_rate > 0


def test_objective_teacher_updates_only_after_an_optimizer_update() -> None:
    objective = TeacherTrackingObjective()
    engine = build_engine(accumulation=2, objective=objective)

    engine.train_microbatch(batch(2.0, 1.0))
    assert objective.teacher_updates == 0
    engine.train_microbatch(batch(2.0, 1.0))

    assert objective.teacher_updates == 1


def test_nonfinite_loss_is_logged_and_discards_pending_gradients() -> None:
    engine = build_engine(accumulation=2)

    engine.train_microbatch(batch(1.0, 0.0), batch_ids=("valid",))
    result = engine.train_microbatch(batch(float("nan"), 0.0), batch_ids=("broken",))

    assert result.skipped_nonfinite
    assert not result.optimizer_updated
    assert engine.accumulation_index == 0
    assert engine.nonfinite_ledger == [(2, ("broken",))]
    assert all(parameter.grad is None for parameter in engine.unwrapped_model.parameters())


def test_finite_rank_skips_before_backward_when_another_rank_reports_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tardis.training.engine as engine_module

    engine = build_engine(accumulation=1)
    initial_weight = engine.unwrapped_model.weight.detach().clone()
    reductions: list[torch.Tensor] = []

    def report_remote_nonfinite(value: torch.Tensor, *_args: object, **_kwargs: object) -> None:
        reductions.append(value.clone())
        value.zero_()

    monkeypatch.setattr(engine_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(engine_module.dist, "all_reduce", report_remote_nonfinite)

    result = engine.train_microbatch(batch(2.0, 1.0), batch_ids=("local-finite",))

    assert reductions and reductions[0].item() == 1
    assert result.skipped_nonfinite
    assert not result.optimizer_updated
    assert engine.nonfinite_ledger == [(1, ("local-finite",))]
    assert torch.equal(engine.unwrapped_model.weight, initial_weight)
    assert all(parameter.grad is None for parameter in engine.unwrapped_model.parameters())


def test_engine_state_round_trip_reproduces_next_losses_and_weights() -> None:
    uninterrupted = build_engine(accumulation=2)
    uninterrupted.train_microbatch(batch(2.0, 1.0))
    uninterrupted.train_microbatch(batch(1.5, 0.3))
    state = uninterrupted.state_dict(epoch=3, next_batch_index=8)

    resumed = build_engine(accumulation=2)
    position = resumed.load_state_dict(state)
    assert position == (3, 8)

    expected_results = [
        uninterrupted.train_microbatch(batch(0.5, 0.7)),
        uninterrupted.train_microbatch(batch(1.2, -0.1)),
    ]
    resumed_results = [
        resumed.train_microbatch(batch(0.5, 0.7)),
        resumed.train_microbatch(batch(1.2, -0.1)),
    ]

    assert [item.total_loss for item in resumed_results] == pytest.approx(
        [item.total_loss for item in expected_results]
    )
    assert torch.equal(resumed.unwrapped_model.weight, uninterrupted.unwrapped_model.weight)
    assert resumed.optimizer_step == uninterrupted.optimizer_step
    assert torch.equal(resumed.ema.shadow["weight"], uninterrupted.ema.shadow["weight"])


def test_stochastic_state_round_trip_restores_rank_local_generator_and_rngs() -> None:
    class RankStateObjective(StatefulObjective):
        def rank_state_dict(self) -> dict[str, object]:
            return {"running_scale": self.running_scale}

        def load_rank_state_dict(self, state: object) -> None:
            assert isinstance(state, Mapping)
            self.running_scale = float(state["running_scale"])

    objective = RankStateObjective()
    engine = build_engine(accumulation=1, objective=objective)
    engine.train_microbatch(batch(2.0, 1.0))
    state = engine.stochastic_state_dict()
    expected_generator = engine.generator.get_state().clone()
    expected_torch = torch.get_rng_state().clone()
    expected_scale = objective.running_scale

    torch.manual_seed(999)
    engine.generator.manual_seed(1001)
    objective.running_scale = 101.0
    engine.load_stochastic_state_dict(state)

    assert torch.equal(engine.generator.get_state(), expected_generator)
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert objective.running_scale == expected_scale


def test_engine_unwraps_compile_style_wrapper_for_state_and_ema() -> None:
    class CompileStyleWrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self._orig_mod = module

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self._orig_mod(value)

    model = ScalarModel()
    wrapped = CompileStyleWrapper(model)
    engine = TrainEngine(
        wrapped,
        objective=stochastic_objective,
        options=TrainEngineOptions(
            learning_rate=0.1,
            weight_decay=0,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1,
            warmup_steps=0,
            total_optimizer_steps=6,
            precision="fp32",
        ),
        curriculum=CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)),
        generator=torch.Generator().manual_seed(7),
    )

    engine.train_microbatch(batch(2.0, 1.0))

    assert engine.execution_model is wrapped
    assert engine.unwrapped_model is model
    assert set(engine.ema.shadow) == {"weight"}


def test_mid_accumulation_resume_restores_objective_state_and_pending_gradient() -> None:
    uninterrupted_objective = StatefulObjective()
    uninterrupted = build_engine(accumulation=2, objective=uninterrupted_objective)
    uninterrupted.train_microbatch(batch(2.0, 1.0))
    state = uninterrupted.state_dict(epoch=4, next_batch_index=9)

    resumed_objective = StatefulObjective()
    resumed = build_engine(accumulation=2, objective=resumed_objective)
    assert resumed.load_state_dict(state) == (4, 9)

    expected = uninterrupted.train_microbatch(batch(1.5, 0.4))
    actual = resumed.train_microbatch(batch(1.5, 0.4))

    assert actual.total_loss == pytest.approx(expected.total_loss)
    assert resumed_objective.running_scale == uninterrupted_objective.running_scale
    assert torch.equal(resumed.unwrapped_model.weight, uninterrupted.unwrapped_model.weight)
    assert torch.equal(resumed.ema.shadow["weight"], uninterrupted.ema.shadow["weight"])


def test_tardis_training_checkpoint_excludes_and_never_loads_frozen_priors() -> None:
    source_assembly = build_tiny_tardis()
    options = TrainEngineOptions(
        learning_rate=0.01,
        weight_decay=0,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1,
        warmup_steps=0,
        total_optimizer_steps=6,
        precision="fp32",
    )
    curriculum = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1))
    source = TrainEngine(
        source_assembly.model,
        objective=stochastic_objective,
        options=options,
        curriculum=curriculum,
        generator=torch.Generator().manual_seed(5),
    )
    state = source.state_dict(epoch=0, next_batch_index=0)

    model_state = cast(dict[str, torch.Tensor], state["model"])
    assert model_state
    assert all(not name.startswith("priors.") for name in model_state)

    target_assembly = build_tiny_tardis()
    with torch.no_grad():
        for parameter in target_assembly.priors.parameters():
            parameter.add_(7)
    prior_before = {
        name: value.detach().clone()
        for name, value in target_assembly.model.state_dict().items()
        if name.startswith("priors.")
    }
    target = TrainEngine(
        target_assembly.model,
        objective=stochastic_objective,
        options=options,
        curriculum=curriculum,
        generator=torch.Generator().manual_seed(5),
    )

    target.load_state_dict(state)

    for name, value in target_assembly.model.state_dict().items():
        if name.startswith("priors."):
            assert torch.equal(value, prior_before[name])


def test_selective_tardis_ema_keeps_complete_temporal_state() -> None:
    from tardis.training.modes import configure_train_mode

    assembly = build_tiny_tardis(
        keyframe_lite_alignment=True,
        keyframe_residual_generation=True,
    )
    configure_train_mode(assembly.model, "keyframe_only")
    engine = TrainEngine(
        assembly.model,
        objective=stochastic_objective,
        options=TrainEngineOptions(
            learning_rate=0.01,
            weight_decay=0,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1,
            warmup_steps=0,
            total_optimizer_steps=6,
            precision="fp32",
            ema_decay=0.9,
        ),
        curriculum=CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)),
        generator=torch.Generator().manual_seed(5),
    )

    expected = {
        name
        for name, parameter in assembly.model.named_parameters()
        if not name.startswith("priors.") and parameter.is_floating_point()
    }

    assert set(engine.ema.shadow) == expected
    assert any(name.startswith("motion_scaffold.") for name in engine.ema.shadow)
    assert any(name.startswith("keyframe_residual_dit.") for name in engine.ema.shadow)


def test_epoch_checkpoint_always_updates_latest_and_only_improvement_updates_best(
    tmp_path: Path,
) -> None:
    engine = build_engine(accumulation=1)
    checkpoint_dir = tmp_path / "run"

    assert engine.save_epoch(
        checkpoint_dir,
        epoch=1,
        next_batch_index=0,
        validation_metrics=baseline_metrics(),
    )
    first_best = load_checkpoint(checkpoint_dir / "best.pt")
    assert not engine.save_epoch(
        checkpoint_dir,
        epoch=2,
        next_batch_index=0,
        validation_metrics=baseline_metrics(),
    )
    assert load_checkpoint(checkpoint_dir / "best.pt")["epoch"] == first_best["epoch"]
    assert load_checkpoint(checkpoint_dir / "latest.pt")["epoch"] == 2

    assert engine.save_epoch(
        checkpoint_dir,
        epoch=3,
        next_batch_index=0,
        validation_metrics=baseline_metrics(tc_scale=0.5),
    )
    best = load_checkpoint(checkpoint_dir / "best.pt")
    assert best["epoch"] == 3
    assert best["validation_score"]["average_metrics"]["tc"] == pytest.approx(
        METRIC_BASELINES[ValidationMetric.TC] * 0.5
    )
