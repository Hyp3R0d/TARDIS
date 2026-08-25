from __future__ import annotations

import torch

from tardis.models.tardis import TARDISTrainingBatch
from tardis.training.curriculum import CurriculumSchedule
from tardis.training.losses import residual_reconstruction_loss
from tardis.training.modes import configure_train_mode
from tardis.training.objective import (
    TARDISKeyframeObjective,
    TARDISObjective,
    _decode_video_for_metric,
)
from tests.helpers.tardis_model import build_tiny_tardis


class TinyPerceptualMetric:
    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3), keepdim=True)


def test_metric_decode_matches_bounded_deployment_video_range() -> None:
    class Decoder:
        def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    decoded = _decode_video_for_metric(
        Decoder(),
        torch.tensor([[[[[-2.0, -0.25], [0.75, 2.0]]]]]),
    )

    assert decoded.min().item() == -1.0
    assert decoded.max().item() == 1.0


def training_batch() -> TARDISTrainingBatch:
    return TARDISTrainingBatch(
        prompts=["a luminous object moving slowly"],
        video=torch.randn(1, 4, 3, 16, 16),
    )


def test_tardis_objective_activates_only_the_current_curriculum_losses() -> None:
    model = build_tiny_tardis().model.train()
    objective = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    schedule = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1))

    warmup = objective(
        model,
        training_batch(),
        schedule.at_step(0),
        torch.Generator().manual_seed(9),
    )
    aligned = objective(
        model,
        training_batch(),
        schedule.at_step(5),
        torch.Generator().manual_seed(9),
    )

    assert set(warmup.losses) == {"diffusion", "transport", "flow", "visibility", "lite"}
    assert {"lpips", "tc", "text", "crcd", "survival"}.issubset(aligned.losses)
    assert warmup.total.ndim == 0
    assert aligned.total.ndim == 0
    assert torch.isfinite(warmup.total)
    assert torch.isfinite(aligned.total)


def test_transport_warmup_skips_inactive_long_horizon_losses() -> None:
    model = build_tiny_tardis().model.train()
    objective = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    point = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)).at_step(0)
    two_frame_batch = TARDISTrainingBatch(
        prompts=["a short warmup clip"],
        video=torch.randn(1, 2, 3, 16, 16),
    )

    output = objective(
        model,
        two_frame_batch,
        point,
        torch.Generator().manual_seed(17),
    )
    output.total.backward()

    assert set(output.losses) == {"diffusion", "transport", "flow", "visibility", "lite"}
    assert torch.isfinite(output.total)


def test_keyframe_alignment_supervises_keyframe_and_transition_residuals_independently() -> None:
    model = build_tiny_tardis(
        keyframe_lite_alignment=True,
        keyframe_residual_generation=True,
    ).model.train()
    objective = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    batch = training_batch()
    rollout = model.forward_train(
        batch,
        stage="transport_warmup",
        teacher_forcing_ratio=1.0,
        generator=torch.Generator().manual_seed(19),
    )

    losses = objective._candidate_losses(model, batch, rollout, active={"keyframe", "lite"})

    assert rollout.keyframe_prior is not None
    assert rollout.keyframe_lite_residual is not None
    keyframe_lite = residual_reconstruction_loss(
        rollout.keyframe_lite_residual,
        rollout.target_latents[:, 0] - rollout.keyframe_prior,
    )
    transition_lite = [
        residual_reconstruction_loss(item.lite_residual, item.tangent_target)
        for item in rollout.transitions
        if item.tangent_target is not None
    ]
    assert torch.allclose(losses["lite"], torch.stack([keyframe_lite, *transition_lite]).mean())
    assert rollout.keyframe_residual is not None
    assert "keyframe" in losses
    assert torch.isfinite(losses["keyframe"])
    assert all(item.lite_residual.requires_grad for item in rollout.transitions)


def test_keyframe_objective_only_updates_keyframe_modules() -> None:
    model = build_tiny_tardis(
        keyframe_lite_alignment=True,
        keyframe_residual_generation=True,
        diffusion_steps=2,
        diffusion_time_sampling="endpoint",
        sampler_trajectory_alignment=True,
    ).model.train()
    summary = configure_train_mode(model, "keyframe_only")
    objective = TARDISKeyframeObjective(perceptual_metric=TinyPerceptualMetric())
    point = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)).at_step(0)
    batch = TARDISTrainingBatch(
        prompts=["a luminous object moving slowly"],
        video=torch.randn(1, 1, 3, 16, 16),
    )

    output = objective(
        model,
        batch,
        point,
        torch.Generator().manual_seed(9),
    )
    output.total.backward()

    assert set(output.losses) == {"keyframe", "lpips"}
    assert summary.trainable_groups == ("keyframe_residual_dit", "lite_corrector")
    assert any(parameter.grad is not None for parameter in model.lite_corrector.parameters())
    assert any(parameter.grad is not None for parameter in model.keyframe_residual_dit.parameters())
    assert all(parameter.grad is None for parameter in model.motion_scaffold.parameters())
    assert all(parameter.grad is None for parameter in model.residual_dit.parameters())


def test_tardis_objective_backpropagates_into_motion_router_and_residual_network() -> None:
    model = build_tiny_tardis().model.train()
    objective = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    point = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)).at_step(5)

    output = objective(
        model,
        training_batch(),
        point,
        torch.Generator().manual_seed(21),
    )
    output.total.backward()

    assert any(parameter.grad is not None for parameter in model.motion_scaffold.parameters())
    assert any(parameter.grad is not None for parameter in model.router.parameters())
    assert any(parameter.grad is not None for parameter in model.residual_dit.parameters())
    assert all(parameter.grad is None for parameter in model.priors.parameters())


def test_tardis_objective_normalizer_state_round_trips_exactly() -> None:
    model = build_tiny_tardis().model.train()
    objective = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    point = CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)).at_step(5)
    objective(
        model,
        training_batch(),
        point,
        torch.Generator().manual_seed(5),
    )
    state = objective.state_dict()

    restored = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    restored.load_state_dict(state)

    restored_state = restored.state_dict()
    assert restored_state.keys() == state.keys()
    for key in ("version", "weights", "temporal_levels", "normalizers"):
        assert restored_state[key] == state[key]
    distiller = state["distiller"]
    restored_distiller = restored_state["distiller"]
    assert isinstance(distiller, dict) and isinstance(restored_distiller, dict)
    for key in ("version", "ema_decay", "transition_ratio"):
        assert restored_distiller[key] == distiller[key]
    teacher = distiller["teacher"]
    restored_teacher = restored_distiller["teacher"]
    assert isinstance(teacher, dict) and isinstance(restored_teacher, dict)
    assert teacher.keys() == restored_teacher.keys()
    for name, value in teacher.items():
        assert isinstance(value, torch.Tensor)
        assert torch.equal(value, restored_teacher[name])


def test_tardis_objective_rank_state_restores_only_rank_local_normalizers() -> None:
    source = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    source.normalizers["tc"].value = 3.5
    rank_state = source.rank_state_dict()

    target = TARDISObjective(perceptual_metric=TinyPerceptualMetric())
    target.normalizers["tc"].value = 9.0
    target.load_rank_state_dict(rank_state)

    assert target.normalizers["tc"].value == 3.5
    assert target.rank_state_dict() == rank_state
