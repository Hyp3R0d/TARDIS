from __future__ import annotations

import torch

from tardis.models.residual import ResidualDenoisingContext, SparseResidualDiT
from tardis.models.router import InnovationSelection
from tardis.training.distillation import CausalResidualDistiller
from tardis.training.losses import crcd_loss


def _student() -> SparseResidualDiT:
    model = SparseResidualDiT(
        latent_channels=4,
        patch_size=2,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        text_dim=12,
        motion_dim=10,
        state_dim=8,
        max_grid_size=8,
    )
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.02)
        model.output_projection.bias.fill_(0.01)
    return model


def _context() -> ResidualDenoisingContext:
    active_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    active_mask.flatten()[torch.tensor([0, 3, 7, 12])] = True
    selection = InnovationSelection(
        indices=torch.tensor([[0, 3, 7, 12]]),
        valid_tokens=torch.ones(1, 4, dtype=torch.bool),
        active_counts=torch.tensor([4]),
        active_mask=active_mask,
    )
    noise = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(3))
    clean = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(5))
    time = torch.tensor([0.8])
    return ResidualDenoisingContext(
        noisy_residual=(1 - time[:, None, None, None]) * clean + time[:, None, None, None] * noise,
        diffusion_noise=noise,
        transported_prior=torch.randn(1, 4, 8, 8),
        diffusion_time=time,
        event_probability=torch.rand(1, 1, 8, 8),
        text_tokens=torch.randn(1, 5, 12),
        text_mask=torch.ones(1, 5, dtype=torch.bool),
        motion_tokens=torch.randn(1, 3, 10),
        state_tokens=torch.randn(1, 2, 8),
        selection=selection,
    )


def _predict(student: SparseResidualDiT, context: ResidualDenoisingContext) -> torch.Tensor:
    return student(
        noisy_residual=context.noisy_residual,
        transported_prior=context.transported_prior,
        diffusion_time=context.diffusion_time,
        event_probability=context.event_probability,
        text_tokens=context.text_tokens,
        text_mask=context.text_mask,
        motion_tokens=context.motion_tokens,
        state_tokens=context.state_tokens,
        selection=context.selection,
    ).residual


def test_crcd_ema_teacher_builds_detached_lower_noise_target_on_same_causal_context() -> None:
    student = _student()
    context = _context()
    distiller = CausalResidualDistiller(ema_decay=0.5, transition_ratio=0.5)

    target = distiller.build_target(student, context)
    student_prediction = _predict(student, context)
    loss = crcd_loss(student_prediction, target)
    loss.backward()

    assert distiller.teacher is not None
    assert not distiller.teacher.training
    assert all(not parameter.requires_grad for parameter in distiller.teacher.parameters())
    assert not target.teacher_residual.requires_grad
    assert torch.equal(target.causal_history, context.transported_prior.detach())
    assert student.output_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in distiller.teacher.parameters())


def test_crcd_teacher_updates_by_ema_only_when_explicitly_requested() -> None:
    student = _student()
    distiller = CausalResidualDistiller(ema_decay=0.75, transition_ratio=0.5)
    distiller.build_target(student, _context())
    assert distiller.teacher is not None
    name, teacher_parameter = next(iter(distiller.teacher.named_parameters()))
    old_teacher = teacher_parameter.detach().clone()
    student_parameter = dict(student.named_parameters())[name]
    with torch.no_grad():
        student_parameter.add_(2)

    assert torch.equal(teacher_parameter, old_teacher)
    distiller.update(student)

    expected = old_teacher * 0.75 + student_parameter.detach() * 0.25
    assert torch.allclose(dict(distiller.teacher.named_parameters())[name], expected)


def test_crcd_teacher_state_round_trip_restores_exact_target_before_first_use() -> None:
    student = _student()
    context = _context()
    original = CausalResidualDistiller(ema_decay=0.9, transition_ratio=0.25)
    expected = original.build_target(student, context)
    state = original.state_dict()

    restored = CausalResidualDistiller(ema_decay=0.9, transition_ratio=0.25)
    restored.load_state_dict(state)
    actual = restored.build_target(student, context)

    assert torch.equal(actual.teacher_residual, expected.teacher_residual)
    assert torch.equal(actual.causal_history, expected.causal_history)
