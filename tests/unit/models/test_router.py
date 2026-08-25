from __future__ import annotations

import math

import pytest
import torch

from tardis.models.router import (
    VisibilityCalibratedInnovationRouter,
    brier_score,
    expected_calibration_error,
    oracle_innovation,
    select_innovation_budget,
)


def test_oracle_innovation_has_correct_visibility_and_residual_extremes() -> None:
    visibility = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    transported = torch.zeros(1, 2, 2, 2)
    target = transported.clone()
    target[:, :, 1, 1] = 100.0

    oracle = oracle_innovation(target, transported, visibility, residual_temperature=1.0)

    assert oracle.shape == (1, 1, 2, 2)
    assert oracle[0, 0, 0, 0] == 0
    assert oracle[0, 0, 0, 1] == 1
    assert oracle[0, 0, 1, 1] > 0.999


def test_oracle_innovation_stops_gradient_through_transport_prior() -> None:
    target = torch.ones(1, 2, 2, 2, requires_grad=True)
    transported = torch.zeros(1, 2, 2, 2, requires_grad=True)
    visibility = torch.ones(1, 1, 2, 2)

    oracle = oracle_innovation(
        target,
        transported,
        visibility,
        residual_temperature=1.0,
        detach_transport=True,
    )
    oracle.mean().backward()

    assert target.grad is not None
    assert transported.grad is None


def test_oracle_innovation_uses_quotient_normal_energy_when_provided() -> None:
    transported = torch.zeros(1, 4, 3, 3)
    target = torch.ones_like(transported)
    visibility = torch.ones(1, 1, 3, 3)

    oracle = oracle_innovation(
        target,
        transported,
        visibility,
        residual_temperature=0.25,
        quotient_residual=torch.zeros_like(target),
    )

    assert torch.count_nonzero(oracle) == 0


def test_budget_selection_respects_exact_ceiling_including_halo() -> None:
    scores = torch.zeros(1, 1, 5, 5)
    scores[0, 0, 2, 2] = 1.0
    scores[0, 0, 0, 0] = 0.9
    active_ratio = 0.2

    selection = select_innovation_budget(
        scores,
        active_ratio=active_ratio,
        threshold=0.1,
        halo_radius=1,
    )

    budget = math.ceil(active_ratio * 25)
    assert selection.indices.shape == (1, budget)
    assert selection.active_mask.sum() == budget
    assert selection.active_mask[0, 0, 2, 2]
    assert selection.active_mask[0, 0, 1:4, 1:4].sum() >= 4


def test_budget_selection_ties_are_deterministic_by_flat_index() -> None:
    scores = torch.ones(2, 1, 2, 3)

    first = select_innovation_budget(scores, active_ratio=0.5, threshold=0, halo_radius=0)
    second = select_innovation_budget(scores, active_ratio=0.5, threshold=0, halo_radius=0)

    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.indices[0], torch.tensor([0, 1, 2]))


def test_threshold_can_leave_budget_slots_inactive_without_fake_tokens() -> None:
    scores = torch.tensor([[[[0.9, 0.1], [0.05, 0.0]]]])

    selection = select_innovation_budget(
        scores,
        active_ratio=1.0,
        threshold=0.5,
        halo_radius=0,
    )

    assert selection.active_counts.tolist() == [1]
    assert selection.valid_tokens[0].tolist() == [True, False, False, False]
    assert selection.active_mask.sum() == 1


def test_calibration_helpers_match_hand_computation() -> None:
    probabilities = torch.tensor([0.1, 0.4, 0.8, 0.9])
    targets = torch.tensor([0.0, 1.0, 1.0, 1.0])

    assert brier_score(probabilities, targets).item() == torch.tensor(0.105).item()
    ece = expected_calibration_error(probabilities, targets, num_bins=2)
    assert ece.item() == pytest.approx(0.2, abs=1e-7)


def build_router() -> VisibilityCalibratedInnovationRouter:
    return VisibilityCalibratedInnovationRouter(
        latent_channels=4,
        motion_channels=6,
        state_channels=8,
        text_dim=16,
        hidden_size=24,
        patch_size=2,
        active_ratio=0.25,
        threshold=0.1,
        halo_radius=1,
    )


def test_router_predicts_patch_probabilities_and_fixed_budget() -> None:
    router = build_router().eval()
    transported = torch.randn(2, 4, 8, 8)
    visibility = torch.rand(2, 1, 8, 8)
    motion = torch.randn(2, 6, 8, 8)
    state = torch.randn(2, 8, 8, 8)
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)

    output = router(transported, visibility, motion, state, text, mask)

    assert output.pixel_probability.shape == (2, 1, 8, 8)
    assert output.patch_probability.shape == (2, 1, 4, 4)
    assert output.selection.indices.shape == (2, 4)
    assert output.selection.active_counts.max() <= 4
    assert output.pixel_probability.min() >= 0
    assert output.pixel_probability.max() <= 1


def test_router_training_soft_gate_is_differentiable() -> None:
    router = build_router().train()
    transported = torch.randn(1, 4, 8, 8, requires_grad=True)
    visibility = torch.rand(1, 1, 8, 8)
    motion = torch.randn(1, 6, 8, 8)
    state = torch.randn(1, 8, 8, 8)
    text = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)

    output = router(transported, visibility, motion, state, text, mask)
    output.pixel_probability.mean().backward()

    assert transported.grad is not None
    assert any(parameter.grad is not None for parameter in router.parameters())
