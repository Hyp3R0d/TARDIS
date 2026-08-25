from __future__ import annotations

import torch

from tardis.models.state import CausalStateUpdater


def build_updater() -> CausalStateUpdater:
    updater = CausalStateUpdater(latent_channels=4, state_channels=8, anchor_decay=0.75)
    with torch.no_grad():
        updater.encoder.weight.zero_()
        updater.encoder.bias.zero_()
        for channel in range(4):
            updater.encoder.weight[channel, channel, 0, 0] = 1
    return updater


def test_state_shape_is_independent_of_sequence_length() -> None:
    updater = build_updater()
    state = updater.initialize(torch.randn(2, 4, 8, 8), detach=True)
    initial_shapes = (
        state.latent.shape,
        state.short.shape,
        state.anchor.shape,
        state.innovation_hazard.shape,
    )

    for _ in range(100):
        state = updater.update(
            state,
            torch.randn(2, 4, 8, 8),
            innovation_probability=torch.rand(2, 1, 8, 8),
            reset_mask=torch.zeros(2, dtype=torch.bool),
            detach=True,
        )

    assert (
        state.latent.shape,
        state.short.shape,
        state.anchor.shape,
        state.innovation_hazard.shape,
    ) == initial_shapes
    assert state.frame_index.tolist() == [100, 100]


def test_anchor_uses_exponential_update_weighted_by_confidence() -> None:
    updater = build_updater()
    initial = torch.zeros(1, 4, 2, 2)
    state = updater.initialize(initial, detach=False)
    current = torch.ones(1, 4, 2, 2)

    updated = updater.update(
        state,
        current,
        innovation_probability=torch.zeros(1, 1, 2, 2),
        reset_mask=torch.zeros(1, dtype=torch.bool),
        detach=False,
    )

    assert torch.allclose(updated.anchor[:, :4], torch.full((1, 4, 2, 2), 0.25))
    assert torch.count_nonzero(updated.anchor[:, 4:]) == 0


def test_high_innovation_preserves_anchor_but_updates_short_state() -> None:
    updater = build_updater()
    state = updater.initialize(torch.zeros(1, 4, 2, 2), detach=False)

    updated = updater.update(
        state,
        torch.ones(1, 4, 2, 2),
        innovation_probability=torch.ones(1, 1, 2, 2),
        reset_mask=torch.zeros(1, dtype=torch.bool),
        detach=False,
    )

    assert torch.count_nonzero(updated.anchor) == 0
    assert torch.count_nonzero(updated.short[:, :4]) > 0


def test_reset_replaces_history_and_resets_frame_index_per_batch() -> None:
    updater = build_updater()
    state = updater.initialize(torch.zeros(2, 4, 2, 2), detach=False)
    current = torch.stack((torch.ones(4, 2, 2), torch.full((4, 2, 2), 2.0)))

    updated = updater.update(
        state,
        current,
        innovation_probability=torch.zeros(2, 1, 2, 2),
        reset_mask=torch.tensor([True, False]),
        detach=False,
    )

    assert torch.equal(updated.anchor[0, :4], current[0])
    assert torch.allclose(updated.anchor[1, :4], torch.full((4, 2, 2), 0.5))
    assert updated.frame_index.tolist() == [0, 1]


def test_inference_state_detaches_all_autograd_history() -> None:
    updater = build_updater()
    initial = torch.randn(1, 4, 4, 4, requires_grad=True)
    state = updater.initialize(initial, detach=True)
    current = torch.randn(1, 4, 4, 4, requires_grad=True)

    state = updater.update(
        state,
        current,
        innovation_probability=torch.rand(1, 1, 4, 4),
        reset_mask=torch.zeros(1, dtype=torch.bool),
        detach=True,
    )

    assert state.latent.grad_fn is None
    assert state.short.grad_fn is None
    assert state.anchor.grad_fn is None
    assert state.innovation_hazard.grad_fn is None


def test_state_accepts_a_settled_innovation_hazard_and_resets_it_on_scene_cut() -> None:
    updater = build_updater()
    state = updater.initialize(torch.zeros(2, 4, 2, 2), detach=False)
    settled = torch.stack(
        (
            torch.ones(1, 2, 2),
            torch.full((1, 2, 2), 2.0),
        )
    )

    updated = updater.update(
        state,
        torch.zeros(2, 4, 2, 2),
        innovation_probability=torch.zeros(2, 1, 2, 2),
        innovation_hazard=settled,
        reset_mask=torch.tensor([True, False]),
        detach=False,
    )

    assert torch.count_nonzero(updated.innovation_hazard[0]) == 0
    assert torch.equal(updated.innovation_hazard[1], settled[1])


def test_state_exports_fixed_spatial_and_token_conditions() -> None:
    updater = build_updater()
    state = updater.initialize(torch.randn(2, 4, 4, 4), detach=True)

    spatial = state.spatial_condition()
    tokens = state.anchor_tokens(stride=2)

    assert spatial.shape == (2, 8, 4, 4)
    assert tokens.shape == (2, 4, 8)
