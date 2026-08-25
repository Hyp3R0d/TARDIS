from __future__ import annotations

import pytest
import torch

from tardis.models.residual import (
    LiteResidualCorrector,
    SparseResidualDiT,
    gather_tokens,
    patchify,
    scatter_tokens,
    unpatchify,
)
from tardis.models.router import InnovationSelection


def make_selection() -> InnovationSelection:
    indices = torch.tensor([[1, 6], [2, 5]])
    valid = torch.tensor([[True, True], [True, False]])
    mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    mask[0, 0].flatten()[indices[0]] = True
    mask[1, 0].flatten()[indices[1, 0]] = True
    return InnovationSelection(
        indices=indices,
        valid_tokens=valid,
        active_counts=valid.sum(1),
        active_mask=mask,
    )


def test_patchify_unpatchify_is_exact_round_trip() -> None:
    latent = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(2, 3, 8, 10)

    tokens, grid = patchify(latent, patch_size=2)
    reconstructed = unpatchify(tokens, grid)

    assert tokens.shape == (2, 20, 12)
    assert grid.height == 4 and grid.width == 5
    assert torch.equal(reconstructed, latent)


def test_gather_and_scatter_are_batched_and_preserve_inactive_tokens() -> None:
    tokens = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    indices = torch.tensor([[1, 6], [2, 5]])
    valid = torch.tensor([[True, True], [True, False]])

    gathered = gather_tokens(tokens, indices, valid)
    updates = gathered + 100
    scattered = scatter_tokens(tokens, updates, indices, valid)

    assert torch.equal(gathered[0, 0], tokens[0, 1])
    assert torch.equal(gathered[0, 1], tokens[0, 6])
    assert torch.count_nonzero(gathered[1, 1]) == 0
    assert torch.equal(scattered[0, 1], tokens[0, 1] + 100)
    assert torch.equal(scattered[0, 6], tokens[0, 6] + 100)
    assert torch.equal(scattered[1, 5], tokens[1, 5])
    inactive = torch.tensor([0, 2, 3, 4, 5, 7])
    assert torch.equal(scattered[0, inactive], tokens[0, inactive])


def test_sparse_residual_dit_has_stable_active_count_and_zero_initialized_head() -> None:
    model = build_model()
    selection = make_selection()
    inputs = model_inputs()

    output = model(selection=selection, **inputs)

    assert output.residual.shape == (2, 4, 8, 8)
    assert output.active_tokens.shape == (2, 2, 16)
    assert torch.equal(output.active_counts, torch.tensor([2, 1]))
    assert torch.count_nonzero(output.residual) == 0
    assert torch.count_nonzero(output.active_tokens[1, 1]) == 0


def test_sparse_residual_scatter_never_changes_inactive_patches() -> None:
    model = build_model()
    with torch.no_grad():
        model.output_projection.weight.fill_(0.1)
        model.output_projection.bias.fill_(0.2)
    selection = make_selection()

    output = model(selection=selection, **model_inputs())
    tokens, _ = patchify(output.residual, patch_size=2)

    inactive0 = torch.tensor([0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    assert torch.count_nonzero(tokens[0, inactive0]) == 0
    inactive1 = torch.tensor([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    assert torch.count_nonzero(tokens[1, inactive1]) == 0
    assert torch.count_nonzero(tokens[0, selection.indices[0]]) > 0


def test_sparse_residual_dit_is_differentiable() -> None:
    model = build_model()
    inputs = model_inputs()
    inputs["noisy_residual"].requires_grad_()
    inputs["event_probability"].requires_grad_()
    output = model(selection=make_selection(), **inputs)

    loss = output.active_tokens.square().sum() + output.residual.square().sum()
    loss.backward()

    assert inputs["noisy_residual"].grad is not None
    assert inputs["event_probability"].grad is not None
    assert model.output_projection.weight.grad is not None


def test_sparse_residual_dit_checkpoints_every_training_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tardis.models.residual as residual_module

    calls = 0
    original = residual_module.checkpoint

    def tracked_checkpoint(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(residual_module, "checkpoint", tracked_checkpoint)
    model = build_model(gradient_checkpointing=True)
    inputs = model_inputs()
    inputs["noisy_residual"].requires_grad_()

    output = model(selection=make_selection(), **inputs)
    output.active_tokens.square().sum().backward()

    assert calls == len(model.blocks)
    assert inputs["noisy_residual"].grad is not None


def test_cross_attention_mask_covers_text_motion_and_state_memory() -> None:
    model = build_model()
    inputs = model_inputs()
    inputs["text_mask"][0, -2:] = False

    output = model(selection=make_selection(), **inputs)

    assert output.memory_key_padding_mask.shape == (2, 5 + 3 + 2)
    assert output.memory_key_padding_mask[0, :5].tolist() == [False, False, False, True, True]
    assert not output.memory_key_padding_mask[:, 5:].any()


def test_lite_corrector_is_bounded_low_frequency_and_trainable() -> None:
    corrector = LiteResidualCorrector(
        latent_channels=4,
        condition_channels=3,
        hidden_channels=8,
        max_magnitude=0.2,
    )
    prior = torch.randn(2, 4, 8, 8, requires_grad=True)
    condition = torch.randn(2, 3, 8, 8)
    gate = torch.rand(2, 1, 8, 8)

    correction = corrector(prior, condition, gate)
    assert torch.count_nonzero(correction) == 0
    correction.mean().backward()

    assert correction.shape == prior.shape
    assert correction.abs().max() <= 0.2
    assert prior.grad is not None


def test_lite_corrector_text_conditioning_is_zero_initialized_and_prompt_sensitive() -> None:
    corrector = LiteResidualCorrector(
        latent_channels=4,
        condition_channels=3,
        hidden_channels=8,
        max_magnitude=0.2,
        text_dim=6,
    )
    prior = torch.randn(2, 4, 8, 8)
    condition = torch.randn(2, 3, 8, 8)
    gate = torch.ones(2, 1, 8, 8)
    text = torch.randn(2, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)

    zero_conditioned = corrector(prior, condition, gate, text_embeddings=text, text_mask=mask)
    assert torch.count_nonzero(zero_conditioned) == 0

    with torch.no_grad():
        corrector.text_projection.weight.fill_(0.1)
        corrector.output_projection.weight.fill_(0.1)
    first = corrector(prior, condition, gate, text_embeddings=text, text_mask=mask)
    changed_text = text.clone()
    changed_text[0] *= -1
    second = corrector(
        prior,
        condition,
        gate,
        text_embeddings=changed_text,
        text_mask=mask,
    )
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def build_model(*, gradient_checkpointing: bool = False) -> SparseResidualDiT:
    return SparseResidualDiT(
        latent_channels=4,
        patch_size=2,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        text_dim=12,
        motion_dim=10,
        state_dim=8,
        max_grid_size=16,
        gradient_checkpointing=gradient_checkpointing,
    )


def model_inputs() -> dict[str, torch.Tensor]:
    return {
        "noisy_residual": torch.randn(2, 4, 8, 8),
        "transported_prior": torch.randn(2, 4, 8, 8),
        "diffusion_time": torch.tensor([0.2, 0.7]),
        "event_probability": torch.rand(2, 1, 8, 8),
        "text_tokens": torch.randn(2, 5, 12),
        "text_mask": torch.ones(2, 5, dtype=torch.bool),
        "motion_tokens": torch.randn(2, 3, 10),
        "state_tokens": torch.randn(2, 2, 8),
    }
