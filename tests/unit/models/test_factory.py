from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tardis.models.factory import (
    TARDISFactoryOptions,
    build_production_tardis,
    build_tardis_from_priors,
    load_tardis_temporal_state_dict,
    tardis_temporal_state_dict,
)
from tardis.models.priors import FrozenPriorBundle
from tests.helpers.tardis_model import build_tiny_tardis


def test_factory_builds_trainable_temporal_network_around_frozen_priors() -> None:
    assembly = build_tiny_tardis()
    options = TARDISFactoryOptions(
        height=16,
        width=16,
        motion_noise_channels=3,
        state_channels=8,
        motion_hidden_size=16,
        motion_token_dim=10,
        motion_token_stride=2,
        router_hidden_size=16,
        patch_size=2,
        residual_hidden_size=16,
        residual_layers=1,
        residual_heads=4,
        state_token_stride=2,
        active_ratio=0.25,
        gradient_checkpointing=True,
    )

    model = build_tardis_from_priors(
        assembly.priors,
        motion_teacher=assembly.motion_teacher,
        options=options,
    )

    assert model.config.height == 16
    assert model.config.width == 16
    assert model.router.active_ratio == pytest.approx(0.25)
    assert model.residual_dit.gradient_checkpointing is True
    assert all(not parameter.requires_grad for parameter in model.priors.parameters())
    assert any(parameter.requires_grad for parameter in model.motion_scaffold.parameters())
    assert any(parameter.requires_grad for parameter in model.residual_dit.parameters())


def test_factory_rejects_incompatible_latent_geometry() -> None:
    assembly = build_tiny_tardis()
    with pytest.raises(ValueError, match="height and width must be divisible"):
        build_tardis_from_priors(
            assembly.priors,
            motion_teacher=assembly.motion_teacher,
            options=TARDISFactoryOptions(height=15, width=16),
        )


def test_factory_model_can_run_a_tiny_prompt_only_transition() -> None:
    assembly = build_tiny_tardis()
    model = build_tardis_from_priors(
        assembly.priors,
        motion_teacher=assembly.motion_teacher,
        options=TARDISFactoryOptions(
            height=16,
            width=16,
            motion_noise_channels=3,
            state_channels=8,
            motion_hidden_size=16,
            motion_token_dim=10,
            motion_token_stride=2,
            router_hidden_size=16,
            patch_size=2,
            residual_hidden_size=16,
            residual_layers=1,
            residual_heads=4,
            state_token_stride=2,
            active_ratio=0.25,
        ),
    ).eval()
    output = model.generate(
        ["a blue comet"],
        num_frames=2,
        fps=8,
        generator=torch.Generator().manual_seed(4),
    )
    assert output.video.shape == (1, 2, 3, 16, 16)


def test_production_factory_loads_shared_prior_and_moves_full_model_lifecycle() -> None:
    assembly = build_tiny_tardis()
    calls: list[dict[str, object]] = []

    def prior_loader(model_id: str, **kwargs: object) -> FrozenPriorBundle:
        calls.append({"model_id": model_id, **kwargs})
        return assembly.priors

    model = build_production_tardis(
        model_id="mirror/sd-turbo",
        cache_dir=Path("model-cache"),
        torch_dtype=torch.float64,
        device=torch.device("cpu"),
        local_files_only=True,
        motion_teacher=assembly.motion_teacher,
        options=TARDISFactoryOptions(
            height=16,
            width=16,
            motion_noise_channels=3,
            state_channels=8,
            motion_hidden_size=16,
            motion_token_dim=10,
            motion_token_stride=2,
            router_hidden_size=16,
            patch_size=2,
            residual_hidden_size=16,
            residual_layers=1,
            residual_heads=4,
            state_token_stride=2,
        ),
        prior_loader=prior_loader,
    )

    assert calls == [
        {
            "model_id": "mirror/sd-turbo",
            "cache_dir": "model-cache",
            "torch_dtype": torch.float64,
            "device": torch.device("cpu"),
            "local_files_only": True,
        }
    ]
    assert all(
        parameter.device.type == "cpu" and parameter.dtype == torch.float64
        for parameter in model.parameters()
    )
    model.train()
    assert not model.priors.training
    assert all(not module.training for module in model.priors.modules())


def test_temporal_checkpoint_state_excludes_and_never_overwrites_frozen_priors() -> None:
    source = build_tiny_tardis().model
    target = build_tiny_tardis().model
    state = tardis_temporal_state_dict(source)
    prior_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if name.startswith("priors.")
    }

    assert state
    assert all(not name.startswith("priors.") for name in state)
    replacement = {
        name: value.detach().clone().add_(1)
        if value.is_floating_point()
        else value.detach().clone()
        for name, value in state.items()
    }
    load_tardis_temporal_state_dict(target, replacement)

    for name, value in target.state_dict().items():
        if name.startswith("priors."):
            assert torch.equal(value, prior_before[name])
        else:
            assert torch.equal(value, replacement[name])

    with pytest.raises(ValueError, match="frozen prior"):
        load_tardis_temporal_state_dict(target, {"priors.codec.weight": torch.ones(())})


def test_temporal_checkpoint_migrates_complete_keyframe_and_transition_branches() -> None:
    source = build_tiny_tardis().model
    target = build_tiny_tardis(keyframe_residual_generation=True).model
    with torch.no_grad():
        source.residual_dit.input_projection.weight.fill_(0.25)
        source.residual_dit.output_projection.weight.fill_(0.75)
        source.residual_dit.output_projection.bias.fill_(0.75)
        source.lite_corrector.input_projection.weight.fill_(0.5)
        source.lite_corrector.output_projection.weight.fill_(0.9)
        source.lite_corrector.output_projection.bias.fill_(0.9)
    legacy = {
        name: value
        for name, value in tardis_temporal_state_dict(source).items()
        if not name.startswith(("keyframe_residual_dit.", "transition_lite_corrector."))
    }

    load_tardis_temporal_state_dict(target, legacy)

    assert torch.count_nonzero(target.keyframe_residual_dit.output_projection.weight) == 0
    assert torch.count_nonzero(target.keyframe_residual_dit.output_projection.bias) == 0
    assert torch.count_nonzero(target.transition_lite_corrector.output_projection.weight) == 0
    assert torch.count_nonzero(target.transition_lite_corrector.output_projection.bias) == 0
    assert torch.equal(
        target.keyframe_residual_dit.input_projection.weight,
        source.residual_dit.input_projection.weight,
    )
    assert torch.equal(
        target.transition_lite_corrector.input_projection.weight,
        source.lite_corrector.input_projection.weight,
    )
