from __future__ import annotations

import torch

from tardis.models.tardis import TARDISTrainingBatch
from tests.helpers.tardis_model import build_tiny_tardis


def test_training_reference_is_teacher_only_and_temporal_network_receives_gradients() -> None:
    assembly = build_tiny_tardis()
    model = assembly.model.train()
    frame = torch.randn(1, 1, 3, 16, 16)
    batch = TARDISTrainingBatch(
        prompts=["a fixed geometric object"],
        video=frame.expand(-1, 4, -1, -1, -1).contiguous(),
    )

    output = model.forward_train(batch, stage="transport")
    output.predicted_latents[:, 1:].square().mean().backward()

    assert output.stage == "transport"
    assert output.target_latents.shape == (1, 4, 4, 8, 8)
    assert output.predicted_latents.shape == output.target_latents.shape
    assert len(output.transitions) == 3
    assert assembly.motion_teacher.calls == 1
    assert all(parameter.grad is None for parameter in assembly.priors.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("priors.")
    )


def test_training_uses_teacher_forcing_but_generation_never_calls_motion_teacher() -> None:
    assembly = build_tiny_tardis()
    frame = torch.randn(1, 1, 3, 16, 16)
    batch = TARDISTrainingBatch(
        prompts=["a moving cube"],
        video=frame.expand(-1, 3, -1, -1, -1).contiguous(),
    )
    assembly.model.forward_train(batch, stage="router")
    teacher_calls = assembly.motion_teacher.calls

    generated = assembly.model.generate(
        ["a moving cube"],
        num_frames=3,
        fps=8,
        generator=torch.Generator().manual_seed(3),
    )

    assert teacher_calls == 1
    assert assembly.motion_teacher.calls == teacher_calls
    assert generated.latents.shape[1] == 3
