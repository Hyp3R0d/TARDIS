from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from tardis.models.motion import FlowMotionTeacher, PromptMotionScaffold
from tardis.training.losses import flow_loss


def translated_texture(dx: int, dy: int, *, height: int = 64, width: int = 64) -> torch.Tensor:
    rng = np.random.default_rng(7)
    base = np.zeros((height, width), dtype=np.uint8)
    base[14:48, 14:48] = rng.integers(20, 240, size=(34, 34), dtype=np.uint8)
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(base, transform, (width, height), flags=cv2.INTER_LINEAR)
    frames = np.stack([base, moved], axis=0)
    rgb = np.repeat(frames[:, None, :, :], 3, axis=1)
    return torch.from_numpy(rgb).float().div(127.5).sub(1.0).unsqueeze(0)


def test_prompt_motion_scaffold_starts_from_identity_transport() -> None:
    scaffold = PromptMotionScaffold(
        text_dim=12,
        state_channels=8,
        noise_channels=3,
        hidden_size=16,
        motion_token_dim=10,
        token_stride=2,
        max_flow_pixels=2.0,
        num_time_frequencies=2,
    )

    output = scaffold(
        torch.randn(2, 5, 12),
        torch.ones(2, 5, dtype=torch.bool),
        time=torch.tensor([0.25, 0.75]),
        state=torch.randn(2, 8, 8, 8),
        motion_noise=torch.randn(2, 3, 8, 8),
    )

    assert torch.count_nonzero(output.backward_flow) == 0
    assert output.visibility_logits.sigmoid().min().item() > 0.999


def test_farneback_teacher_recovers_forward_and_backward_direction() -> None:
    teacher = FlowMotionTeacher(pyramid_scale=0.5, levels=3, window_size=21, iterations=5)
    video = translated_texture(dx=4, dy=2)

    targets = teacher.estimate(video, output_size=(64, 64))

    assert targets.forward_flow.shape == (1, 1, 2, 64, 64)
    assert targets.backward_flow.shape == (1, 1, 2, 64, 64)
    region = (slice(None), slice(None), slice(None), slice(20, 42), slice(20, 42))
    forward = targets.forward_flow[region].median(dim=-1).values.median(dim=-1).values
    backward = targets.backward_flow[region].median(dim=-1).values.median(dim=-1).values
    assert forward[0, 0, 0].item() == pytest.approx(4.0, abs=1.0)
    assert forward[0, 0, 1].item() == pytest.approx(2.0, abs=1.0)
    assert backward[0, 0, 0].item() == pytest.approx(-4.0, abs=1.0)
    assert backward[0, 0, 1].item() == pytest.approx(-2.0, abs=1.0)


def test_teacher_visibility_is_bounded_and_cycle_consistent_interior_is_confident() -> None:
    teacher = FlowMotionTeacher(window_size=21, iterations=5)
    targets = teacher.estimate(translated_texture(dx=5, dy=0), output_size=(64, 64))

    assert targets.visibility.shape == (1, 1, 1, 64, 64)
    assert targets.visibility.min() >= 0
    assert targets.visibility.max() <= 1
    interior = targets.visibility[0, 0, 0, 22:40, 22:40].mean()
    entering_border = targets.visibility[0, 0, 0, :, :5].mean()
    assert interior > 0.7
    assert interior > entering_border


def test_teacher_rescales_flow_vectors_for_latent_grid() -> None:
    teacher = FlowMotionTeacher(window_size=21, iterations=5)

    targets = teacher.estimate(translated_texture(dx=4, dy=2), output_size=(32, 32))

    region = targets.forward_flow[0, 0, :, 10:21, 10:21]
    median = region.flatten(1).median(dim=1).values
    assert median[0].item() == pytest.approx(2.0, abs=0.8)
    assert median[1].item() == pytest.approx(1.0, abs=0.8)


def test_teacher_targets_are_nondifferentiable_constants_usable_during_backward() -> None:
    teacher = FlowMotionTeacher(window_size=21, iterations=5)
    targets = teacher.estimate(translated_texture(dx=2, dy=1), output_size=(32, 32))
    prediction = torch.zeros(targets.backward_flow.shape, requires_grad=True)

    loss = flow_loss(prediction, targets.backward_flow, targets.visibility)
    loss.backward()

    assert prediction.grad is not None
    assert not targets.backward_flow.requires_grad
    assert not targets.visibility.requires_grad


def build_scaffold() -> PromptMotionScaffold:
    return PromptMotionScaffold(
        text_dim=16,
        state_channels=8,
        noise_channels=4,
        hidden_size=32,
        motion_token_dim=24,
        token_stride=2,
        max_flow_pixels=6.0,
        num_time_frequencies=4,
    )


def test_prompt_motion_scaffold_is_prompt_only_bounded_and_multiscale() -> None:
    scaffold = build_scaffold()
    text = torch.randn(2, 8, 16)
    text_mask = torch.ones(2, 8, dtype=torch.bool)
    state = torch.randn(2, 8, 16, 16)
    noise = torch.randn(2, 4, 16, 16)
    time = torch.tensor([0.25, 0.75])

    output = scaffold(text, text_mask, time=time, state=state, motion_noise=noise)

    assert output.backward_flow.shape == (2, 2, 16, 16)
    assert output.visibility_logits.shape == (2, 1, 16, 16)
    assert output.motion_tokens.shape == (2, 64, 24)
    assert len(output.flow_pyramid) == 3
    assert output.flow_pyramid[1].shape == (2, 2, 8, 8)
    assert output.flow_pyramid[2].shape == (2, 2, 4, 4)
    assert output.backward_flow.abs().max() <= 6.0
    assert output.flow_pyramid[1].abs().max() <= 3.0


def test_prompt_motion_scaffold_is_deterministic_and_trainable() -> None:
    torch.manual_seed(4)
    scaffold = build_scaffold()
    text = torch.randn(1, 8, 16)
    mask = torch.ones(1, 8, dtype=torch.bool)
    state = torch.randn(1, 8, 16, 16)
    noise = torch.randn(1, 4, 16, 16)
    time = torch.tensor([0.5])

    first = scaffold(text, mask, time=time, state=state, motion_noise=noise)
    second = scaffold(text, mask, time=time, state=state, motion_noise=noise)
    loss = first.backward_flow.square().mean() + first.visibility_logits.square().mean()
    loss.backward()

    assert torch.equal(first.backward_flow, second.backward_flow)
    assert any(parameter.grad is not None for parameter in scaffold.parameters())
