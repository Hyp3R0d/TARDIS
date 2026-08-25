from __future__ import annotations

import pytest
import torch

from tardis.training.losses import (
    EmaLossNormalizer,
    LossWeights,
    build_crcd_target,
    crcd_loss,
    diffusion_loss,
    flow_loss,
    lite_residual_loss,
    lpips_loss,
    multi_scale_temporal_consistency_loss,
    official_temporal_consistency_loss,
    router_loss,
    transport_loss,
    weighted_loss,
)


def test_official_tc_matches_frame_difference_equation_and_multiscale_level_zero() -> None:
    generated = torch.tensor([0.0, 2.0, 5.0]).reshape(1, 3, 1, 1, 1)
    reference = torch.tensor([0.0, 1.0, 4.0]).reshape(1, 3, 1, 1, 1)

    expected = torch.tensor(0.5)
    assert official_temporal_consistency_loss(generated, reference) == expected
    assert multi_scale_temporal_consistency_loss(generated, reference, levels=1) == expected


def test_mechanism_losses_are_zero_at_exact_targets_and_support_masks() -> None:
    target = torch.ones(1, 2, 4, 4)
    zero = torch.zeros_like(target)
    visibility = torch.ones(1, 1, 4, 4)

    assert diffusion_loss(target, target) == 0
    assert transport_loss(target, target, visibility) == 0
    assert flow_loss(zero, zero) == 0
    assert lite_residual_loss(zero, zero, torch.zeros_like(visibility)) == 0
    assert router_loss(torch.zeros_like(visibility), torch.zeros_like(visibility)) > 0
    assert weighted_loss({"tc": torch.tensor(2.0)}, LossWeights(tc=0.5)).item() == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast requires a GPU")
def test_router_loss_is_safe_under_cuda_autocast() -> None:
    logits = torch.randn(1, 1, 4, 4, device="cuda", requires_grad=True)
    target = torch.rand_like(logits)

    with torch.autocast("cuda", dtype=torch.float16):
        loss = router_loss(logits.sigmoid(), target)
    loss.backward()

    assert loss.dtype == torch.float32
    assert logits.grad is not None


def test_lpips_loss_uses_an_injected_metric_without_eager_model_dependencies() -> None:
    class TinyPerceptualMetric:
        def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return (generated - reference).abs().mean(dim=(1, 2, 3), keepdim=True)

    generated = torch.zeros(2, 3, 8, 8)
    reference = torch.ones_like(generated)

    assert lpips_loss(TinyPerceptualMetric(), generated, reference).item() == pytest.approx(1.0)


def test_lpips_training_loss_checkpoints_bounded_frame_chunks() -> None:
    class RecordingPerceptualMetric:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(generated.shape[0])
            return (generated - reference).square().mean(dim=(1, 2, 3))

    metric = RecordingPerceptualMetric()
    generated = torch.zeros(5, 3, 8, 8, requires_grad=True)
    reference = torch.ones_like(generated)

    loss = lpips_loss(metric, generated, reference, frame_chunk_size=2)
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert metric.batch_sizes and max(metric.batch_sizes) == 2
    assert generated.grad is not None


def test_ema_normalizer_has_checkpointable_state_and_detached_denominator() -> None:
    normalizer = EmaLossNormalizer(decay=0.5, epsilon=1.0e-6)
    loss = torch.tensor(4.0, requires_grad=True)

    normalized = normalizer.normalize(loss)
    normalized.backward()

    assert normalizer.value == pytest.approx(2.5)
    assert normalized.item() == pytest.approx(1.6)
    assert loss.grad is not None
    state = normalizer.state_dict()
    restored = EmaLossNormalizer(decay=0.5, epsilon=1.0e-6)
    restored.load_state_dict(state)
    assert restored.value == pytest.approx(normalizer.value)


def test_crcd_target_is_teacher_detached_but_uses_student_compatible_history() -> None:
    teacher_residual = torch.randn(1, 4, 8, 8, requires_grad=True)
    student_history = torch.randn(1, 8, 8, 8, requires_grad=True)
    target = build_crcd_target(teacher_residual, student_history)
    student_residual = torch.zeros_like(teacher_residual, requires_grad=True)

    loss = crcd_loss(student_residual, target)
    loss.backward()

    assert not target.teacher_residual.requires_grad
    assert not target.causal_history.requires_grad
    assert torch.equal(target.causal_history, student_history.detach())
    assert student_residual.grad is not None
    assert teacher_residual.grad is None
