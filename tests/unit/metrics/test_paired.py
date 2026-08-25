from __future__ import annotations

import importlib
from types import ModuleType

import pytest
import torch


def _paired() -> ModuleType:
    return importlib.import_module("tardis.metrics.paired")


def _constant_video(values: list[float], *, height: int = 4, width: int = 4) -> torch.Tensor:
    frames = [torch.full((3, height, width), value) for value in values]
    return torch.stack(frames)


def test_official_tc_matches_reference_delta_equation_and_identity_is_zero() -> None:
    metric = _paired().TemporalConsistencyMetric()
    generated = _constant_video([0.0, 0.5, 1.0])
    reference = _constant_video([0.0, 1.0, 1.0])

    metric.update(generated, reference)

    assert metric.compute(aggregation="macro") == pytest.approx(0.5)
    assert metric.compute(aggregation="micro") == pytest.approx(0.5)

    identical = _paired().TemporalConsistencyMetric()
    identical.update(reference, reference.clone())
    assert identical.compute() == pytest.approx(0.0)


def test_tc_keeps_per_video_macro_separate_from_transition_element_micro() -> None:
    metric = _paired().TemporalConsistencyMetric()
    metric.update(_constant_video([0.0, 1.0], height=1, width=1), torch.zeros(2, 3, 1, 1))
    metric.update(torch.zeros(4, 3, 1, 1), torch.zeros(4, 3, 1, 1))

    assert metric.compute(aggregation="macro") == pytest.approx(0.5)
    assert metric.compute(aggregation="micro") == pytest.approx(0.25)


class _InjectedLPIPS:
    provenance_id = "unit-test/lpips"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        assert generated.ndim == 4
        assert generated.shape == reference.shape
        assert generated.min() >= -1 and generated.max() <= 1
        return (generated - reference).abs().mean(dim=(1, 2, 3))


def test_lpips_streams_injected_frame_scores_with_macro_and_micro_aggregation() -> None:
    injected = _InjectedLPIPS()
    metric = _paired().LPIPSMetric(injected)
    metric.update(_constant_video([0.2, 0.4]), torch.zeros(2, 3, 4, 4))
    metric.update(_constant_video([0.6]), torch.zeros(1, 3, 4, 4))

    assert injected.calls == 2
    assert metric.provenance_id == "unit-test/lpips"
    assert metric.compute(aggregation="macro") == pytest.approx(0.45)
    assert metric.compute(aggregation="micro") == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("generated", "match"),
    [
        (torch.zeros(3, 4, 4), "shape"),
        (torch.zeros(2, 1, 4, 4), "three channels"),
        (torch.zeros(2, 3, 4, 4, dtype=torch.int64), "floating point"),
        (torch.full((2, 3, 4, 4), 1.01), r"\[-1, 1\]"),
    ],
)
def test_lpips_rejects_videos_outside_shape_dtype_and_range_contracts(
    generated: torch.Tensor,
    match: str,
) -> None:
    metric = _paired().LPIPSMetric(_InjectedLPIPS())

    with pytest.raises(ValueError, match=match):
        metric.update(generated, torch.zeros_like(generated))


class _InvalidLPIPS:
    provenance_id = "unit-test/invalid-lpips"

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        del reference
        return -torch.ones(generated.shape[0] + 1)


def test_lpips_requires_one_finite_nonnegative_score_per_frame() -> None:
    metric = _paired().LPIPSMetric(_InvalidLPIPS())

    with pytest.raises(ValueError, match="one finite non-negative value per frame"):
        metric.update(torch.zeros(2, 3, 4, 4), torch.zeros(2, 3, 4, 4))


def test_multichannel_ssim_identity_is_one_and_nonidentity_is_finite_and_bounded() -> None:
    frame = torch.linspace(-1.0, 1.0, 3 * 3 * 12 * 12).reshape(3, 3, 12, 12)
    identity = _paired().SSIMMetric()
    identity.update(frame, frame.clone())

    assert identity.compute() == pytest.approx(1.0, abs=1.0e-6)

    changed = _paired().SSIMMetric()
    changed.update(frame, -frame)
    value = changed.compute()
    assert torch.isfinite(torch.tensor(value))
    assert -1.0 <= value <= 1.0
