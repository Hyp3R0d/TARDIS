from __future__ import annotations

import gc
import importlib
import io
import weakref
from types import ModuleType

import pytest
import torch


def _metrics() -> ModuleType:
    return importlib.import_module("tardis.metrics")


def _constant_video(values: list[float]) -> torch.Tensor:
    return torch.stack([torch.full((3, 12, 12), value) for value in values])


class _LPIPSFeatures:
    provenance_id = "unit-test/lpips"

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3))


class _FrameFeatures:
    provenance_id = "unit-test/inception"
    feature_dim = 2

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        means = video.mean(dim=(1, 2, 3))
        return torch.stack((means, means.square()), dim=1)


class _VideoFeatures:
    provenance_id = "unit-test/i3d"
    feature_dim = 2

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        mean = video.mean()
        return torch.stack((mean, mean.square())).reshape(1, 2)


class _CLIPFeatures:
    provenance_id = "unit-test/openclip"
    feature_dim = 2

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        means = video.mean(dim=(1, 2, 3))
        return torch.stack((torch.ones_like(means), means), dim=1)

    def encode_text(self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor([[1.0, 0.0]])


class _KnownCosineFeatures:
    provenance_id = "unit-test/known-cosine"
    feature_dim = 2

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape[0] == 2:
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return torch.tensor([[-1.0, 0.0]])

    def encode_text(self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor([[1.0, 0.0]])


def _suite() -> object:
    metrics = _metrics()
    return metrics.MetricSuite(
        lpips=metrics.LPIPSMetric(_LPIPSFeatures()),
        fid=metrics.FIDMetric(_FrameFeatures()),
        fvd=metrics.FVDMetric(_VideoFeatures()),
        clipscore=metrics.CLIPScoreMetric(_CLIPFeatures()),
    )


def _updates() -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    return [
        (_constant_video([0.0, 1.0]), _constant_video([0.0, 0.0]), "first"),
        (_constant_video([0.0, 0.0, 0.0, 0.0]), _constant_video([0.0] * 4), "second"),
    ]


def _assert_results_equal(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
) -> None:
    assert first.keys() == second.keys()
    for aggregation in first:
        assert first[aggregation].keys() == second[aggregation].keys()
        for metric in first[aggregation]:
            assert first[aggregation][metric] == pytest.approx(second[aggregation][metric])


def test_clipscore_aggregates_frame_cosines_as_video_macro_and_frame_micro() -> None:
    metric = _metrics().CLIPScoreMetric(_KnownCosineFeatures())
    metric.update(torch.zeros(2, 3, 4, 4), "first")
    metric.update(torch.zeros(1, 3, 4, 4), "second")

    assert metric.compute(aggregation="macro") == pytest.approx(-0.25)
    assert metric.compute(aggregation="micro") == pytest.approx(0.0)


def test_metric_suite_reports_six_metrics_with_separate_macro_and_micro_values() -> None:
    suite = _suite()
    for generated, reference, prompt in _updates():
        suite.update(generated, reference, prompt)

    result = suite.compute()

    expected = {"tc", "lpips", "fvd", "fid", "clipscore", "ssim"}
    assert result.keys() == {"macro", "micro"}
    assert result["macro"].keys() == expected
    assert result["micro"].keys() == expected
    assert result["macro"]["tc"] == pytest.approx(0.5)
    assert result["micro"]["tc"] == pytest.approx(0.25)
    assert result["macro"]["fid"] == pytest.approx(result["micro"]["fid"])
    assert result["macro"]["fvd"] == pytest.approx(result["micro"]["fvd"])


def test_metric_suite_checkpoint_resume_and_merge_match_uninterrupted_stream() -> None:
    updates = _updates()
    uninterrupted = _suite()
    for update in updates:
        uninterrupted.update(*update)

    checkpointed = _suite()
    checkpointed.update(*updates[0])
    buffer = io.BytesIO()
    torch.save(checkpointed.state_dict(), buffer)
    buffer.seek(0)
    resumed = _suite()
    resumed.load_state_dict(torch.load(buffer, weights_only=True))
    resumed.update(*updates[1])

    left = _suite()
    right = _suite()
    left.update(*updates[0])
    right.update(*updates[1])
    left.merge(right)

    _assert_results_equal(resumed.compute(), uninterrupted.compute())
    _assert_results_equal(left.compute(), uninterrupted.compute())


def test_metric_suite_does_not_retain_processed_video_tensors() -> None:
    suite = _suite()
    generated, reference, prompt = _updates()[0]
    generated_reference = weakref.ref(generated)
    reference_reference = weakref.ref(reference)

    suite.update(generated, reference, prompt)
    del generated
    del reference
    gc.collect()

    assert generated_reference() is None
    assert reference_reference() is None
