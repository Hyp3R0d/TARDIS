from __future__ import annotations

import importlib
import io
from types import ModuleType

import pytest
import torch


def _frechet() -> ModuleType:
    return importlib.import_module("tardis.metrics.frechet")


def test_identical_gaussians_have_zero_frechet_distance() -> None:
    mean = torch.tensor([1.5, -2.0], dtype=torch.float64)
    covariance = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    distance = _frechet().frechet_distance(mean, covariance, mean, covariance)

    assert distance == pytest.approx(0.0, abs=1.0e-10)


def test_covariance_square_root_is_stable_for_singular_and_roundoff_eigenvalues() -> None:
    covariance = torch.tensor([[4.0, 0.0], [0.0, -1.0e-14]], dtype=torch.float64)

    root = _frechet().symmetric_matrix_square_root(covariance)

    assert root.dtype == torch.float64
    assert torch.isfinite(root).all()
    assert torch.allclose(root, root.T)
    assert torch.allclose(root, torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float64))


def test_float64_sufficient_statistics_merge_exactly_without_retaining_samples() -> None:
    frechet = _frechet()
    features = torch.tensor([[1.0, 2.0], [3.0, 5.0], [-2.0, 4.0]], dtype=torch.float32)
    direct = frechet.OnlineFeatureStats(feature_dim=2)
    left = frechet.OnlineFeatureStats(feature_dim=2)
    right = frechet.OnlineFeatureStats(feature_dim=2)

    direct.update(features)
    left.update(features[:2])
    right.update(features[2:])
    left.merge(right)

    assert direct.count.dtype == torch.float64
    assert direct.sum.dtype == torch.float64
    assert direct.cross_product.dtype == torch.float64
    assert direct.count.item() == 3
    assert torch.equal(direct.sum, features.to(torch.float64).sum(dim=0))
    assert torch.equal(
        direct.cross_product,
        features.to(torch.float64).T @ features.to(torch.float64),
    )
    assert torch.equal(left.count, direct.count)
    assert torch.equal(left.sum, direct.sum)
    assert torch.equal(left.cross_product, direct.cross_product)
    assert not hasattr(direct, "samples")


def test_sufficient_statistics_state_is_checkpoint_serializable_and_reducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frechet = _frechet()
    stats = frechet.OnlineFeatureStats(feature_dim=2)
    stats.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    buffer = io.BytesIO()
    torch.save(stats.state_dict(), buffer)
    buffer.seek(0)
    restored = frechet.OnlineFeatureStats(feature_dim=2)
    restored.load_state_dict(torch.load(buffer, weights_only=True))

    assert torch.equal(restored.reduction_tensor(), stats.reduction_tensor())

    monkeypatch.setattr(frechet, "_distributed_sum", lambda value: value * 2)
    restored.all_reduce()
    assert restored.count.item() == 4
    assert torch.equal(restored.sum, stats.sum * 2)
    assert torch.equal(restored.cross_product, stats.cross_product * 2)


def test_online_statistics_reconstruct_known_mean_and_unbiased_covariance() -> None:
    stats = _frechet().OnlineFeatureStats(feature_dim=2)
    stats.update(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    assert torch.equal(stats.mean, torch.tensor([3.0, 4.0], dtype=torch.float64))
    assert torch.allclose(
        stats.covariance,
        torch.tensor([[4.0, 4.0], [4.0, 4.0]], dtype=torch.float64),
    )


class _FrameFeatures:
    provenance_id = "unit-test/frame-features"
    feature_dim = 2

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        means = video.mean(dim=(1, 2, 3))
        return torch.stack((means, means.square()), dim=1)


def test_fid_uses_injected_features_and_round_trips_distribution_state() -> None:
    frechet = _frechet()
    metric = frechet.FIDMetric(_FrameFeatures())
    video = torch.stack([torch.full((3, 2, 2), value) for value in (-0.5, 0.0, 0.75)])
    metric.update(video, video.clone())

    assert metric.compute() == pytest.approx(0.0, abs=1.0e-10)
    assert metric.provenance_id == "unit-test/frame-features"

    restored = frechet.FIDMetric(_FrameFeatures())
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == pytest.approx(metric.compute())
