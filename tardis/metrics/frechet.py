"""Online sufficient statistics and stable Frechet distribution metrics."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, isqrt
from typing import Protocol, Self, cast

import torch

from tardis.metrics.base import _distributed_sum, validate_video_pair


class VideoFeature(Protocol):
    """A frozen video-to-feature adapter with a fixed output dimension."""

    provenance_id: str
    feature_dim: int

    def __call__(self, video: torch.Tensor) -> torch.Tensor: ...


class OnlineFeatureStats:
    """Float64 count, sum, and cross-product sufficient statistics."""

    def __init__(self, feature_dim: int | None = None) -> None:
        if feature_dim is not None and feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        dimension = feature_dim or 0
        self.count = torch.zeros((), dtype=torch.float64)
        self.sum = torch.zeros(dimension, dtype=torch.float64)
        self.cross_product = torch.zeros(dimension, dimension, dtype=torch.float64)

    @property
    def feature_dim(self) -> int | None:
        return self.sum.numel() or None

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must have shape [N,D] with positive dimensions")
        values = features.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("features must be finite")
        self._ensure_dimension(values.shape[1])
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.cross_product += values.T @ values

    @property
    def mean(self) -> torch.Tensor:
        count = float(self.count.item())
        if count <= 0:
            raise RuntimeError("feature statistics have no observations")
        return self.sum / count

    @property
    def covariance(self) -> torch.Tensor:
        count = float(self.count.item())
        if count < 2:
            raise RuntimeError("at least two features are required for covariance")
        centered = self.cross_product - torch.outer(self.sum, self.sum) / count
        covariance = centered / (count - 1)
        return (covariance + covariance.T) * 0.5

    def merge(self, other: Self) -> None:
        other_dimension = other.feature_dim
        if other_dimension is None:
            return
        self._ensure_dimension(other_dimension)
        self.count += other.count
        self.sum += other.sum
        self.cross_product += other.cross_product

    def reset(self) -> None:
        self.count.zero_()
        self.sum.zero_()
        self.cross_product.zero_()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "count": self.count.clone(),
            "sum": self.sum.clone(),
            "cross_product": self.cross_product.clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"count", "sum", "cross_product"}:
            raise ValueError("feature statistics state has an incompatible schema")
        count = state["count"]
        total = state["sum"]
        cross_product = state["cross_product"]
        if not all(isinstance(value, torch.Tensor) for value in (count, total, cross_product)):
            raise ValueError("feature statistics state values must be tensors")
        count = cast(torch.Tensor, count).detach().to(device="cpu", dtype=torch.float64)
        total = cast(torch.Tensor, total).detach().to(device="cpu", dtype=torch.float64)
        cross_product = (
            cast(torch.Tensor, cross_product).detach().to(device="cpu", dtype=torch.float64)
        )
        if count.ndim != 0 or total.ndim != 1 or cross_product.shape != (total.numel(),) * 2:
            raise ValueError("feature statistics state tensor shapes are incompatible")
        count_value = float(count.item())
        if (
            not isfinite(count_value)
            or count_value < 0
            or not count_value.is_integer()
            or not bool(torch.isfinite(total).all().item())
            or not bool(torch.isfinite(cross_product).all().item())
        ):
            raise ValueError("feature statistics state contains invalid values")
        self._ensure_dimension(total.numel())
        self.count.copy_(count)
        self.sum.copy_(total)
        self.cross_product.copy_(cross_product)

    def reduction_tensor(self) -> torch.Tensor:
        if self.feature_dim is None:
            raise RuntimeError("feature dimension is unknown")
        return torch.cat((self.count.reshape(1), self.sum, self.cross_product.reshape(-1)))

    def load_reduction_tensor(self, reduction: torch.Tensor) -> None:
        values = reduction.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        dimension = self.feature_dim or _dimension_from_reduction_size(values.numel())
        expected = 1 + dimension + dimension * dimension
        if values.numel() != expected or not bool(torch.isfinite(values).all().item()):
            raise ValueError("feature reduction tensor has an incompatible shape or values")
        self._ensure_dimension(dimension)
        self.count.copy_(values[0])
        self.sum.copy_(values[1 : 1 + dimension])
        self.cross_product.copy_(values[1 + dimension :].reshape(dimension, dimension))

    def all_reduce(self) -> None:
        self.load_reduction_tensor(_distributed_sum(self.reduction_tensor()))

    def _ensure_dimension(self, dimension: int) -> None:
        current = self.feature_dim
        if current is None:
            if dimension <= 0:
                raise ValueError("feature dimension must be positive")
            self.sum = torch.zeros(dimension, dtype=torch.float64)
            self.cross_product = torch.zeros(dimension, dimension, dtype=torch.float64)
        elif current != dimension:
            raise ValueError(f"feature dimension changed from {current} to {dimension}")


def symmetric_matrix_square_root(matrix: torch.Tensor) -> torch.Tensor:
    """Return a stable real square root of a numerically PSD matrix."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("matrix square root requires a non-empty square matrix")
    values = matrix.detach().to(torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("matrix square root requires finite values")
    symmetric = (values + values.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    root = (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T
    return cast(torch.Tensor, (root + root.T) * 0.5)


def frechet_distance(
    first_mean: torch.Tensor,
    first_covariance: torch.Tensor,
    second_mean: torch.Tensor,
    second_covariance: torch.Tensor,
) -> float:
    """Compute the Gaussian Frechet distance through a symmetric PSD product."""

    first_mean = first_mean.detach().to(torch.float64)
    second_mean = second_mean.detach().to(torch.float64)
    first_covariance = first_covariance.detach().to(torch.float64)
    second_covariance = second_covariance.detach().to(torch.float64)
    dimension = first_mean.numel()
    if (
        first_mean.ndim != 1
        or second_mean.shape != first_mean.shape
        or first_covariance.shape != (dimension, dimension)
        or second_covariance.shape != (dimension, dimension)
    ):
        raise ValueError("Frechet means and covariances have incompatible shapes")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (first_mean, second_mean, first_covariance, second_covariance)
    ):
        raise ValueError("Frechet means and covariances must be finite")

    first_root = symmetric_matrix_square_root(first_covariance)
    covariance_product = first_root @ second_covariance @ first_root
    product_root = symmetric_matrix_square_root(covariance_product)
    mean_term = (first_mean - second_mean).square().sum()
    covariance_term = torch.trace(first_covariance + second_covariance - 2 * product_root)
    return max(float((mean_term + covariance_term).item()), 0.0)


class FrechetMetric:
    """Streaming generated/reference feature distributions for one extractor."""

    def __init__(self, feature: VideoFeature) -> None:
        if feature.feature_dim <= 0 or not feature.provenance_id.strip():
            raise ValueError("feature extractor dimension and provenance must be valid")
        self.feature = feature
        self.provenance_id = feature.provenance_id
        self.generated = OnlineFeatureStats(feature.feature_dim)
        self.reference = OnlineFeatureStats(feature.feature_dim)

    @torch.no_grad()
    def update(self, generated: torch.Tensor, reference: torch.Tensor) -> None:
        validate_video_pair(generated, reference)
        generated_features = self._extract(generated)
        reference_features = self._extract(reference)
        self.generated.update(generated_features)
        self.reference.update(reference_features)

    def compute(self) -> float:
        return frechet_distance(
            self.generated.mean,
            self.generated.covariance,
            self.reference.mean,
            self.reference.covariance,
        )

    def merge(self, other: Self) -> None:
        if type(self) is not type(other) or self.provenance_id != other.provenance_id:
            raise ValueError("only matching Frechet metrics can be merged")
        self.generated.merge(other.generated)
        self.reference.merge(other.reference)

    def reset(self) -> None:
        self.generated.reset()
        self.reference.reset()

    def state_dict(self) -> dict[str, object]:
        return {
            "metric_type": type(self).__name__,
            "provenance_id": self.provenance_id,
            "generated": self.generated.state_dict(),
            "reference": self.reference.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"metric_type", "provenance_id", "generated", "reference"}
        if set(state) != required:
            raise ValueError("Frechet metric state has an incompatible schema")
        if (
            state["metric_type"] != type(self).__name__
            or state["provenance_id"] != self.provenance_id
        ):
            raise ValueError("Frechet metric state provenance does not match")
        generated = state["generated"]
        reference = state["reference"]
        if not isinstance(generated, Mapping) or not isinstance(reference, Mapping):
            raise ValueError("Frechet metric statistics state must be mappings")
        self.generated.load_state_dict(cast(Mapping[str, object], generated))
        self.reference.load_state_dict(cast(Mapping[str, object], reference))

    def all_reduce(self) -> None:
        self.generated.all_reduce()
        self.reference.all_reduce()

    def _extract(self, video: torch.Tensor) -> torch.Tensor:
        features = self.feature(video)
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != self.feature.feature_dim
            or not bool(torch.isfinite(features).all().item())
        ):
            raise ValueError(
                f"feature extractor must return finite [N,{self.feature.feature_dim}] values"
            )
        return features


class FIDMetric(FrechetMetric):
    def __init__(self, feature: VideoFeature | None = None) -> None:
        if feature is None:
            from tardis.metrics.features import InceptionV3PoolFeatures

            feature = InceptionV3PoolFeatures()
        super().__init__(feature)


class FVDMetric(FrechetMetric):
    def __init__(self, feature: VideoFeature | None = None) -> None:
        if feature is None:
            from tardis.metrics.features import I3DKineticsFeatures

            feature = I3DKineticsFeatures()
        super().__init__(feature)


FeatureStats = OnlineFeatureStats
matrix_square_root = symmetric_matrix_square_root


def _dimension_from_reduction_size(size: int) -> int:
    discriminant = 4 * size - 3
    root = isqrt(discriminant)
    if root * root != discriminant or (root - 1) % 2:
        raise ValueError("cannot infer feature dimension from reduction tensor")
    dimension = (root - 1) // 2
    if dimension <= 0 or 1 + dimension + dimension * dimension != size:
        raise ValueError("cannot infer feature dimension from reduction tensor")
    return dimension
