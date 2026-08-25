"""Validation aggregation and metric-aligned checkpoint selection.

Checkpoint selection has one intentionally strict order:

1. accept exactly one selected dataset's validation metrics;
2. mark a checkpoint as passing only when TC and LPIPS both meet their locked targets;
3. prioritize passing checkpoints, then rank checkpoints only by the TC/LPIPS score.

The test split is not accepted by this module. Test metrics belong to the final ``infer``
report and can never influence ``best.pt``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final, cast


class ValidationMetric(StrEnum):
    TC = "tc"
    LPIPS = "lpips"
    FVD = "fvd"
    FID = "fid"
    CLIPSCORE = "clipscore"
    SSIM = "ssim"


METRIC_WEIGHTS: Final[dict[ValidationMetric, float]] = {
    ValidationMetric.TC: 0.625,
    ValidationMetric.LPIPS: 0.375,
    ValidationMetric.FVD: 0.0,
    ValidationMetric.FID: 0.0,
    ValidationMetric.CLIPSCORE: 0.0,
    ValidationMetric.SSIM: 0.0,
}

METRIC_SCALES: Final[dict[ValidationMetric, float]] = {
    # DataVerse is the default CLI dataset. Train constructs source-specific scales
    # below, while the four zero-weight metrics retain stable diagnostic scales.
    ValidationMetric.TC: 0.06,
    ValidationMetric.LPIPS: 0.60,
    ValidationMetric.FVD: 100.0,
    ValidationMetric.FID: 50.0,
    ValidationMetric.CLIPSCORE: 0.50,
    ValidationMetric.SSIM: 0.50,
}
# Kept as a compatibility alias for checkpoint/test code written before scales were
# frozen. These values are targets, not measured external baselines.
METRIC_BASELINES: Final[dict[ValidationMetric, float]] = METRIC_SCALES

LOWER_IS_BETTER: Final[frozenset[ValidationMetric]] = frozenset(
    {
        ValidationMetric.TC,
        ValidationMetric.LPIPS,
        ValidationMetric.FVD,
        ValidationMetric.FID,
    }
)
HIGHER_IS_BETTER: Final[frozenset[ValidationMetric]] = frozenset(
    {ValidationMetric.CLIPSCORE, ValidationMetric.SSIM}
)
VALIDATION_SOURCES: Final[tuple[str, str, str]] = (
    "dataverse_validation",
    "openvid_validation",
    "seedance_validation",
)
SELECTION_SCALES_BY_SOURCE: Final[dict[str, dict[ValidationMetric, float]]] = {
    "dataverse_validation": {
        ValidationMetric.TC: 0.06,
        ValidationMetric.LPIPS: 0.60,
    },
    "openvid_validation": {
        ValidationMetric.TC: 0.07,
        ValidationMetric.LPIPS: 0.60,
    },
    "seedance_validation": {
        ValidationMetric.TC: 0.10,
        ValidationMetric.LPIPS: 0.60,
    },
}


@dataclass(frozen=True, slots=True)
class ValidationScore:
    """A checkpoint-selectable score with source macro metrics retained."""

    source_metrics: dict[str, dict[str, float]]
    average_metrics: dict[str, float]
    normalized_metrics: dict[str, float]
    composite: float

    @property
    def target_pass(self) -> bool:
        """Whether both competition metrics meet their dataset-specific targets."""

        return all(
            self.normalized_metrics.get(metric.value, float("inf")) <= 1.0
            for metric in (ValidationMetric.TC, ValidationMetric.LPIPS)
        )


def aggregate_validation_metrics(
    source_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Validate and return all six metrics for one selected validation source.

    Checkpoint selection is intentionally dataset-local. Test rows and multi-dataset
    validation events are rejected so one run can never select a checkpoint using
    another dataset's evidence.
    """

    if any("test" in source.lower() for source in source_metrics):
        raise ValueError("checkpoint selection accepts validation metrics, not test metrics")
    if len(source_metrics) != 1:
        raise ValueError("validation metrics must contain exactly one selected dataset")
    source, metrics = next(iter(source_metrics.items()))
    if source not in VALIDATION_SOURCES:
        expected = ", ".join(VALIDATION_SOURCES)
        raise ValueError(f"validation source must be one of: {expected}")
    return _validate_metric_mapping(metrics)


def normalize_validation_metric(
    metric: ValidationMetric | str,
    value: float,
    *,
    baseline: float | None = None,
) -> float:
    """Map one macro metric to a dimensionless target-relative cost.

    A target-scale result maps to one. Lower costs are always better. TC and LPIPS
    therefore follow the exact locked selection formula; the remaining zero-weight
    diagnostic metrics use the same direction-normalized convention.
    """

    selected = ValidationMetric(metric)
    reference = METRIC_BASELINES[selected] if baseline is None else baseline
    if not isfinite(value) or not isfinite(reference) or reference <= 0:
        raise ValueError("metric value and positive baseline must be finite")
    bounded_value = max(value, 0.0)
    if selected in LOWER_IS_BETTER:
        return bounded_value / reference
    if selected in HIGHER_IS_BETTER:
        return reference / max(bounded_value, 1.0e-12)
    raise ValueError(f"metric has no direction: {selected.value}")


def composite_validation_score(
    source_metrics: Mapping[str, Mapping[str, float]],
    *,
    baselines: Mapping[ValidationMetric | str, float] | None = None,
) -> float:
    """Return the selected dataset's TC/LPIPS competition score.

    The official objective allocates 50% to TC and 30% to LPIPS. After excluding the
    subjective 20%, those two machine metrics are renormalized to 0.625 and 0.375.
    """

    average = aggregate_validation_metrics(source_metrics)
    selected_baselines = cast(
        Mapping[ValidationMetric | str, float],
        _event_scales(source_metrics) if baselines is None else baselines,
    )
    normalized = _normalized_metrics(average, baselines=selected_baselines)
    return sum(METRIC_WEIGHTS[metric] * normalized[metric.value] for metric in ValidationMetric)


def score_validation_event(
    source_metrics: Mapping[str, Mapping[str, float]],
    *,
    baselines: Mapping[ValidationMetric | str, float] | None = None,
) -> ValidationScore:
    """Materialize all checkpoint-selection evidence for one validation event."""

    average = aggregate_validation_metrics(source_metrics)
    selected_baselines = cast(
        Mapping[ValidationMetric | str, float],
        _event_scales(source_metrics) if baselines is None else baselines,
    )
    normalized = _normalized_metrics(average, baselines=selected_baselines)
    canonical_sources = {
        source: _validate_metric_mapping(metrics) for source, metrics in source_metrics.items()
    }
    composite = sum(
        METRIC_WEIGHTS[metric] * normalized[metric.value] for metric in ValidationMetric
    )
    return ValidationScore(canonical_sources, average, normalized, composite)


def should_replace_best(
    candidate: ValidationScore,
    incumbent: ValidationScore | None,
    *,
    tolerance: float = 0.0,
    pareto_tolerance: float = 0.0,
) -> bool:
    """Prioritize target-passing checkpoints, then compare TC/LPIPS weighted cost.

    ``pareto_tolerance`` remains in the API and serialized selector state so existing
    checkpoints can resume exactly, but Pareto behavior is no longer an acceptance or
    replacement gate under the target-only protocol.
    """

    if tolerance < 0 or pareto_tolerance < 0:
        raise ValueError("selector tolerances must be non-negative")
    if incumbent is None:
        return True
    if candidate.target_pass != incumbent.target_pass:
        return candidate.target_pass
    return candidate.composite < incumbent.composite - tolerance


class ValidationCheckpointSelector:
    """Keep the best checkpoint evidence from validation events only."""

    def __init__(
        self,
        *,
        baselines: Mapping[ValidationMetric | str, float] | None = None,
        tolerance: float = 0.0,
        pareto_tolerance: float = 0.0,
    ) -> None:
        if tolerance < 0 or pareto_tolerance < 0:
            raise ValueError("selector tolerances must be non-negative")
        self.baselines: Mapping[ValidationMetric | str, float] = cast(
            Mapping[ValidationMetric | str, float],
            dict(METRIC_BASELINES) if baselines is None else _canonical_baselines(baselines),
        )
        self.tolerance = tolerance
        self.pareto_tolerance = pareto_tolerance
        self.best_score: ValidationScore | None = None
        self.best_epoch: int | None = None

    def update(
        self,
        source_metrics: Mapping[str, Mapping[str, float]],
        *,
        epoch: int,
    ) -> bool:
        """Evaluate one dataset-local validation event and report whether it is a new best."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        candidate = score_validation_event(source_metrics, baselines=self.baselines)
        if not should_replace_best(
            candidate,
            self.best_score,
            tolerance=self.tolerance,
            pareto_tolerance=self.pareto_tolerance,
        ):
            return False
        self.best_score = candidate
        self.best_epoch = epoch
        return True


def _normalized_metrics(
    average: Mapping[str, float],
    *,
    baselines: Mapping[ValidationMetric | str, float] | None,
) -> dict[str, float]:
    selected_baselines = METRIC_BASELINES if baselines is None else _canonical_baselines(baselines)
    return {
        metric.value: normalize_validation_metric(
            metric,
            average[metric.value],
            baseline=selected_baselines[metric],
        )
        for metric in ValidationMetric
    }


def _canonical_baselines(
    baselines: Mapping[ValidationMetric | str, float],
) -> dict[ValidationMetric, float]:
    result = {ValidationMetric(metric): value for metric, value in baselines.items()}
    if set(result) != set(ValidationMetric):
        raise ValueError("baselines must define all six validation metrics")
    return result


def selection_scales_for_source(source: str) -> dict[ValidationMetric, float]:
    """Return all six frozen scales for one canonical validation source."""

    if source not in SELECTION_SCALES_BY_SOURCE:
        expected = ", ".join(VALIDATION_SOURCES)
        raise ValueError(f"validation source must be one of: {expected}")
    scales = dict(METRIC_SCALES)
    scales.update(SELECTION_SCALES_BY_SOURCE[source])
    return scales


def _event_scales(
    source_metrics: Mapping[str, Mapping[str, float]],
) -> dict[ValidationMetric, float]:
    source = next(iter(source_metrics))
    return selection_scales_for_source(source)


def _validate_metric_mapping(metrics: Mapping[str, float]) -> dict[str, float]:
    expected = {metric.value for metric in ValidationMetric}
    if set(metrics) != expected:
        missing = sorted(expected - set(metrics))
        extra = sorted(set(metrics) - expected)
        raise ValueError(
            f"validation metrics must contain six values; missing={missing}, extra={extra}"
        )
    result = {metric: float(metrics[metric]) for metric in expected}
    if any(not isfinite(value) for value in result.values()):
        raise ValueError("validation metrics must be finite")
    return result
