"""Optimization, curriculum, validation, and checkpoint-selection utilities."""

from tardis.training.validation import (
    METRIC_BASELINES,
    METRIC_SCALES,
    METRIC_WEIGHTS,
    SELECTION_SCALES_BY_SOURCE,
    ValidationCheckpointSelector,
    ValidationMetric,
    aggregate_validation_metrics,
    composite_validation_score,
    score_validation_event,
    selection_scales_for_source,
)

__all__ = [
    "METRIC_BASELINES",
    "METRIC_SCALES",
    "METRIC_WEIGHTS",
    "SELECTION_SCALES_BY_SOURCE",
    "ValidationCheckpointSelector",
    "ValidationMetric",
    "aggregate_validation_metrics",
    "composite_validation_score",
    "score_validation_event",
    "selection_scales_for_source",
]
