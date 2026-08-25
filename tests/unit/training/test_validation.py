from __future__ import annotations

import math

import pytest

from tardis.training.validation import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRIC_BASELINES,
    METRIC_WEIGHTS,
    SELECTION_SCALES_BY_SOURCE,
    VALIDATION_SOURCES,
    ValidationCheckpointSelector,
    ValidationMetric,
    ValidationScore,
    aggregate_validation_metrics,
    composite_validation_score,
    normalize_validation_metric,
    score_validation_event,
    should_replace_best,
)


def test_checkpoint_score_uses_exactly_one_selected_validation_set() -> None:
    validation = {
        "dataverse_validation": {
            "tc": 0.10,
            "lpips": 0.20,
            "fvd": 10.0,
            "fid": 20.0,
            "clipscore": 0.70,
            "ssim": 0.80,
        },
    }
    averaged = aggregate_validation_metrics(validation)

    assert averaged == {
        "tc": pytest.approx(0.10),
        "lpips": pytest.approx(0.20),
        "fvd": pytest.approx(10.0),
        "fid": pytest.approx(20.0),
        "clipscore": pytest.approx(0.70),
        "ssim": pytest.approx(0.80),
    }
    assert composite_validation_score(validation) > 0

    with pytest.raises(ValueError, match="exactly one"):
        aggregate_validation_metrics(
            {
                **validation,
                "openvid_validation": dict(validation["dataverse_validation"]),
            }
        )


def test_checkpoint_score_rejects_average_only_rows_without_source_provenance() -> None:
    average = {metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric}

    with pytest.raises(ValueError, match="validation"):
        aggregate_validation_metrics({"average": average})
    with pytest.raises(ValueError, match="validation"):
        composite_validation_score({"average": average})


def test_validation_weights_match_competition_priorities_and_sum_to_one() -> None:
    assert METRIC_WEIGHTS == {
        ValidationMetric.TC: 0.625,
        ValidationMetric.LPIPS: 0.375,
        ValidationMetric.FVD: 0.0,
        ValidationMetric.FID: 0.0,
        ValidationMetric.CLIPSCORE: 0.0,
        ValidationMetric.SSIM: 0.0,
    }
    assert sum(METRIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_validation_sources_are_the_three_canonical_validation_rows() -> None:
    assert VALIDATION_SOURCES == (
        "dataverse_validation",
        "openvid_validation",
        "seedance_validation",
    )


def test_each_frozen_baseline_normalizes_to_one_and_composite_is_one() -> None:
    baseline = {
        "dataverse_validation": {
            metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric
        }
    }

    assert all(
        normalize_validation_metric(metric, METRIC_BASELINES[metric]) == pytest.approx(1.0)
        for metric in ValidationMetric
    )
    assert composite_validation_score(baseline) == pytest.approx(1.0)


def test_normalization_is_bounded_monotonic_and_preserves_improvement_resolution() -> None:
    for metric in LOWER_IS_BETTER:
        baseline = METRIC_BASELINES[metric]
        values = [0.0, baseline / 4, baseline / 2, baseline, 2 * baseline]
        scores = [normalize_validation_metric(metric, value) for value in values]
        assert scores == sorted(scores)
        assert scores[1] < scores[2] < scores[3] < scores[4]

    for metric in HIGHER_IS_BETTER:
        baseline = METRIC_BASELINES[metric]
        values = [baseline / 2, baseline, 2 * baseline]
        scores = [normalize_validation_metric(metric, value) for value in values]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[1] > scores[2]


def test_dataset_selection_scales_match_the_locked_protocol() -> None:
    assert SELECTION_SCALES_BY_SOURCE["dataverse_validation"] == {
        ValidationMetric.TC: pytest.approx(0.06),
        ValidationMetric.LPIPS: pytest.approx(0.60),
    }
    assert SELECTION_SCALES_BY_SOURCE["openvid_validation"] == {
        ValidationMetric.TC: pytest.approx(0.07),
        ValidationMetric.LPIPS: pytest.approx(0.60),
    }
    assert SELECTION_SCALES_BY_SOURCE["seedance_validation"] == {
        ValidationMetric.TC: pytest.approx(0.10),
        ValidationMetric.LPIPS: pytest.approx(0.60),
    }
    assert METRIC_BASELINES[ValidationMetric.LPIPS] == pytest.approx(0.60)


def test_checkpoint_selector_rejects_test_split_metrics() -> None:
    test_metrics = {
        "dataverse_test": {metric.value: 1.0 for metric in ValidationMetric},
    }
    with pytest.raises(ValueError, match="validation"):
        aggregate_validation_metrics(test_metrics)
    with pytest.raises(ValueError, match="validation"):
        ValidationCheckpointSelector().update(test_metrics, epoch=1)


def test_selector_replaces_only_on_a_strictly_better_validation_score() -> None:
    baseline = {
        "dataverse_validation": {
            metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric
        }
    }
    selector = ValidationCheckpointSelector()

    assert selector.update(baseline, epoch=1)
    assert selector.best_epoch == 1
    assert selector.best_score is not None
    assert selector.best_score.composite == pytest.approx(1.0)
    assert not selector.update(baseline, epoch=2)
    assert selector.best_epoch == 1

    improved = {source: dict(metrics) for source, metrics in baseline.items()}
    for metrics in improved.values():
        metrics[ValidationMetric.TC.value] = METRIC_BASELINES[ValidationMetric.TC] / 2
    assert selector.update(improved, epoch=3)
    assert selector.best_epoch == 3
    assert selector.best_score is not None
    assert selector.best_score.composite < 1.0


def test_strict_tie_rule_rejects_only_equal_scores() -> None:
    metrics = {"tc": 0.06, "lpips": 0.60}
    incumbent = ValidationScore({}, metrics, {}, 1.0)
    tied = ValidationScore({}, metrics, {}, 1.0)
    minimally_better = ValidationScore(
        {},
        {"tc": math.nextafter(0.06, 0.0), "lpips": 0.60},
        {},
        math.nextafter(1.0, 0.0),
    )

    assert not should_replace_best(tied, incumbent)
    assert should_replace_best(minimally_better, incumbent)


def test_target_passing_candidate_replaces_lower_scoring_non_passing_incumbent() -> None:
    incumbent = score_validation_event(
        {
            "dataverse_validation": {
                "tc": 0.01,
                "lpips": 0.61,
                "fvd": 1.0,
                "fid": 1.0,
                "clipscore": 1.0,
                "ssim": 1.0,
            }
        }
    )
    candidate = score_validation_event(
        {
            "dataverse_validation": {
                "tc": 0.06,
                "lpips": 0.60,
                "fvd": 1_000_000.0,
                "fid": 1_000_000.0,
                "clipscore": 0.0,
                "ssim": 0.0,
            }
        }
    )

    assert incumbent.composite < candidate.composite
    assert not incumbent.target_pass
    assert candidate.target_pass
    assert should_replace_best(candidate, incumbent)


def test_non_passing_candidate_never_replaces_target_passing_incumbent() -> None:
    incumbent = score_validation_event(
        {
            "dataverse_validation": {
                "tc": 0.06,
                "lpips": 0.60,
                "fvd": 1.0,
                "fid": 1.0,
                "clipscore": 1.0,
                "ssim": 1.0,
            }
        }
    )
    candidate = score_validation_event(
        {
            "dataverse_validation": {
                "tc": 0.01,
                "lpips": 0.61,
                "fvd": 0.0,
                "fid": 0.0,
                "clipscore": 1.0,
                "ssim": 1.0,
            }
        }
    )

    assert candidate.composite < incumbent.composite
    assert incumbent.target_pass
    assert not candidate.target_pass
    assert not should_replace_best(candidate, incumbent)


def test_non_passing_candidates_use_tc_lpips_weighted_score_without_pareto_gate() -> None:
    incumbent = ValidationScore({}, {"tc": 0.08, "lpips": 0.80}, {}, 1.833333)
    candidate = ValidationScore({}, {"tc": 0.04, "lpips": 0.90}, {}, 1.541667)

    assert should_replace_best(candidate, incumbent)
