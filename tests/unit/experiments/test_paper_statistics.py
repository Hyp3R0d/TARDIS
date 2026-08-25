from __future__ import annotations

import pytest

from tardis.experiments.statistics import (
    align_metric_records,
    compare_lower_is_better,
    holm_adjust,
)


def test_align_metric_records_uses_record_id_not_file_order() -> None:
    ours = [
        {"record_id": "a", "tc": 0.1},
        {"record_id": "b", "tc": 0.2},
    ]
    benchmark = [
        {"record_id": "b", "tc": 0.4},
        {"record_id": "a", "tc": 0.3},
    ]

    ours_values, benchmark_values = align_metric_records(ours, benchmark, "tc")

    assert ours_values.tolist() == pytest.approx([0.1, 0.2])
    assert benchmark_values.tolist() == pytest.approx([0.3, 0.4])


def test_align_metric_records_rejects_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="coverage"):
        align_metric_records(
            [{"record_id": "a", "tc": 0.1}],
            [{"record_id": "b", "tc": 0.2}],
            "tc",
        )


def test_paired_comparison_reports_positive_improvement_for_lower_ours() -> None:
    result = compare_lower_is_better(
        [0.10, 0.12, 0.11, 0.09, 0.10],
        [0.20, 0.22, 0.21, 0.19, 0.20],
        bootstrap_samples=2_000,
        seed=7,
    )

    assert result.ours_mean == pytest.approx(0.104)
    assert result.benchmark_mean == pytest.approx(0.204)
    assert result.absolute_improvement == pytest.approx(0.1)
    assert result.relative_improvement_percent > 0
    assert result.ci_low > 0
    assert result.win_rate == pytest.approx(1.0)


def test_holm_adjust_is_monotonic_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])

