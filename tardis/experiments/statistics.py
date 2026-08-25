"""Paired uncertainty and significance analysis for lower-is-better video metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True, slots=True)
class PairedComparison:
    sample_count: int
    ours_mean: float
    benchmark_mean: float
    absolute_improvement: float
    relative_improvement_percent: float
    ci_low: float
    ci_high: float
    win_rate: float
    tie_rate: float
    wilcoxon_statistic: float
    p_value_one_sided: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def align_metric_records(
    ours: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align complete per-video results by record ID and reject hidden exclusions."""

    ours_by_id = _metric_by_id(ours, metric)
    benchmark_by_id = _metric_by_id(benchmark, metric)
    if set(ours_by_id) != set(benchmark_by_id):
        missing = sorted(set(ours_by_id) - set(benchmark_by_id))
        extra = sorted(set(benchmark_by_id) - set(ours_by_id))
        raise ValueError(f"paired result coverage mismatch; missing={missing}, extra={extra}")
    ordered_ids = sorted(ours_by_id)
    return (
        np.asarray([ours_by_id[record_id] for record_id in ordered_ids], dtype=np.float64),
        np.asarray([benchmark_by_id[record_id] for record_id in ordered_ids], dtype=np.float64),
    )


def compare_lower_is_better(
    ours: list[float] | np.ndarray,
    benchmark: list[float] | np.ndarray,
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> PairedComparison:
    """Compare paired observations using bootstrap CIs and one-sided Wilcoxon."""

    ours_values = np.asarray(ours, dtype=np.float64)
    benchmark_values = np.asarray(benchmark, dtype=np.float64)
    if (
        ours_values.ndim != 1
        or benchmark_values.shape != ours_values.shape
        or ours_values.size < 2
        or not np.isfinite(ours_values).all()
        or not np.isfinite(benchmark_values).all()
    ):
        raise ValueError("paired values must be finite equal-length vectors with at least 2 items")
    if bootstrap_samples <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap_samples and confidence are invalid")

    difference = benchmark_values - ours_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(bootstrap_samples, difference.size))
    bootstrap_means = difference[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1 - tail])
    if np.allclose(difference, 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            ours_values,
            benchmark_values,
            alternative="less",
            zero_method="pratt",
            method="auto",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    ours_mean = float(ours_values.mean())
    benchmark_mean = float(benchmark_values.mean())
    absolute = benchmark_mean - ours_mean
    relative = 100 * absolute / benchmark_mean if benchmark_mean else float("nan")
    return PairedComparison(
        sample_count=int(difference.size),
        ours_mean=ours_mean,
        benchmark_mean=benchmark_mean,
        absolute_improvement=absolute,
        relative_improvement_percent=float(relative),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        win_rate=float(np.mean(difference > 0)),
        tie_rate=float(np.mean(difference == 0)),
        wilcoxon_statistic=statistic,
        p_value_one_sided=p_value,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in the original order."""

    if not p_values or any(not isfinite(value) or not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be a non-empty finite list in [0, 1]")
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _metric_by_id(records: list[dict[str, Any]], metric: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in records:
        record_id = str(item.get("record_id", ""))
        if not record_id or record_id in result:
            raise ValueError("per-video results contain empty or duplicate record IDs")
        try:
            value = float(item[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"per-video result is missing finite metric {metric!r}") from error
        if not isfinite(value):
            raise ValueError(f"per-video metric {metric!r} must be finite")
        result[record_id] = value
    return result


__all__ = [
    "PairedComparison",
    "align_metric_records",
    "compare_lower_is_better",
    "holm_adjust",
]

