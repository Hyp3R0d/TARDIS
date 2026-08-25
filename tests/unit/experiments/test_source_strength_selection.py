from __future__ import annotations

import pytest

from tardis.experiments.source_strength_selection import select_candidate, weighted_score


def test_weighted_score_uses_locked_primary_metric_weights() -> None:
    score = weighted_score(tc=0.05, lpips=0.30)

    assert score == pytest.approx(0.5)


def test_select_candidate_uses_score_then_primary_metric_tiebreakers() -> None:
    selected = select_candidate(
        [
            {"source_strength": 0.30, "tc": 0.020, "lpips": 0.175},
            {"source_strength": 0.35, "tc": 0.018, "lpips": 0.200},
            {"source_strength": 0.40, "tc": 0.030, "lpips": 0.190},
        ]
    )

    assert selected["source_strength"] == pytest.approx(0.30)
