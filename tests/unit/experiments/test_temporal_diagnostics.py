from __future__ import annotations

import pytest
import torch

from tardis.experiments.temporal_diagnostics import temporal_diagnostic_details


class MeanAbsoluteDistance:
    provenance_id = "test/mean-absolute-distance"

    def __call__(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first - second).abs().mean(dim=(1, 2, 3), keepdim=True)


def test_temporal_diagnostics_are_zero_for_static_identical_video() -> None:
    video = torch.zeros(4, 3, 8, 8)
    backward_flow = torch.zeros(3, 2, 8, 8)

    result = temporal_diagnostic_details(
        video,
        video,
        backward_flow,
        MeanAbsoluteDistance(),
        lag_values=(1, 2),
    )

    assert result["flow_warp_error"] == pytest.approx(0.0)
    assert result["tlpips"] == pytest.approx(0.0)
    assert result["flicker_rate"] == pytest.approx(0.0)
    assert result["drift_slope"] == pytest.approx(0.0)
    assert result["motion_magnitude"] == pytest.approx(0.0)
    assert result["tc_by_lag"] == {"1": pytest.approx(0.0), "2": pytest.approx(0.0)}
    assert result["flow_warp_error_per_transition"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["tlpips_per_transition"] == pytest.approx([0.0, 0.0, 0.0])


def test_temporal_diagnostics_record_flicker_and_lag_error() -> None:
    reference = torch.zeros(4, 3, 8, 8)
    generated = reference.clone()
    generated[1] = 1.0
    generated[2] = -1.0
    backward_flow = torch.zeros(3, 2, 8, 8)

    result = temporal_diagnostic_details(
        generated,
        reference,
        backward_flow,
        MeanAbsoluteDistance(),
        lag_values=(1, 2, 8),
        brightness_threshold=0.1,
    )

    assert result["flicker_flags"] == [True, True, True]
    assert result["flicker_rate"] == pytest.approx(1.0)
    assert set(result["tc_by_lag"]) == {"1", "2"}
    assert result["tc_by_lag"]["1"] > 0
    assert result["tc_by_lag"]["2"] > 0


def test_temporal_diagnostics_reject_incompatible_flow_shape() -> None:
    video = torch.zeros(4, 3, 8, 8)

    with pytest.raises(ValueError, match="backward flow"):
        temporal_diagnostic_details(
            video,
            video,
            torch.zeros(4, 2, 8, 8),
            MeanAbsoluteDistance(),
        )
