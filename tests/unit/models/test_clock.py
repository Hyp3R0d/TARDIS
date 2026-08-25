from __future__ import annotations

import math

import torch

from tardis.models.clock import InnovationProperTime


def build_clock(*, active_ratio: float = 0.25, threshold: float = 0.5) -> InnovationProperTime:
    return InnovationProperTime(
        patch_size=2,
        active_ratio=active_ratio,
        threshold=threshold,
        halo_radius=0,
        maximum_hazard=20.0,
    )


def test_proper_time_is_the_exact_cumulative_hazard_of_unserved_risk() -> None:
    clock = build_clock(active_ratio=1.0, threshold=0.99)
    previous = torch.zeros(1, 1, 4, 4)
    visibility = torch.ones_like(previous)
    risk = torch.full_like(previous, 0.1)

    first = clock(previous, risk, visibility)
    second = clock(first.settled_hazard, risk, visibility)

    assert torch.allclose(first.event_probability, torch.full_like(previous, 0.1))
    assert torch.allclose(
        second.event_probability,
        torch.full_like(previous, 1 - 0.9**2),
        atol=1.0e-6,
    )
    assert torch.allclose(second.accrued_hazard, torch.full_like(previous, -2 * math.log(0.9)))


def test_selected_renewal_events_clear_only_the_served_patch() -> None:
    clock = build_clock(active_ratio=0.25, threshold=0.1)
    previous = torch.zeros(1, 1, 4, 4)
    visibility = torch.ones_like(previous)
    risk = torch.zeros_like(previous)
    risk[..., :2, :2] = 0.9
    risk[..., 2:, 2:] = 0.8

    output = clock(previous, risk, visibility)

    assert output.selection.active_counts.tolist() == [1]
    assert output.service_mask[..., :2, :2].all()
    assert torch.count_nonzero(output.settled_hazard[..., :2, :2]) == 0
    assert torch.count_nonzero(output.settled_hazard[..., 2:, 2:]) > 0


def test_motion_visibility_discards_debt_that_cannot_be_transport_aligned() -> None:
    clock = build_clock(active_ratio=1.0, threshold=0.99)
    previous = torch.full((1, 1, 4, 4), 2.0)
    risk = torch.zeros_like(previous)
    visibility = torch.zeros_like(previous)

    output = clock(previous, risk, visibility)

    assert torch.count_nonzero(output.accrued_hazard) == 0
    assert torch.count_nonzero(output.event_probability) == 0


def test_repeated_small_hazard_eventually_outranks_a_newer_larger_risk() -> None:
    clock = build_clock(active_ratio=0.25, threshold=0.0)
    previous = torch.zeros(1, 1, 4, 4)
    previous[..., :2, :2] = -math.log(1 - 0.7)
    risk = torch.zeros_like(previous)
    risk[..., 2:, 2:] = 0.6

    output = clock(previous, risk, torch.ones_like(previous))

    assert output.selection.active_counts.tolist() == [1]
    assert output.service_mask[..., :2, :2].all()
    assert not output.service_mask[..., 2:, 2:].any()


def test_half_precision_hazard_math_is_finite_near_unit_risk() -> None:
    clock = build_clock(active_ratio=1.0, threshold=0.0)
    previous = torch.zeros(1, 1, 2, 2, dtype=torch.float16)
    risk = torch.ones_like(previous)

    output = clock(previous, risk, torch.ones_like(previous))

    assert output.accrued_hazard.dtype == torch.float16
    assert torch.isfinite(output.accrued_hazard).all()
    assert torch.isfinite(output.event_probability).all()
