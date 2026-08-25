from __future__ import annotations

import torch

from tardis.models.transport import MotionStateTransport, flow_to_sampling_grid, warp_tensor


def coordinate_image(height: int = 4, width: int = 5) -> torch.Tensor:
    values = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
    return values


def test_zero_backward_flow_is_exact_identity() -> None:
    source = coordinate_image()
    flow = torch.zeros(1, 2, 4, 5)

    warped, valid = warp_tensor(source, flow)

    assert torch.equal(warped, source)
    assert torch.equal(valid, torch.ones(1, 1, 4, 5, dtype=torch.bool))


def test_positive_backward_x_flow_samples_one_cell_to_the_right() -> None:
    source = coordinate_image()
    flow = torch.zeros(1, 2, 4, 5)
    flow[:, 0] = 1.0

    warped, valid = warp_tensor(source, flow)

    assert torch.equal(warped[:, :, :, :-1], source[:, :, :, 1:])
    assert torch.equal(warped[:, :, :, -1], torch.zeros_like(warped[:, :, :, -1]))
    assert valid[:, :, :, :-1].all()
    assert not valid[:, :, :, -1].any()


def test_sampling_grid_uses_align_corners_false_pixel_centers() -> None:
    flow = torch.zeros(1, 2, 2, 4)

    grid, valid = flow_to_sampling_grid(flow)

    assert torch.allclose(grid[0, 0, :, 0], torch.tensor([-0.75, -0.25, 0.25, 0.75]))
    assert torch.allclose(grid[0, :, 0, 1], torch.tensor([-0.5, 0.5]))
    assert valid.all()


def test_transport_blends_warped_latent_with_null_state_by_visibility() -> None:
    transport = MotionStateTransport(channels=1, max_correction_pixels=0.5)
    with torch.no_grad():
        transport.null_latent.fill_(-2.0)
    previous = torch.ones(2, 1, 4, 4)
    flow = torch.zeros(2, 2, 4, 4)
    visibility = torch.stack((torch.ones(1, 4, 4), torch.zeros(1, 4, 4)))

    output = transport(previous, flow, visibility)

    assert torch.equal(output.prior[0], previous[0])
    assert torch.equal(output.prior[1], torch.full_like(previous[1], -2.0))
    assert torch.equal(output.corrected_flow, flow)


def test_transport_can_preserve_history_in_unreliable_regions() -> None:
    transport = MotionStateTransport(
        channels=1,
        max_correction_pixels=0.5,
        history_fallback_weight=1.0,
    )
    with torch.no_grad():
        transport.null_latent.fill_(-2.0)
    previous = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    flow = torch.zeros(1, 2, 4, 4)
    visibility = torch.zeros(1, 1, 4, 4)

    output = transport(previous, flow, visibility)

    assert torch.equal(output.prior, previous)


def test_transport_correction_is_bounded_and_out_of_bounds_visibility_is_zero() -> None:
    transport = MotionStateTransport(channels=1, max_correction_pixels=0.25)
    previous = torch.ones(1, 1, 4, 4)
    flow = torch.full((1, 2, 4, 4), 10.0)
    visibility = torch.ones(1, 1, 4, 4)
    raw_correction = torch.full((1, 2, 4, 4), 100.0)

    output = transport(previous, flow, visibility, raw_correction=raw_correction)

    correction = output.corrected_flow - flow
    assert correction.abs().max() <= 0.25
    assert torch.count_nonzero(output.effective_visibility) == 0
    expected = transport.null_latent.expand_as(previous)
    assert torch.equal(output.prior, expected)


def test_transport_warps_each_state_tensor_with_same_flow() -> None:
    transport = MotionStateTransport(channels=1, max_correction_pixels=0.5)
    previous = coordinate_image(4, 5)
    flow = torch.zeros(1, 2, 4, 5)
    flow[:, 0] = 1
    state = {"keys": previous + 100, "values": previous + 200}

    output = transport(previous, flow, torch.ones(1, 1, 4, 5), state=state)

    assert torch.equal(output.warped_state["keys"][:, :, :, :-1], state["keys"][:, :, :, 1:])
    assert torch.equal(output.warped_state["values"][:, :, :, :-1], state["values"][:, :, :, 1:])
