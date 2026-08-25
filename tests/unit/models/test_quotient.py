from __future__ import annotations

import torch

from tardis.models.quotient import TransportOrbitProjector


def test_transport_quotient_is_an_exact_additive_decomposition() -> None:
    projector = TransportOrbitProjector()
    prior = torch.randn(2, 4, 7, 9)
    value = torch.randn_like(prior)
    basis = projector.build_basis(prior, torch.ones(2, 1, 7, 9))

    result = projector.decompose(value, basis)

    assert torch.allclose(result.tangent + result.innovation, value)
    assert result.flow_coefficients.shape == (2, 2, 7, 9)
    assert result.tangent_rank.shape == (2, 1, 7, 9)


def test_transport_tangent_change_is_removed_from_quotient_innovation() -> None:
    projector = TransportOrbitProjector(regularization=1.0e-7)
    prior = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    basis = projector.build_basis(prior, torch.ones(1, 1, 8, 8, dtype=torch.float64))
    tangent_change = 0.3 * basis.gradient_x - 0.2 * basis.gradient_y

    result = projector.decompose(tangent_change, basis)

    assert result.innovation.square().mean() < 1.0e-10
    assert torch.allclose(result.tangent, tangent_change, atol=2.0e-5, rtol=2.0e-5)


def test_invisible_region_cannot_discard_change_as_motion() -> None:
    projector = TransportOrbitProjector()
    prior = torch.randn(1, 4, 5, 6)
    value = torch.randn_like(prior)
    basis = projector.build_basis(prior, torch.zeros(1, 1, 5, 6))

    result = projector.decompose(value, basis)

    assert torch.count_nonzero(result.tangent) == 0
    assert torch.equal(result.innovation, value)
    assert torch.count_nonzero(result.tangent_rank) == 0


def test_channel_change_orthogonal_to_transport_orbit_remains_innovation() -> None:
    projector = TransportOrbitProjector()
    horizontal = torch.arange(7, dtype=torch.float32)[None, None, None].expand(1, 1, 6, 7)
    vertical = torch.arange(6, dtype=torch.float32)[None, None, :, None].expand(1, 1, 6, 7)
    prior = torch.cat((horizontal, vertical, torch.zeros(1, 2, 6, 7)), dim=1)
    value = torch.zeros_like(prior)
    value[:, 2] = 1
    basis = projector.build_basis(prior, torch.ones(1, 1, 6, 7))

    result = projector.decompose(value, basis)

    assert torch.count_nonzero(result.tangent) == 0
    assert torch.equal(result.innovation, value)
