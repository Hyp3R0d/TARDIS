from __future__ import annotations

import pytest
import torch

from tardis.experiments.benchmark import parse_args
from tardis.experiments.generators import _canny_video, _farneback_warp


def test_source_method_requires_source_protocol(tmp_path) -> None:
    with pytest.raises(ValueError, match="source-conditioned"):
        parse_args(
            [
                "--method",
                "streamdiffusion_img2img",
                "--dataset",
                "dataverse",
                "--output",
                str(tmp_path),
            ]
        )


def test_source_protocol_accepts_fixed_strength(tmp_path) -> None:
    args = parse_args(
        [
            "--method",
            "streamdiffusion_img2img",
            "--dataset",
            "dataverse",
            "--protocol",
            "source50",
            "--source-strength",
            "0.45",
            "--output",
            str(tmp_path),
        ]
    )

    assert args.protocol == "source50"
    assert args.source_strength == pytest.approx(0.45)


def test_farneback_identity_motion_preserves_generated_frame() -> None:
    source = torch.linspace(-1, 1, 32).reshape(1, 4, 8).expand(3, -1, -1)
    generated = torch.rand(3, 4, 8).mul(2).sub(1)

    warped = _farneback_warp(generated, source, source)

    assert warped.shape == generated.shape
    assert torch.allclose(warped, generated, atol=1.0e-4)


def test_canny_video_preserves_protocol_shape_and_range() -> None:
    source = torch.rand(3, 3, 16, 16).mul(2).sub(1)

    edges = _canny_video(source)

    assert edges.shape == source.shape
    assert edges.min() >= -1
    assert edges.max() <= 1
