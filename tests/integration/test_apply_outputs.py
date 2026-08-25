from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from tardis.utils.video_io import probe_video


class FakeContext:
    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device("cpu")
    is_main = True

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeModel:
    def __init__(self) -> None:
        self.prompts: list[list[str]] = []

    def eval(self) -> FakeModel:
        return self

    def generate(
        self,
        prompts: list[str],
        num_frames: int,
        fps: int,
        generator: torch.Generator,
    ) -> SimpleNamespace:
        del fps, generator
        self.prompts.append(prompts)
        return SimpleNamespace(video=torch.zeros(1, num_frames, 3, 16, 18))


def test_apply_parser_is_import_light_and_has_no_source_video_option() -> None:
    from tardis.cli.apply import build_parser

    parser = build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--source" not in option_strings
    assert "--source-video" not in option_strings
    assert "--video" not in option_strings
    assert "--keyframe-lite-alignment" in option_strings
    assert "--lite-max-magnitude" in option_strings
    assert "--transport-history-fallback-weight" in option_strings
    assert all(
        action.default is not argparse.SUPPRESS
        for action in parser._actions
        if action.dest != "help"
    )

    result = subprocess.run(
        [sys.executable, "-m", "tardis.cli.apply", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prompt" in result.stdout
    assert "--dataset" in result.stdout
    assert "torchvision" not in result.stderr


def test_apply_generates_prompt_only_mp4_and_sidecar(tmp_path: Path) -> None:
    from tardis.cli.apply import ApplyServices, parse_args, run_application

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    model = FakeModel()
    runtime = SimpleNamespace(
        model=model,
        checkpoint=SimpleNamespace(path=checkpoint, sha256="b" * 64, used_ema=True),
        device=torch.device("cpu"),
    )

    def runtime_builder(args: object, **kwargs: object) -> SimpleNamespace:
        del args
        assert kwargs["use_ema"] is True
        return runtime

    args = parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--prompt",
            "A lighthouse at dawn",
            "--style",
            "watercolor",
            "--duration",
            "0.5",
            "--fps",
            "4",
            "--height",
            "16",
            "--width",
            "18",
            "--seed",
            "17",
        ]
    )
    services = ApplyServices(
        context_factory=lambda _device_type: FakeContext(),
        runtime_builder=runtime_builder,
    )

    output = run_application(args, services=services)

    assert output is not None
    assert output.parent == tmp_path / "outputs" / "apply" / "dataverse"
    mp4 = output / "video.mp4"
    sidecar_path = output / "video.json"
    info = probe_video(mp4)
    assert info.playable
    assert info.frame_count == 2
    assert info.fps == 4
    assert (info.height, info.width) == (16, 18)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar == {
        "checkpoint": {"path": str(checkpoint), "sha256": "b" * 64},
        "dataset": "dataverse",
        "dimensions": {"height": 16, "width": 18},
        "duration": 0.5,
        "effective_prompt": "A lighthouse at dawn. Style: watercolor",
        "fps": 4,
        "frame_count": 2,
        "latency": sidecar["latency"],
        "innovation_proper_time": {"enabled": True, "maximum_hazard": 20.0},
        "prompt": "A lighthouse at dawn",
        "seed": 17,
        "style": "watercolor",
        "transport_quotient": {
            "enabled": True,
            "rank_threshold": 1.0e-5,
            "regularization": 1.0e-4,
        },
    }
    assert sidecar["latency"]["total_seconds"] >= 0
    assert sidecar["latency"]["seconds_per_frame"] >= 0
    assert model.prompts == [["A lighthouse at dawn. Style: watercolor"]]
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.jpg"))
    assert not list(output.rglob("*.jpeg"))
