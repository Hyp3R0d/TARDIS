"""Prompt-only production video generation command."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tardis.cli.common import DATASET_CHOICES, ApplyOptions, DataOptions
from tardis.cli.generation import (
    add_model_arguments,
    add_runtime_arguments,
    validate_generated_video,
    validate_model_arguments,
)


class _Context(Protocol):
    @property
    def is_main(self) -> bool: ...

    @property
    def device(self) -> Any: ...

    def initialize(self) -> None: ...

    def close(self) -> None: ...


class _Runtime(Protocol):
    @property
    def model(self) -> Any: ...

    @property
    def checkpoint(self) -> Any: ...

    @property
    def device(self) -> Any: ...


def _default_context_factory(device_type: str | None) -> _Context:
    from tardis.utils.distributed import DistributedContext

    return DistributedContext.from_environment(device_type=device_type)


def _default_runtime_builder(
    args: argparse.Namespace,
    *,
    use_ema: bool,
) -> _Runtime:
    from tardis.cli.runtime import build_generation_runtime

    return build_generation_runtime(args, use_ema=use_ema)


@dataclass(frozen=True, slots=True)
class ApplyServices:
    """Injection boundaries for prompt-only orchestration tests."""

    context_factory: Callable[[str | None], _Context] = _default_context_factory
    runtime_builder: Callable[..., _Runtime] = _default_runtime_builder


def build_parser() -> argparse.ArgumentParser:
    """Build the apply parser without importing torch or model-weight libraries."""

    defaults = ApplyOptions()
    data = DataOptions()
    parser = argparse.ArgumentParser(description="Generate one prompt-only TARDIS MP4")
    selection = parser.add_argument_group("checkpoint selection")
    selection.add_argument("--dataset", choices=DATASET_CHOICES, default=data.dataset)
    add_model_arguments(parser, include_num_frames=False)
    add_runtime_arguments(parser)
    prompt = parser.add_argument_group("prompt")
    prompt.add_argument("--prompt", default=defaults.prompt)
    prompt.add_argument("--style", default="")
    prompt.add_argument("--duration", type=float, default=defaults.duration_seconds)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_model_arguments(args)
    if float(args.duration) <= 0:
        raise ValueError("duration must be positive")
    if not str(args.prompt).strip():
        raise ValueError("prompt cannot be empty")
    return args


def effective_prompt(prompt: str, style: str) -> str:
    """Combine style with prompt through one stable text-only template."""

    normalized_prompt = prompt.strip()
    normalized_style = style.strip()
    if not normalized_style:
        return normalized_prompt
    return f"{normalized_prompt}. Style: {normalized_style}"


def run_application(
    args: argparse.Namespace,
    *,
    services: ApplyServices | None = None,
) -> Path | None:
    """Generate and encode exactly one video on rank zero."""

    import torch

    from tardis.utils.manifest import create_output_run_dir, write_json_manifest
    from tardis.utils.random import make_generator, seed_everything
    from tardis.utils.video_io import write_mp4

    selected = ApplyServices() if services is None else services
    requested_device = torch.device(str(args.device))
    context = selected.context_factory(requested_device.type)
    try:
        context.initialize()
        if not context.is_main:
            return None
        local_args = argparse.Namespace(**vars(args))
        local_args.device = str(context.device)
        seed_everything(int(args.seed), deterministic=bool(args.deterministic))
        runtime = selected.runtime_builder(local_args, use_ema=bool(args.use_ema))
        model = runtime.model.eval()
        frame_count = max(1, int(round(float(args.duration) * int(args.fps))))
        combined_prompt = effective_prompt(str(args.prompt), str(args.style))
        generator = make_generator(int(args.seed), runtime.device)
        _synchronize(runtime.device)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                [combined_prompt],
                num_frames=frame_count,
                fps=int(args.fps),
                generator=generator,
            )
            generated_video = validate_generated_video(
                generated.video,
                batch_size=1,
                num_frames=frame_count,
                height=int(args.height),
                width=int(args.width),
            )
        _synchronize(runtime.device)
        generation_seconds = time.perf_counter() - generation_started

        output_dir = create_output_run_dir(
            Path(args.output_root),
            f"apply/{args.dataset}",
        )
        video_path = output_dir / "video.mp4"
        encoding_started = time.perf_counter()
        write_mp4(generated_video, video_path, fps=float(args.fps))
        encoding_seconds = time.perf_counter() - encoding_started
        total_seconds = generation_seconds + encoding_seconds
        write_json_manifest(
            output_dir / "video.json",
            {
                "dataset": str(args.dataset),
                "prompt": str(args.prompt),
                "style": str(args.style),
                "effective_prompt": combined_prompt,
                "seed": int(args.seed),
                "checkpoint": {
                    "path": str(runtime.checkpoint.path),
                    "sha256": str(runtime.checkpoint.sha256),
                },
                "dimensions": {"height": int(args.height), "width": int(args.width)},
                "duration": float(args.duration),
                "fps": int(args.fps),
                "frame_count": frame_count,
                "transport_quotient": {
                    "enabled": bool(args.transport_quotient),
                    "regularization": float(args.quotient_regularization),
                    "rank_threshold": float(args.quotient_rank_threshold),
                },
                "innovation_proper_time": {
                    "enabled": bool(args.innovation_proper_time),
                    "maximum_hazard": float(args.proper_time_maximum_hazard),
                },
                "latency": {
                    "generation_seconds": generation_seconds,
                    "encoding_seconds": encoding_seconds,
                    "total_seconds": total_seconds,
                    "seconds_per_frame": generation_seconds / frame_count,
                },
            },
        )
        del generated, generated_video
        return output_dir
    finally:
        context.close()


def _synchronize(device: Any) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: Sequence[str] | None = None) -> int:
    run_application(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
