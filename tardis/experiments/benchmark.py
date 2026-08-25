"""Unified, resumable prompt-to-video benchmark under the locked TARDIS protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tardis.metrics.base import FramePairFeature, validate_video_pair
from tardis.utils.checkpoint import atomic_torch_save
from tardis.utils.manifest import write_json_manifest
from tardis.utils.resources import ResourceMonitor, sample_resources
from tardis.utils.video_io import write_mp4

METHODS = (
    "tardis",
    "animatediff_lightning",
    "sd_turbo_independent",
    "text2video_zero",
    "streamdiffusion_img2img",
    "rerender_flow",
    "tokenflow_core",
    "vid2vid_zero_core",
    "controlvideo_canny",
    "stablevideo_propagation",
    *(f"tardis_a{index}" for index in range(11)),
)
SOURCE_CONDITIONED_METHODS = {
    "streamdiffusion_img2img",
    "rerender_flow",
    "tokenflow_core",
    "vid2vid_zero_core",
    "controlvideo_canny",
    "stablevideo_propagation",
}
DATASETS = ("dataverse", "seedance", "openvid")
InputRange = Literal["zero_one", "minus_one_one"]
MetricMode = Literal["primary", "full"]


class MetricSuiteLike(Protocol):
    provenance_ids: Mapping[str, str]

    def update(self, generated: torch.Tensor, reference: torch.Tensor, prompt: str) -> None: ...

    def compute(self) -> dict[str, dict[str, float]]: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


@dataclass(slots=True)
class PrimaryMetricSuite:
    """Small resumable suite used by pilot runs."""

    tc_sum: float = 0.0
    lpips_sum: float = 0.0
    count: int = 0

    @property
    def provenance_ids(self) -> dict[str, str]:
        return {
            "tc": "tardis/official-temporal-consistency:v1",
            "lpips": "lpips:alex:lpips-v0.1",
        }

    def update_values(self, tc: float, lpips: float) -> None:
        if not math.isfinite(tc) or not math.isfinite(lpips) or min(tc, lpips) < 0:
            raise ValueError("primary metric values must be finite and non-negative")
        self.tc_sum += tc
        self.lpips_sum += lpips
        self.count += 1

    def update(self, generated: torch.Tensor, reference: torch.Tensor, prompt: str) -> None:
        del generated, reference, prompt
        raise RuntimeError("PrimaryMetricSuite requires update_values with shared LPIPS output")

    def compute(self) -> dict[str, dict[str, float]]:
        if self.count <= 0:
            raise RuntimeError("primary metric suite has no observations")
        values = {"tc": self.tc_sum / self.count, "lpips": self.lpips_sum / self.count}
        return {"macro": dict(values), "micro": dict(values)}

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "tc_sum": self.tc_sum,
            "lpips_sum": self.lpips_sum,
            "count": self.count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"version", "tc_sum", "lpips_sum", "count"} or state["version"] != 1:
            raise ValueError("primary metric suite state is incompatible")
        tc_sum = float(cast(float, state["tc_sum"]))
        lpips_sum = float(cast(float, state["lpips_sum"]))
        count = int(cast(int, state["count"]))
        if count < 0 or not math.isfinite(tc_sum) or not math.isfinite(lpips_sum):
            raise ValueError("primary metric suite state is invalid")
        self.tc_sum = tc_sum
        self.lpips_sum = lpips_sum
        self.count = count


def normalize_video(
    value: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
    input_range: InputRange,
) -> torch.Tensor:
    """Normalize common pipeline layouts to float32 [T,3,H,W] in [-1, 1]."""

    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    video = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("generated video batch dimension must equal one")
        video = video[0]
    if video.ndim != 4:
        raise ValueError("generated video must have four dimensions after batch removal")
    if video.shape[1] == 3:
        normalized = video
    elif video.shape[-1] == 3:
        normalized = video.permute(0, 3, 1, 2)
    else:
        raise ValueError("generated video must expose one RGB channel axis")
    expected = (num_frames, 3, height, width)
    if tuple(normalized.shape) != expected:
        raise ValueError(
            f"generated video does not match protocol shape {expected}; "
            f"received {tuple(normalized.shape)}"
        )
    normalized = normalized.to(dtype=torch.float32)
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("generated video contains non-finite values")
    minimum = float(normalized.min().item())
    maximum = float(normalized.max().item())
    tolerance = 1.0e-3
    if input_range == "zero_one":
        if minimum < -tolerance or maximum > 1 + tolerance:
            raise ValueError("zero-one generated video lies outside [0, 1]")
        normalized = normalized.clamp(0, 1).mul(2).sub(1)
    elif input_range == "minus_one_one":
        if minimum < -1 - tolerance or maximum > 1 + tolerance:
            raise ValueError("normalized generated video lies outside [-1, 1]")
        normalized = normalized.clamp(-1, 1)
    else:
        raise ValueError(f"unsupported input range: {input_range!r}")
    return normalized.contiguous()


def primary_metric_details(
    generated: torch.Tensor,
    reference: torch.Tensor,
    lpips_feature: FramePairFeature,
) -> dict[str, Any]:
    """Return official per-transition TC and framewise AlexNet LPIPS."""

    validate_video_pair(generated, reference, min_frames=2)
    generated_delta = generated[1:] - generated[:-1]
    reference_delta = reference[1:] - reference[:-1]
    tc_values = (generated_delta - reference_delta).abs().mean(dim=(1, 2, 3))
    lpips_values = lpips_feature(generated, reference).detach().reshape(-1)
    if lpips_values.numel() != generated.shape[0]:
        raise RuntimeError("LPIPS feature did not return one score per frame")
    return {
        "tc": float(tc_values.mean().item()),
        "lpips": float(lpips_values.mean().item()),
        "tc_per_transition": [float(value) for value in tc_values.cpu().tolist()],
        "lpips_per_frame": [float(value) for value in lpips_values.cpu().tolist()],
    }


def primary_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    lpips_feature: FramePairFeature,
) -> dict[str, float]:
    details = primary_metric_details(generated, reference, lpips_feature)
    return {"tc": float(details["tc"]), "lpips": float(details["lpips"])}


def append_jsonl(path: Path, item: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(item), ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        record_id = str(item.get("record_id", ""))
        if not record_id or record_id in seen:
            raise ValueError(
                f"JSONL contains an empty or duplicate record ID at line {line_number}"
            )
        seen.add(record_id)
        records.append(item)
    return records


def summarize_latencies(latencies: Sequence[float], *, num_frames: int) -> dict[str, float | int]:
    values = np.asarray(latencies, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latencies must be a non-empty finite non-negative vector")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return {
        "video_count": int(values.size),
        "mean_video_seconds": float(values.mean()),
        "p50_video_seconds": float(np.quantile(values, 0.50)),
        "p95_video_seconds": float(np.quantile(values, 0.95)),
        "p99_video_seconds": float(np.quantile(values, 0.99)),
        "max_video_seconds": float(values.max()),
        "mean_frame_milliseconds": float(values.mean() * 1000 / num_frames),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-split",
        choices=("test", "validation"),
        default="test",
        help="dataset split to evaluate; validation runs require an explicit record manifest",
    )
    parser.add_argument(
        "--protocol",
        choices=(
            "paper50",
            "source50",
            "source50_diagnostics",
            "formal512",
            "pilot",
            "source_pilot",
        ),
        default="paper50",
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--record-ids-file", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--datasets-file", type=Path, default=Path("/home/TARDIS/datasets.txt"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--source-strength", type=float, default=0.45)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--metrics", choices=("primary", "full"), default="full")
    parser.add_argument(
        "--temporal-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record flow-warp error, tLPIPS, flicker and long-range TC diagnostics",
    )
    parser.add_argument(
        "--flow-cache-root",
        type=Path,
        default=Path("/root/autodl-tmp/TARDIS/metric_cache/raft_small_backward"),
    )
    parser.add_argument("--flow-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--showcase-count", type=int, default=0)
    parser.add_argument(
        "--qualitative-root",
        type=Path,
        default=Path("/home/TARDIS/RTVD-TC-DataPackage-v1.0/qualitative_raw"),
    )
    parser.add_argument(
        "--sd-turbo-model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/TARDIS/cache/huggingface/hub/"
            "models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2"
        ),
    )
    parser.add_argument(
        "--sd15-model",
        type=Path,
        default=Path("/root/autodl-tmp/TARDIS/cache/sd15"),
    )
    parser.add_argument(
        "--animatediff-adapter",
        type=Path,
        default=Path(
            "/root/autodl-tmp/TARDIS/cache/animatediff-lightning/"
            "animatediff_lightning_2step_diffusers.safetensors"
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--keep-state", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    positive = (
        args.limit,
        args.validation_size,
        args.test_size,
        args.height,
        args.width,
        args.num_frames,
        args.fps,
        args.prefetch_factor,
        args.checkpoint_every,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("benchmark dimensions and counts must be positive")
    if args.num_workers < 0 or args.max_retries < 0 or args.showcase_count < 0:
        raise ValueError("workers, retries, and showcase count cannot be negative")
    if args.flow_batch_size <= 0:
        raise ValueError("flow batch size must be positive")
    if not 0 <= args.source_strength <= 1:
        raise ValueError("source strength must be in [0, 1]")
    pilot_protocol = args.protocol in {"pilot", "source_pilot"}
    if pilot_protocol and args.limit > args.test_size:
        raise ValueError("pilot limit cannot exceed test_size")
    if not pilot_protocol and args.limit != 2:
        raise ValueError("--limit is only valid for pilot protocol")
    if args.method in SOURCE_CONDITIONED_METHODS and args.protocol not in {
        "source50",
        "source50_diagnostics",
        "source_pilot",
    }:
        raise ValueError("source-conditioned benchmark methods require a source protocol")
    if args.temporal_diagnostics and not _is_source_protocol(args.protocol):
        raise ValueError("temporal diagnostics require a source-conditioned protocol")
    if args.method.startswith("tardis"):
        if args.checkpoint is None:
            args.checkpoint = Path(f"/home/TARDIS/TARDIS_SOTA/weights/{args.dataset}_best.pt")
        if args.train_manifest is None:
            args.train_manifest = _infer_train_manifest(args.checkpoint, args.dataset)
    return args


def run_benchmark(args: argparse.Namespace) -> Path:
    """Run or resume one method/dataset/seed unit and write auditable raw results."""

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/TARDIS/cache/huggingface")
    os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/TARDIS/cache/torch")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            print(f"already completed: {output}")
            return output

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper benchmark requires one available CUDA device")
    selected_records, all_test_records = _selected_records(args)
    selected_ids = [record.id for record in selected_records]
    settings = _settings(args, selected_ids)
    records_path = output / "per_video.jsonl"
    state_path = output / "resume_state.pt"
    completed = load_jsonl(records_path)
    suite, lpips_feature = _build_metrics(str(args.metrics), device)
    if state_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"resume state exists and --no-resume was requested: {state_path}"
            )
        completed = _restore_state(
            state_path,
            settings=settings,
            suite=suite,
            records_path=records_path,
            records=completed,
        )
    elif completed:
        raise RuntimeError("per-video ledger exists without a matching resume state")

    completed_ids = [str(item["record_id"]) for item in completed]
    expected_prefix = selected_ids[: len(completed_ids)]
    if completed_ids != expected_prefix:
        raise RuntimeError("resume ledger is not the canonical selected-record prefix")

    monitor = ResourceMonitor(sample_fn=lambda: sample_resources(device), interval_seconds=1.0)
    monitor.sample_once()
    monitor.start()
    generator = None
    diagnostic_cache: Any | None = None
    diagnostic_provenance: dict[str, str] = {}
    started = time.perf_counter()
    try:
        from tardis.experiments.generators import build_generator

        generator = build_generator(
            str(args.method),
            dataset=str(args.dataset),
            checkpoint=args.checkpoint,
            train_manifest=args.train_manifest,
            device=device,
            precision=str(args.precision),
            sd_turbo_model=args.sd_turbo_model.expanduser().resolve(strict=True),
            sd15_model=args.sd15_model.expanduser().resolve(strict=True),
            animatediff_adapter=args.animatediff_adapter.expanduser().resolve(strict=True),
        )
        generator_provenance = dict(generator.provenance)
        if args.temporal_diagnostics:
            from tardis.experiments.flow_cache import (
                BackwardFlowCache,
                TorchvisionRAFTSmallBackwardFlow,
            )

            estimator = TorchvisionRAFTSmallBackwardFlow(
                device=device,
                batch_size=int(args.flow_batch_size),
            )
            diagnostic_cache = BackwardFlowCache(
                args.flow_cache_root,
                estimator=estimator,
            )
            diagnostic_provenance = {
                "flow_warp_error": estimator.provenance_id,
                "tlpips": "lpips:alex:lpips-v0.1:flow-warped",
                "flicker_rate": "tardis/brightness-delta-threshold-0.1:v1",
                "tc_by_lag": "tardis/official-temporal-consistency:multi-lag-v1",
            }
        _write_run_manifest(
            output,
            status="running",
            settings=settings,
            generator_provenance=generator_provenance,
        )
        loader = _build_loader(args, all_test_records, selected_ids, set(completed_ids))
        showcase_ids = _showcase_ids(
            selected_records,
            int(args.showcase_count),
            int(args.split_seed),
        )
        progress = tqdm(
            total=len(selected_records),
            initial=len(completed),
            desc=f"{args.method}:{args.dataset}:seed{args.seed}",
            dynamic_ncols=True,
        )
        try:
            for batch in loader:
                from tardis.data.dataset import BenchmarkFailure

                if isinstance(batch, BenchmarkFailure):
                    failure = {
                        "record_id": batch.record.id,
                        "seed": int(batch.sample_seed),
                        "error_type": batch.error_type,
                        "error_message": batch.error_message,
                    }
                    append_jsonl(output / "failures.jsonl", failure)
                    raise RuntimeError(f"dataset decode failed for {batch.record.id}")
                if len(batch.record_ids) != 1:
                    raise RuntimeError("benchmark loader must yield batch size one")
                record_id = str(batch.record_ids[0])
                prompt = str(batch.prompts[0])
                sample_seed = int(batch.sample_seeds[0])
                reference = batch.video[0].to(device=device, non_blocking=True)
                torch.cuda.synchronize(device)
                generation_started = time.perf_counter()
                generated = generator.generate(
                    prompt,
                    seed=sample_seed,
                    num_frames=int(args.num_frames),
                    height=int(args.height),
                    width=int(args.width),
                    source_video=(reference if _is_source_protocol(args.protocol) else None),
                    source_strength=float(args.source_strength),
                )
                torch.cuda.synchronize(device)
                generation_seconds = time.perf_counter() - generation_started
                generated = generated.to(device=device, dtype=torch.float32)
                details = primary_metric_details(generated, reference, lpips_feature)
                diagnostics: dict[str, Any] = {}
                if diagnostic_cache is not None:
                    from tardis.experiments.temporal_diagnostics import (
                        temporal_diagnostic_details,
                    )

                    backward_flow = diagnostic_cache.get_or_compute(
                        str(args.dataset),
                        record_id,
                        reference,
                    )
                    diagnostics = temporal_diagnostic_details(
                        generated,
                        reference,
                        backward_flow,
                        lpips_feature,
                    )
                if isinstance(suite, PrimaryMetricSuite):
                    suite.update_values(float(details["tc"]), float(details["lpips"]))
                else:
                    suite.update(generated, reference, prompt)
                item = {
                    "experiment_id": f"exp01_{args.method}_{args.dataset}_seed{args.seed}",
                    "dataset": str(args.dataset),
                    "method": str(args.method),
                    "record_id": record_id,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "seed": sample_seed,
                    "global_seed": int(args.seed),
                    "frame_count": int(args.num_frames),
                    "generation_seconds": generation_seconds,
                    "tc": float(details["tc"]),
                    "lpips": float(details["lpips"]),
                    "tc_per_transition": details["tc_per_transition"],
                    "lpips_per_frame": details["lpips_per_frame"],
                    **diagnostics,
                    "status": "completed",
                }
                append_jsonl(records_path, item)
                completed.append(item)
                if record_id in showcase_ids:
                    _write_showcase(
                        args,
                        record_id=record_id,
                        generated=generated,
                        reference=reference,
                    )
                if len(completed) % int(args.checkpoint_every) == 0:
                    _save_state(
                        state_path,
                        settings=settings,
                        suite=suite,
                        completed_ids=[str(record["record_id"]) for record in completed],
                    )
                progress.update(1)
                progress.set_postfix(
                    tc=f"{details['tc']:.4f}",
                    lpips=f"{details['lpips']:.4f}",
                    seconds=f"{generation_seconds:.2f}",
                )
                del generated, reference
        finally:
            progress.close()

        if len(completed) != len(selected_records):
            raise RuntimeError(
                f"benchmark coverage is incomplete: {len(completed)}/{len(selected_records)}"
            )
        _save_state(
            state_path,
            settings=settings,
            suite=suite,
            completed_ids=[str(record["record_id"]) for record in completed],
        )
        summary = suite.compute()
        latency = summarize_latencies(
            [float(item["generation_seconds"]) for item in completed],
            num_frames=int(args.num_frames),
        )
        monitor.stop()
        resources = asdict(monitor.summary())
        result = {
            "schema_version": 1,
            "status": "completed",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "settings": settings,
            "generator": generator_provenance,
            "coverage": {
                "expected": len(selected_records),
                "completed": len(completed),
                "failed": 0,
            },
            "metrics": summary,
            "latency": latency,
            "resources": resources,
            "elapsed_wall_seconds": time.perf_counter() - started,
            "per_video_jsonl": str(records_path),
            "metric_provenance": dict(suite.provenance_ids),
            "diagnostic_provenance": diagnostic_provenance,
        }
        write_json_manifest(metrics_path, result)
        _write_run_manifest(
            output,
            status="completed",
            settings=settings,
            generator_provenance=generator_provenance,
            result=result,
        )
        if not bool(args.keep_state):
            state_path.unlink(missing_ok=True)
        return output
    except BaseException as error:
        append_jsonl(
            output / "failures.jsonl",
            {
                "record_id": f"run:{args.method}:{args.dataset}:seed{args.seed}",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": time.time(),
            },
        )
        _write_run_manifest(
            output,
            status="failed",
            settings=settings,
            generator_provenance=(
                {} if generator is None else dict(generator.provenance)
            ),
            error=error,
        )
        raise
    finally:
        monitor.stop()


def _selected_records(args: argparse.Namespace) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from tardis.cli.runtime import read_dataset_sources
    from tardis.data.assembly import RangeClientFactory, build_remote_catalog
    from tardis.data.catalog import normalize_local_dataset_roots
    from tardis.data.dataset import build_split_records
    from tardis.data.splits import StablePartition

    roots = normalize_local_dataset_roots(read_dataset_sources(args.datasets_file))
    client_factory = RangeClientFactory(
        max_object_bytes=128 * 1024 * 1024,
        timeout_seconds=float(args.request_timeout_seconds),
        max_retries=int(args.max_retries),
    )
    catalog = build_remote_catalog(
        client_factory=client_factory,
        dataset_roots=roots,
        selected_source=str(args.dataset),
    )
    splits = build_split_records(
        catalog.records_by_source,
        StablePartition(
            seed=int(args.split_seed),
            validation_size=int(args.validation_size),
            test_size=int(args.test_size),
            group_by_caption=str(args.dataset) == "seedance",
        ),
    )
    data_split = str(args.data_split)
    all_records = tuple(splits[data_split][str(args.dataset)])
    expected_size = int(args.validation_size if data_split == "validation" else args.test_size)
    if len(all_records) != expected_size:
        raise RuntimeError(
            f"locked {data_split} split has unexpected size: "
            f"expected {expected_size}, received {len(all_records)}"
        )
    if args.protocol == "formal512":
        return all_records, all_records
    if args.protocol in {"pilot", "source_pilot"}:
        return all_records[: int(args.limit)], all_records
    if args.record_ids_file is not None:
        record_ids_file = args.record_ids_file
    elif data_split == "test":
        record_ids_file = Path(
            f"/home/TARDIS/RTVD-TC-DataPackage-v1.0/01_configs/video_splits/"
            f"{args.dataset}_test.json"
        )
    else:
        raise ValueError(
            "validation paper/source protocol runs require --record-ids-file so the "
            "selection subset is frozen independently of the test manifest"
        )
    payload = json.loads(
        record_ids_file.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    requested_ids = [str(item["record_id"]) for item in payload["records"]]
    if len(requested_ids) != 50 or len(set(requested_ids)) != 50:
        raise ValueError("paper/source record manifest must contain exactly 50 unique IDs")
    by_id = {record.id: record for record in all_records}
    missing = [record_id for record_id in requested_ids if record_id not in by_id]
    if missing:
        raise ValueError(f"record manifest contains IDs outside the selected split: {missing}")
    selected_set = set(requested_ids)
    selected = tuple(record for record in all_records if record.id in selected_set)
    if len(selected) != 50:
        raise RuntimeError("selected-record coverage mismatch")
    return selected, all_records


def _build_loader(
    args: argparse.Namespace,
    all_test_records: tuple[Any, ...],
    selected_ids: list[str],
    completed_ids: set[str],
) -> DataLoader[Any]:
    from tardis.data.assembly import RangeClientFactory
    from tardis.data.dataset import (
        ClipDecodeOptions,
        RemoteClipLoader,
        RemoteSourceClipIterableDataset,
        ResilientRemoteSourceClipIterableDataset,
        collate_benchmark_items,
    )

    client_factory = RangeClientFactory(
        max_object_bytes=128 * 1024 * 1024,
        timeout_seconds=float(args.request_timeout_seconds),
        max_retries=int(args.max_retries),
    )
    dataset = RemoteSourceClipIterableDataset(
        all_test_records,
        source=str(args.dataset),
        split=str(args.data_split),
        seed=int(args.seed),
        rank=0,
        world_size=1,
        client_factory=client_factory,
        clip_loader=RemoteClipLoader(
            ClipDecodeOptions(
                num_frames=int(args.num_frames),
                height=int(args.height),
                width=int(args.width),
                mode="benchmark",
                max_media_bytes=128 * 1024 * 1024,
                timeout_seconds=float(args.request_timeout_seconds),
                random_flip=False,
            )
        ),
        max_retries=int(args.max_retries),
    )
    target = set(selected_ids)
    exclude = {record.id for record in all_test_records if record.id not in target} | completed_ids
    dataset.exclude_record_ids(exclude)
    resilient = ResilientRemoteSourceClipIterableDataset(dataset)
    return DataLoader(
        resilient,
        batch_size=1,
        num_workers=int(args.num_workers),
        collate_fn=collate_benchmark_items,
        pin_memory=True,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) else None,
        persistent_workers=int(args.num_workers) > 0,
        multiprocessing_context="spawn" if int(args.num_workers) else None,
    )


def _build_metrics(mode: str, device: torch.device) -> tuple[MetricSuiteLike, FramePairFeature]:
    if mode == "primary":
        from tardis.metrics.features import AlexNetLPIPS

        feature = AlexNetLPIPS(device=device)
        return PrimaryMetricSuite(), feature
    from tardis.cli.runtime import build_metric_suite

    suite = build_metric_suite(device=device)
    return suite, suite.lpips.feature


def _settings(args: argparse.Namespace, selected_ids: list[str]) -> dict[str, object]:
    digest = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    settings: dict[str, object] = {
        "method": str(args.method),
        "dataset": str(args.dataset),
        "data_split": str(args.data_split),
        "protocol": str(args.protocol),
        "records": len(selected_ids),
        "record_ids_sha256": digest,
        "global_seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "validation_size": int(args.validation_size),
        "test_size": int(args.test_size),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "fps": int(args.fps),
        "precision": str(args.precision),
        "metric_mode": str(args.metrics),
        "reference_role": (
            "source condition and metric evaluator"
            if _is_source_protocol(args.protocol)
            else "metric evaluator only"
        ),
        "generator_inputs": (
            ["prompt", "source_video", "deterministic sample seed"]
            if _is_source_protocol(args.protocol)
            else ["prompt", "deterministic sample seed"]
        ),
        "source_strength": float(args.source_strength),
    }
    if args.temporal_diagnostics:
        settings.update(
            {
                "temporal_diagnostics": True,
                "flow_estimator": "torchvision/raft_small:Raft_Small_Weights.C_T_V2",
                "flow_cache_root": str(args.flow_cache_root.expanduser().resolve()),
                "flow_batch_size": int(args.flow_batch_size),
            }
        )
    return settings


def _is_source_protocol(protocol: object) -> bool:
    return str(protocol) in {"source50", "source50_diagnostics", "source_pilot"}


def _save_state(
    path: Path,
    *,
    settings: Mapping[str, object],
    suite: MetricSuiteLike,
    completed_ids: list[str],
) -> None:
    atomic_torch_save(
        {
            "version": 1,
            "settings": dict(settings),
            "completed_ids": completed_ids,
            "metric_suite": suite.state_dict(),
        },
        path,
    )


def _restore_state(
    path: Path,
    *,
    settings: Mapping[str, object],
    suite: MetricSuiteLike,
    records_path: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1 or payload.get("settings") != dict(settings):
        raise ValueError("resume state settings do not match this benchmark run")
    completed_ids = [str(value) for value in payload.get("completed_ids", [])]
    if len(records) < len(completed_ids):
        raise RuntimeError("resume ledger is shorter than metric state")
    if [str(item["record_id"]) for item in records[: len(completed_ids)]] != completed_ids:
        raise RuntimeError("resume ledger prefix does not match metric state")
    if len(records) > len(completed_ids):
        records = records[: len(completed_ids)]
        _rewrite_jsonl(records_path, records)
    metric_state = payload.get("metric_suite")
    if not isinstance(metric_state, Mapping):
        raise ValueError("resume metric state is missing")
    suite.load_state_dict(metric_state)
    return records


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _showcase_ids(records: tuple[Any, ...], count: int, seed: int) -> set[str]:
    if count <= 0:
        return set()
    if count > len(records):
        raise ValueError("showcase count cannot exceed selected record count")
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}\x1f{record.id}".encode()).hexdigest(),
    )
    return {record.id for record in ranked[:count]}


def _write_showcase(
    args: argparse.Namespace,
    *,
    record_id: str,
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", record_id)
    root = args.qualitative_root.expanduser().resolve()
    generated_path = root / str(args.method) / str(args.dataset) / f"{safe_id}.mp4"
    reference_path = root / "reference" / str(args.dataset) / f"{safe_id}.mp4"
    if not generated_path.is_file():
        write_mp4(generated, generated_path, fps=float(args.fps))
    if not reference_path.is_file():
        write_mp4(reference, reference_path, fps=float(args.fps))


def _write_run_manifest(
    output: Path,
    *,
    status: str,
    settings: Mapping[str, object],
    generator_provenance: Mapping[str, object],
    result: Mapping[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "command": [sys.executable, *sys.argv],
        "settings": dict(settings),
        "generator": dict(generator_provenance),
    }
    if result is not None:
        payload["result"] = dict(result)
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    write_json_manifest(output / "run_manifest.json", payload)


def _infer_train_manifest(checkpoint: Path, dataset: str) -> Path:
    resolved = checkpoint.expanduser().resolve(strict=True)
    run_id = resolved.parent.name
    candidate = Path(
        f"/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/outputs/train/"
        f"{dataset}/{run_id}/manifest.json"
    )
    return candidate.resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = run_benchmark(args)
    print(output)


if __name__ == "__main__":
    main()
