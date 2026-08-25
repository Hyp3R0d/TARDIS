"""Resumable distributed evaluation for one selected test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol, cast

from tardis.cli.common import (
    DATASET_CHOICES,
    DataOptions,
    InferOptions,
    parse_record_ids_csv,
)
from tardis.cli.generation import (
    add_model_arguments,
    add_runtime_arguments,
    validate_generated_video,
    validate_model_arguments,
)

_STATE_VERSION = 2


class _Context(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def is_main(self) -> bool: ...

    @property
    def device(self) -> Any: ...

    def initialize(self) -> None: ...

    def barrier(self) -> None: ...

    def close(self) -> None: ...


class _MetricSuite(Protocol):
    @property
    def provenance_ids(self) -> Mapping[str, str]: ...

    def update(self, generated: Any, reference: Any, prompt: str) -> None: ...

    def compute(self) -> dict[str, dict[str, float]]: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...

    def all_reduce(self) -> None: ...

    def reset(self) -> None: ...


class _Runtime(Protocol):
    @property
    def model(self) -> Any: ...

    @property
    def metric_suite(self) -> _MetricSuite: ...

    @property
    def checkpoint(self) -> Any: ...

    @property
    def dataset_sources(self) -> tuple[str, ...]: ...

    @property
    def device(self) -> Any: ...


class _Loaders(Protocol):
    @property
    def test(self) -> Mapping[str, Any]: ...


class _Monitor(Protocol):
    def sample_once(self) -> object: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def summary(self) -> _ResourceSummary: ...


class _ResourceSummary(Protocol):
    sample_count: int
    peak_allocated_mb: float
    peak_reserved_mb: float
    peak_process_rss_mb: float
    mean_gpu_utilization_percent: float


@dataclass(frozen=True, slots=True)
class _InferenceLoaders:
    test: Mapping[str, Any]


def _default_context_factory(device_type: str | None) -> _Context:
    from tardis.utils.distributed import DistributedContext

    return DistributedContext.from_environment(device_type=device_type)


def _default_runtime_builder(
    args: argparse.Namespace,
    *,
    use_ema: bool,
) -> _Runtime:
    from tardis.cli.runtime import build_production_runtime

    return build_production_runtime(args, use_ema=use_ema)


def _default_loader_builder(args: argparse.Namespace, context: _Context) -> _Loaders:
    from torch.utils.data import DataLoader

    from tardis.cli.runtime import read_dataset_sources
    from tardis.data.assembly import RangeClientFactory, build_remote_catalog
    from tardis.data.catalog import normalize_local_dataset_roots
    from tardis.data.dataset import (
        ClipDecodeOptions,
        RemoteClipLoader,
        RemoteSourceClipIterableDataset,
        ResilientRemoteSourceClipIterableDataset,
        build_split_records,
        collate_benchmark_items,
    )
    from tardis.data.splits import StablePartition

    decode = ClipDecodeOptions(
        num_frames=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
        mode="benchmark",
        timeout_seconds=float(args.request_timeout_seconds),
        random_flip=False,
    )
    client_factory = RangeClientFactory(
        max_object_bytes=decode.max_media_bytes,
        timeout_seconds=float(args.request_timeout_seconds),
        max_retries=int(args.max_retries),
    )
    catalog = build_remote_catalog(
        client_factory=client_factory,
        endpoint=str(args.mirror_endpoint),
        dataset_roots=normalize_local_dataset_roots(
            read_dataset_sources(args.datasets_file, args.mirror_endpoint)
        ),
        max_records_per_source=args.catalog_record_limit,
        openvid_archive_limit=args.openvid_archive_limit,
        record_ids_by_source=(
            {"dataverse": tuple(args.dataverse_record_ids)} if args.dataverse_record_ids else None
        ),
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
    source = str(args.dataset)
    dataset = RemoteSourceClipIterableDataset(
        splits["test"][source],
        source=source,
        split="test",
        seed=int(args.seed),
        rank=int(context.rank),
        world_size=int(context.world_size),
        client_factory=client_factory,
        clip_loader=RemoteClipLoader(decode),
        max_retries=int(args.max_retries),
    )
    test = {
        source: DataLoader(
            ResilientRemoteSourceClipIterableDataset(dataset),
            batch_size=1,
            num_workers=int(args.num_workers),
            collate_fn=collate_benchmark_items,
            pin_memory=str(context.device).startswith("cuda"),
            prefetch_factor=(int(args.prefetch_factor) if int(args.num_workers) else None),
            persistent_workers=int(args.num_workers) > 0,
            multiprocessing_context="spawn" if int(args.num_workers) else None,
        )
    }
    return cast(_Loaders, _InferenceLoaders(test))


def _default_monitor_factory(device: Any) -> _Monitor:
    from tardis.utils.resources import ResourceMonitor, sample_resources

    return cast(_Monitor, ResourceMonitor(sample_fn=lambda: sample_resources(device)))


@dataclass(frozen=True, slots=True)
class InferServices:
    """Injection boundaries for distributed inference tests."""

    context_factory: Callable[[str | None], _Context] = _default_context_factory
    runtime_builder: Callable[..., _Runtime] = _default_runtime_builder
    loader_builder: Callable[[argparse.Namespace, _Context], _Loaders] = _default_loader_builder
    monitor_factory: Callable[[Any], _Monitor] = _default_monitor_factory


def build_parser() -> argparse.ArgumentParser:
    """Build the infer parser without importing torch or model-weight libraries."""

    data = DataOptions()
    infer = InferOptions()
    parser = argparse.ArgumentParser(description="Evaluate TARDIS on one selected test split")
    local_data = parser.add_argument_group("local data")
    local_data.add_argument("--dataset", choices=DATASET_CHOICES, default=data.dataset)
    local_data.add_argument("--datasets-file", type=Path, default=data.datasets_file)
    local_data.add_argument("--mirror-endpoint", default=data.mirror_endpoint)
    local_data.add_argument("--validation-size", type=int, default=data.validation_size)
    local_data.add_argument("--test-size", type=int, default=data.test_size)
    local_data.add_argument("--split-seed", type=int, default=data.split_seed)
    local_data.add_argument("--num-workers", type=int, default=data.num_workers)
    local_data.add_argument("--prefetch-factor", type=int, default=data.prefetch_factor)
    local_data.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=data.request_timeout_seconds,
    )
    local_data.add_argument("--max-retries", type=int, default=data.max_retries)
    local_data.add_argument(
        "--catalog-record-limit",
        type=int,
        default=data.catalog_record_limit,
    )
    local_data.add_argument(
        "--openvid-archive-limit",
        type=int,
        default=data.openvid_archive_limit,
    )
    local_data.add_argument(
        "--dataverse-record-ids",
        type=parse_record_ids_csv,
        default=data.dataverse_record_ids,
    )
    add_model_arguments(parser, include_num_frames=True)
    add_runtime_arguments(parser)
    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--showcase-count", type=int, default=infer.showcase_count)
    evaluation.add_argument(
        "--resume-metrics",
        action=argparse.BooleanOptionalAction,
        default=infer.resume_metrics,
    )
    evaluation.add_argument("--resume-output", type=Path, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_model_arguments(args)
    if int(args.test_size) <= 0 or int(args.validation_size) < 0:
        raise ValueError("test_size must be positive and validation_size cannot be negative")
    if int(args.num_workers) < 0 or int(args.max_retries) < 0:
        raise ValueError("worker and retry counts cannot be negative")
    if int(args.prefetch_factor) <= 0 or float(args.request_timeout_seconds) <= 0:
        raise ValueError("prefetch and timeout values must be positive")
    if int(args.showcase_count) <= 0:
        raise ValueError("showcase_count must be positive")
    return args


def run_inference(
    args: argparse.Namespace,
    *,
    services: InferServices | None = None,
) -> Path:
    """Stream every rank-local test record and reduce only metric state."""

    import torch

    from tardis.utils.manifest import write_json_manifest
    from tardis.utils.random import make_generator, seed_everything

    selected = InferServices() if services is None else services
    requested_device = torch.device(str(args.device))
    context = selected.context_factory(requested_device.type)
    output_dir: Path | None = None
    monitor: _Monitor | None = None
    runtime: _Runtime | None = None
    status = "failed"
    error: BaseException | None = None
    started = time.perf_counter()
    try:
        context.initialize()
        local_args = argparse.Namespace(**vars(args))
        local_args.device = str(context.device)
        seed_everything(
            int(args.seed),
            rank=int(context.rank),
            deterministic=bool(args.deterministic),
        )
        output_dir = _coordinate_output_dir(local_args, context)
        rank_dir = output_dir / f"rank_{int(context.rank):04d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        if context.is_main:
            monitor = selected.monitor_factory(context.device)
            monitor.sample_once()
            monitor.start()

        runtime = selected.runtime_builder(local_args, use_ema=bool(args.use_ema))
        if runtime.checkpoint is None:
            raise RuntimeError("infer requires a loaded checkpoint")
        loaders = selected.loader_builder(local_args, context)
        source = str(local_args.dataset)
        dataset = f"{source}_test"
        if set(loaders.test) != {source}:
            raise ValueError("test loaders must contain exactly the selected dataset")
        model = runtime.model.eval()
        if bool(args.compile_model):
            model = torch.compile(model)

        source_results: dict[str, dict[str, float]] = {}
        suite = runtime.metric_suite
        suite.reset()
        progress = _load_progress(
            rank_dir / f"{dataset}.metrics.pt",
            suite=suite,
            source=source,
            args=local_args,
            context=context,
            checkpoint_sha=str(runtime.checkpoint.sha256),
        )
        _reconcile_journal(rank_dir / "completed.jsonl", progress["records"])
        _exclude_completed_records(loaders.test[source], progress["records"])
        _save_progress(
            rank_dir / f"{dataset}.metrics.pt",
            suite=suite,
            source=source,
            args=local_args,
            context=context,
            checkpoint_sha=str(runtime.checkpoint.sha256),
            records=progress["records"],
        )
        _evaluate_source(
            loader=loaders.test[source],
            model=model,
            suite=suite,
            source=source,
            dataset=dataset,
            args=local_args,
            context=context,
            runtime=runtime,
            rank_dir=rank_dir,
            output_dir=output_dir,
            records=progress["records"],
            make_generator=make_generator,
        )
        suite.all_reduce()
        result = suite.compute()["macro"]
        if context.is_main:
            source_results[dataset] = {name: float(value) for name, value in result.items()}

        context.barrier()
        if context.is_main:
            _write_shared_outputs(
                output_dir,
                source_results=source_results,
                args=local_args,
                context=context,
                runtime=runtime,
                model=model,
                elapsed_seconds=time.perf_counter() - started,
            )
        context.barrier()
        status = "completed"
        return output_dir
    except BaseException as caught:
        error = caught
        status = "interrupted" if isinstance(caught, KeyboardInterrupt) else "failed"
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        if output_dir is not None and context.is_main:
            resources = None if monitor is None else monitor.summary()
            write_json_manifest(
                output_dir / "resources.json",
                {} if resources is None else _resource_payload(resources),
            )
            manifest: dict[str, object] = {
                "status": status,
                "run_id": output_dir.name,
                "world_size": int(context.world_size),
                "settings": _settings(args),
            }
            if runtime is not None and runtime.checkpoint is not None:
                manifest["checkpoint"] = {
                    "path": str(runtime.checkpoint.path),
                    "sha256": str(runtime.checkpoint.sha256),
                    "used_ema": bool(runtime.checkpoint.used_ema),
                }
                manifest["metric_provenance"] = dict(runtime.metric_suite.provenance_ids)
                manifest["dataset_sources"] = list(runtime.dataset_sources)
            if error is not None:
                manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            write_json_manifest(output_dir / "manifest.json", manifest)
        context.close()


def _evaluate_source(
    *,
    loader: Any,
    model: Any,
    suite: _MetricSuite,
    source: str,
    dataset: str,
    args: argparse.Namespace,
    context: _Context,
    runtime: _Runtime,
    rank_dir: Path,
    output_dir: Path,
    records: list[dict[str, object]],
    make_generator: Callable[[int, Any], Any],
) -> None:
    import torch

    from tardis.data.dataset import BenchmarkFailure

    completed = {
        str(record["record_id"])
        for record in records
        if record.get("status") == "completed"
    }
    state_path = rank_dir / f"{dataset}.metrics.pt"
    for batch in loader:
        if isinstance(batch, BenchmarkFailure):
            record_id = batch.record.id
            if record_id in completed:
                continue
            failure_record: dict[str, object] = {
                "dataset": dataset,
                "source": source,
                "record_id": record_id,
                "prompt": batch.record.caption,
                "seed": int(batch.sample_seed),
                "rank": int(context.rank),
                "status": "failed",
                "generation_seconds": 0.0,
                "frame_count": int(args.num_frames),
                "showcase": None,
                "error_type": batch.error_type,
                "error_message": batch.error_message,
            }
            records.append(failure_record)
            completed.add(record_id)
            _save_progress(
                state_path,
                suite=suite,
                source=source,
                args=args,
                context=context,
                checkpoint_sha=str(runtime.checkpoint.sha256),
                records=records,
            )
            _append_jsonl(rank_dir / "completed.jsonl", failure_record)
            continue
        batch_size = len(batch.record_ids)
        if batch.video.shape[0] != batch_size or len(batch.prompts) != batch_size:
            raise ValueError("inference batch fields must have the same batch size")
        for index, record_id in enumerate(batch.record_ids):
            if record_id in completed:
                continue
            prompt = str(batch.prompts[index])
            sample_seed = int(batch.sample_seeds[index])
            reference = batch.video[index].to(runtime.device)
            before = suite.state_dict()
            generated: Any = None
            generated_video: Any | None = None
            metric_video: Any | None = None
            started = time.perf_counter()
            try:
                generator = make_generator(sample_seed, runtime.device)
                _synchronize(runtime.device)
                with torch.inference_mode():
                    generated = model.generate(
                        [prompt],
                        num_frames=int(args.num_frames),
                        fps=int(args.fps),
                        generator=generator,
                    )
                _synchronize(runtime.device)
                generation_seconds = time.perf_counter() - started
                generated_video = validate_generated_video(
                    generated.video,
                    batch_size=1,
                    num_frames=int(args.num_frames),
                    height=int(args.height),
                    width=int(args.width),
                )
                metric_video = generated_video[0] if generated_video.ndim == 5 else generated_video
                suite.update(metric_video, reference, prompt)
                record: dict[str, object] = {
                    "dataset": dataset,
                    "source": source,
                    "record_id": str(record_id),
                    "prompt": prompt,
                    "seed": sample_seed,
                    "rank": int(context.rank),
                    "status": "completed",
                    "generation_seconds": generation_seconds,
                    "frame_count": int(args.num_frames),
                    "showcase": None,
                    "error_type": None,
                    "error_message": None,
                }
            except torch.cuda.OutOfMemoryError:
                suite.load_state_dict(before)
                raise
            except Exception as failure:
                suite.load_state_dict(before)
                record = {
                    "dataset": dataset,
                    "source": source,
                    "record_id": str(record_id),
                    "prompt": prompt,
                    "seed": sample_seed,
                    "rank": int(context.rank),
                    "status": "failed",
                    "generation_seconds": time.perf_counter() - started,
                    "frame_count": int(args.num_frames),
                    "showcase": None,
                    "error_type": type(failure).__name__,
                    "error_message": str(failure),
                }
            records.append(record)
            completed.add(str(record_id))
            _save_progress(
                state_path,
                suite=suite,
                source=source,
                args=args,
                context=context,
                checkpoint_sha=str(runtime.checkpoint.sha256),
                records=records,
            )
            _append_jsonl(rank_dir / "completed.jsonl", record)
            del generated, generated_video, metric_video, reference


def _exclude_completed_records(loader: Any, records: Sequence[Mapping[str, object]]) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return
    for candidate in (dataset, getattr(dataset, "dataset", None)):
        if candidate is not None and hasattr(candidate, "exclude_record_ids"):
            completed = {
                str(record["record_id"])
                for record in records
                if record.get("status") == "completed"
            }
            candidate.exclude_record_ids(completed)
            return


def _load_progress(
    path: Path,
    *,
    suite: _MetricSuite,
    source: str,
    args: argparse.Namespace,
    context: _Context,
    checkpoint_sha: str,
) -> dict[str, Any]:
    import torch

    if not path.is_file():
        return {"records": []}
    if not bool(args.resume_metrics):
        raise FileExistsError(f"metric state already exists and resume is disabled: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "version": _STATE_VERSION,
        "source": source,
        "rank": int(context.rank),
        "world_size": int(context.world_size),
        "checkpoint_sha256": checkpoint_sha,
        "settings": _settings(args),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"incompatible infer resume state field: {key}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or any(not isinstance(item, dict) for item in raw_records):
        raise ValueError("infer resume records must be a list of objects")
    record_ids = [str(item.get("record_id")) for item in raw_records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("infer resume state contains duplicate record IDs")
    metric_state = payload.get("metric_suite")
    if not isinstance(metric_state, Mapping):
        raise ValueError("infer resume metric_suite must be a mapping")
    statuses = {str(item.get("status")) for item in raw_records}
    if not statuses <= {"completed", "failed"}:
        raise ValueError("infer resume records contain an invalid status")
    suite.load_state_dict(metric_state)
    return {
        "records": [dict(item) for item in raw_records if item.get("status") == "completed"]
    }


def _save_progress(
    path: Path,
    *,
    suite: _MetricSuite,
    source: str,
    args: argparse.Namespace,
    context: _Context,
    checkpoint_sha: str,
    records: list[dict[str, object]],
) -> None:
    from tardis.utils.checkpoint import atomic_torch_save

    atomic_torch_save(
        {
            "version": _STATE_VERSION,
            "source": source,
            "rank": int(context.rank),
            "world_size": int(context.world_size),
            "checkpoint_sha256": checkpoint_sha,
            "settings": _settings(args),
            "completed_ids": sorted(
                str(record["record_id"])
                for record in records
                if record.get("status") == "completed"
            ),
            "records": records,
            "metric_suite": suite.state_dict(),
        },
        path,
    )


def _write_shared_outputs(
    output_dir: Path,
    *,
    source_results: Mapping[str, Mapping[str, float]],
    args: argparse.Namespace,
    context: _Context,
    runtime: _Runtime,
    model: Any,
    elapsed_seconds: float,
) -> None:
    from tardis.metrics.report import SingleDatasetMetricReport
    from tardis.utils.manifest import write_json_manifest

    source = str(args.dataset)
    dataset = f"{source}_test"
    records = _collect_rank_records(output_dir, int(context.world_size), dataset=dataset)
    source_records = [record for record in records if record["source"] == source]
    if len(source_records) != int(args.test_size):
        raise RuntimeError(
            f"{dataset} completed ledger has {len(source_records)} records; "
            f"expected {int(args.test_size)}"
        )
    if set(source_results) != {dataset}:
        raise ValueError("metric results must contain exactly the selected test dataset")
    report = SingleDatasetMetricReport.from_mapping(dataset, source_results[dataset])
    report.write_csv(output_dir / "metrics.csv")
    report.write_xlsx(output_dir / "metrics.xlsx")
    showcase_selection = _write_random_showcases(
        output_dir,
        records=records,
        model=model,
        runtime=runtime,
        args=args,
    )
    _write_details(output_dir, records)
    showcases = list((output_dir / "showcases").glob("*.mp4"))
    allocation = {source: len(showcases)}
    if len(showcases) != int(args.showcase_count):
        raise RuntimeError(
            f"showcase output must contain exactly {int(args.showcase_count)} videos; "
            f"got {len(showcases)}"
        )
    latencies = [
        float(cast(float, record["generation_seconds"]))
        for record in records
        if record["status"] == "completed"
    ]
    write_json_manifest(
        output_dir / "latency.json",
        {
            "successful_video_count": len(latencies),
            "total_wall_seconds": elapsed_seconds,
            "mean_generation_seconds": fmean(latencies) if latencies else 0.0,
            "p50_generation_seconds": _percentile(latencies, 0.50),
            "p95_generation_seconds": _percentile(latencies, 0.95),
            "max_generation_seconds": max(latencies, default=0.0),
            "mean_seconds_per_frame": (
                fmean(latencies) / int(args.num_frames) if latencies else 0.0
            ),
        },
    )
    write_json_manifest(
        output_dir / "result_manifest.json",
        {
            "checkpoint": {
                "path": str(runtime.checkpoint.path),
                "sha256": str(runtime.checkpoint.sha256),
                "used_ema": bool(runtime.checkpoint.used_ema),
            },
            "datasets": {
                source: {
                    "expected": int(args.test_size),
                    "completed": sum(
                        record["source"] == source and record["status"] == "completed"
                        for record in records
                    ),
                    "failed": sum(
                        record["source"] == source and record["status"] == "failed"
                        for record in records
                    ),
                }
            },
            "metric_provenance": dict(runtime.metric_suite.provenance_ids),
            "showcase_allocation": allocation,
            "showcase_selection": showcase_selection,
        },
    )


def _select_showcase_records(
    records: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    """Select a reproducible unique showcase set from one evaluated source."""

    if count <= 0:
        raise ValueError("showcase count must be positive")
    completed = sorted(
        (dict(record) for record in records if record.get("status") == "completed"),
        key=lambda record: (str(record["source"]), str(record["record_id"])),
    )
    if len(completed) < count:
        raise RuntimeError(
            f"cannot select {count} showcases from {len(completed)} successful test records"
        )
    rng = random.Random(seed)
    return rng.sample(completed, count)


def _write_random_showcases(
    output_dir: Path,
    *,
    records: list[dict[str, object]],
    model: Any,
    runtime: _Runtime,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Regenerate only the five seeded-random test prompts retained as MP4 artifacts."""

    import torch

    from tardis.utils.random import make_generator
    from tardis.utils.video_io import write_mp4

    selected = _select_showcase_records(
        records,
        count=int(args.showcase_count),
        seed=int(args.seed),
    )
    showcase_dir = output_dir / "showcases"
    showcase_dir.mkdir(parents=True, exist_ok=True)
    for stale in showcase_dir.glob("*.mp4"):
        stale.unlink()
    selection: list[dict[str, object]] = []
    for index, selected_record in enumerate(selected):
        source = str(selected_record["source"])
        record_id = str(selected_record["record_id"])
        prompt = str(selected_record["prompt"])
        sample_seed = int(cast(int, selected_record["seed"]))
        generator = make_generator(sample_seed, runtime.device)
        _synchronize(runtime.device)
        with torch.inference_mode():
            generated = model.generate(
                [prompt],
                num_frames=int(args.num_frames),
                fps=int(args.fps),
                generator=generator,
            )
        _synchronize(runtime.device)
        name = f"{source}__{index:02d}__{_safe_filename(record_id)}.mp4"
        path = showcase_dir / name
        generated_video = validate_generated_video(
            generated.video,
            batch_size=1,
            num_frames=int(args.num_frames),
            height=int(args.height),
            width=int(args.width),
        )
        write_mp4(generated_video, path, fps=float(args.fps))
        relative = str(path.relative_to(output_dir))
        selected_record["showcase"] = relative
        original = next(
            record
            for record in records
            if record["source"] == source and record["record_id"] == record_id
        )
        original["showcase"] = relative
        selection.append(
            {
                "source": source,
                "record_id": record_id,
                "prompt": prompt,
                "seed": sample_seed,
                "path": relative,
            }
        )
    return selection


def _collect_rank_records(
    output_dir: Path,
    world_size: int,
    *,
    dataset: str,
) -> list[dict[str, object]]:
    import torch

    records: list[dict[str, object]] = []
    for rank in range(world_size):
        rank_dir = output_dir / f"rank_{rank:04d}"
        path = rank_dir / f"{dataset}.metrics.pt"
        if not path.is_file():
            raise RuntimeError(f"missing rank-local metric state: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError(f"invalid rank-local records in {path}")
        records.extend(dict(item) for item in raw_records)
    identities = [(record["source"], record["record_id"]) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("rank-local inference states contain duplicate source record IDs")
    return sorted(records, key=lambda item: (str(item["source"]), str(item["record_id"])))


def _write_details(output_dir: Path, records: list[dict[str, object]]) -> None:
    fieldnames = (
        "dataset",
        "source",
        "record_id",
        "prompt",
        "seed",
        "rank",
        "status",
        "generation_seconds",
        "frame_count",
        "showcase",
        "error_type",
        "error_message",
    )

    def csv_payload(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    _atomic_text_write(output_dir / "per_video_details.csv", csv_payload, newline="")
    _atomic_jsonl_write(output_dir / "per_video_details.jsonl", records)
    failures = [record for record in records if record["status"] == "failed"]
    _atomic_jsonl_write(output_dir / "failures.jsonl", failures)


def _settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset": str(args.dataset),
        "test_size": int(args.test_size),
        "validation_size": int(args.validation_size),
        "num_frames": int(args.num_frames),
        "fps": int(args.fps),
        "height": int(args.height),
        "width": int(args.width),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "use_ema": bool(args.use_ema),
        "precision": str(args.precision),
        "compile_model": bool(args.compile_model),
        "deterministic": bool(args.deterministic),
        "pretrained_model": str(args.pretrained_model),
        "latent_channels": int(args.latent_channels),
        "patch_size": int(args.patch_size),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "active_ratio": float(args.active_ratio),
        "transport_quotient": bool(args.transport_quotient),
        "quotient_regularization": float(args.quotient_regularization),
        "quotient_rank_threshold": float(args.quotient_rank_threshold),
        "innovation_proper_time": bool(args.innovation_proper_time),
        "proper_time_maximum_hazard": float(args.proper_time_maximum_hazard),
        "showcase_count": int(args.showcase_count),
    }


def _resource_payload(summary: _ResourceSummary) -> dict[str, int | float]:
    return {
        "sample_count": int(summary.sample_count),
        "peak_allocated_mb": float(summary.peak_allocated_mb),
        "peak_reserved_mb": float(summary.peak_reserved_mb),
        "peak_process_rss_mb": float(summary.peak_process_rss_mb),
        "mean_gpu_utilization_percent": float(summary.mean_gpu_utilization_percent),
    }


def _coordinate_output_dir(args: argparse.Namespace, context: _Context) -> Path:
    import torch.distributed as dist

    from tardis.utils.manifest import create_output_run_dir

    output: Path | None = None
    if context.is_main:
        if args.resume_output is None:
            output = create_output_run_dir(
                Path(args.output_root),
                f"infer/{args.dataset}",
            )
        else:
            output = Path(args.resume_output).expanduser().resolve()
            if not output.is_dir():
                raise FileNotFoundError(f"resume output directory does not exist: {output}")
            if output.parent.name != str(args.dataset):
                raise ValueError(f"resume output must belong to dataset {args.dataset!r}: {output}")
    if int(context.world_size) == 1:
        if output is None:
            raise RuntimeError("rank zero did not allocate an infer output directory")
        return output
    payload: list[object] = [None if output is None else str(output)]
    dist.broadcast_object_list(payload, src=0)
    if not isinstance(payload[0], str):
        raise RuntimeError("rank zero broadcast an invalid infer output directory")
    return Path(payload[0])


def _reconcile_journal(path: Path, records: list[dict[str, object]]) -> None:
    if path.is_file():
        data = path.read_bytes()
        valid_end = 0
        tail_truncated = False
        for raw_line in data.splitlines(keepends=True):
            terminated = raw_line.endswith(b"\n")
            line = raw_line[:-1] if terminated else raw_line
            if not line.strip():
                valid_end += len(raw_line)
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                is_final_line = valid_end + len(raw_line) == len(data)
                if terminated or not is_final_line:
                    raise
                with path.open("r+b") as handle:
                    handle.truncate(valid_end)
                    handle.flush()
                    os.fsync(handle.fileno())
                tail_truncated = True
                break
            if not isinstance(payload, dict):
                raise ValueError("infer journal entries must be objects")
            str(payload["source"]), str(payload["record_id"])
            valid_end += len(raw_line)
        if data and not data.endswith(b"\n") and not tail_truncated:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    if path.is_file() or records:
        _atomic_jsonl_write(path, records)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_jsonl_write(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    def payload(handle: Any) -> None:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    _atomic_text_write(path, payload)


def _atomic_text_write(
    path: Path,
    writer: Callable[[Any], None],
    *,
    newline: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
            newline=newline,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _safe_filename(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value)
    return normalized[:80] or "record"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _synchronize(device: Any) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: Sequence[str] | None = None) -> int:
    run_inference(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
