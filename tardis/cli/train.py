"""Typed assembly and validation primitives for production training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import signal
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any, Literal, Protocol, TextIO, cast, overload

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tardis.cli.common import parse_args
from tardis.data.assembly import (
    RangeClientFactory,
    RemoteDataLoaderOptions,
    RemoteDataLoaders,
    build_remote_dataloaders,
)
from tardis.data.dataset import ClipBatch, ClipDecodeOptions
from tardis.data.splits import StablePartition
from tardis.metrics.suite import MetricSuite
from tardis.models.tardis import TARDISModel, TARDISTrainingBatch
from tardis.training.curriculum import CurriculumSchedule
from tardis.training.engine import (
    ModelEMA,
    TrainEngine,
    TrainEngineOptions,
    TrainStepResult,
    ema_parameter_map,
)
from tardis.training.validation import (
    ValidationCheckpointSelector,
    ValidationMetric,
    ValidationScore,
    score_validation_event,
    selection_scales_for_source,
)
from tardis.utils.checkpoint import atomic_torch_save, load_checkpoint
from tardis.utils.distributed import DistributedContext
from tardis.utils.manifest import RunPaths, create_run_paths, write_json_manifest
from tardis.utils.random import effective_seed, make_generator, seed_everything
from tardis.utils.resources import ResourceMonitor, ResourceSummary, sample_resources

_VALIDATION_SOURCES = ("dataverse", "openvid", "seedance")
_SOURCE_DIRECTORIES = {
    "dataverse": "Vchitect_T2V_DataVerse",
    "openvid": "OpenVid-1M",
    "seedance": "seedance-2-prompts-datasets",
}


class RemoteLoaderBuilder[LoaderResult](Protocol):
    """Injectable boundary around remote loader assembly."""

    def __call__(
        self,
        *,
        partition: StablePartition,
        train_clip_options: ClipDecodeOptions,
        evaluation_clip_options: ClipDecodeOptions,
        loader_options: RemoteDataLoaderOptions,
        rank: int,
        world_size: int,
        endpoint: str,
        client_factory: RangeClientFactory,
        dataset_roots: Mapping[str, Path | str],
        catalog_record_limit: int | None,
        openvid_archive_limit: int | None,
        record_ids_by_source: Mapping[str, Sequence[str]] | None,
        selected_source: str,
    ) -> LoaderResult: ...


class _ValidationDataset(Protocol):
    split: str
    source: str
    records: Sequence[_ValidationRecord]
    rank: int
    world_size: int

    def __len__(self) -> int: ...


class _ValidationRecord(Protocol):
    id: str


class _ValidationLoader(Protocol):
    dataset: _ValidationDataset

    def __iter__(self) -> Iterator[ClipBatch]: ...


class ValidationProgress(Protocol):
    """Progress sink for one combined, source-aware validation bar."""

    def update_validation(self, records: int, source: str) -> None: ...


class _WarmStartIdentity(Protocol):
    @property
    def path(self) -> Path: ...

    @property
    def sha256(self) -> str: ...

    @property
    def used_ema(self) -> bool: ...


class WarmStartLoader(Protocol):
    def __call__(
        self,
        model: TARDISModel,
        checkpoint: Path | str,
        *,
        use_ema: bool,
    ) -> _WarmStartIdentity: ...


@dataclass(frozen=True, slots=True)
class _ValidatedLoader:
    loader: _ValidationLoader
    expected_record_ids: frozenset[str]


def curriculum_durations(total_optimizer_steps: int) -> tuple[int, int, int, int, int, int]:
    """Allocate most optimization to causal and metric-aligned stages.

    The six weights are 5/5/10/20/20/40 percent. Every stage receives one
    optimizer step first, then largest-remainder apportionment preserves the
    exact user-supplied optimization budget.
    """

    if total_optimizer_steps < 6:
        raise ValueError("total_optimizer_steps must be at least 6")
    weights = (5, 5, 10, 20, 20, 40)
    remaining = total_optimizer_steps - len(weights)
    numerators = tuple(remaining * weight for weight in weights)
    allocations = [numerator // sum(weights) for numerator in numerators]
    leftover = remaining - sum(allocations)
    priority = sorted(
        range(len(weights)),
        key=lambda index: (numerators[index] % sum(weights), weights[index], index),
        reverse=True,
    )
    for index in priority[:leftover]:
        allocations[index] += 1
    durations = tuple(1 + allocation for allocation in allocations)
    return (
        durations[0],
        durations[1],
        durations[2],
        durations[3],
        durations[4],
        durations[5],
    )


def curriculum_durations_for_profile(
    total_optimizer_steps: int,
    profile: str,
) -> tuple[int, int, int, int, int, int]:
    """Build full, transport-only, or metric-dominant probe curricula."""

    if profile == "full":
        return curriculum_durations(total_optimizer_steps)
    if profile == "transport":
        if total_optimizer_steps <= 0:
            raise ValueError("total_optimizer_steps must be positive")
        return (total_optimizer_steps, 1, 1, 1, 1, 1)
    if profile == "metric_alignment":
        if total_optimizer_steps < 6:
            raise ValueError(
                "metric_alignment curriculum requires at least 6 optimizer steps"
            )
        return (1, 1, 1, 1, 1, total_optimizer_steps - 5)
    if profile == "closed_loop_motion":
        if total_optimizer_steps < 6:
            raise ValueError(
                "closed_loop_motion curriculum requires at least 6 optimizer steps"
            )
        return (1, 1, 1, total_optimizer_steps - 5, 1, 1)
    raise ValueError(f"unknown curriculum profile: {profile!r}")


def train_engine_options_from_args(args: argparse.Namespace) -> TrainEngineOptions:
    """Map CLI fields while keeping accumulation inside each complete epoch."""

    epochs = _positive_int(args.epochs, "epochs")
    if args.steps_per_epoch is None:
        raise ValueError("steps_per_epoch must be resolved from the training loader")
    steps_per_epoch = _positive_int(args.steps_per_epoch, "steps_per_epoch")
    accumulation = _positive_int(
        args.gradient_accumulation_steps,
        "gradient_accumulation_steps",
    )
    if steps_per_epoch % accumulation:
        raise ValueError(
            "steps_per_epoch must be divisible by gradient_accumulation_steps; "
            "cross-epoch accumulation is not supported"
        )
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accumulation)
    return TrainEngineOptions(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        gradient_accumulation_steps=accumulation,
        gradient_clip_norm=float(args.gradient_clip_norm),
        warmup_steps=int(args.warmup_steps),
        total_optimizer_steps=epochs * optimizer_steps_per_epoch,
        precision=str(args.precision),
        ema_decay=float(args.ema_decay),
    )


@overload
def build_train_dataloaders(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    builder: None = None,
) -> RemoteDataLoaders: ...


@overload
def build_train_dataloaders[LoaderResult](
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    builder: RemoteLoaderBuilder[LoaderResult],
) -> LoaderResult: ...


def build_train_dataloaders(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    builder: RemoteLoaderBuilder[Any] | None = None,
) -> Any:
    """Construct local train and source-local benchmark loaders."""

    from tardis.cli.runtime import read_dataset_sources
    from tardis.data.catalog import normalize_local_dataset_roots

    world_size = _positive_int(context.world_size, "world_size")
    micro_batch_size = _positive_int(args.micro_batch_size, "micro_batch_size")
    timeout_seconds = float(args.request_timeout_seconds)
    max_retries = int(args.max_retries)
    train_clip_options = ClipDecodeOptions(
        num_frames=(1 if str(args.train_mode) == "keyframe_only" else int(args.num_frames)),
        height=int(args.height),
        width=int(args.width),
        mode="train",
        timeout_seconds=timeout_seconds,
        random_flip=True,
    )
    evaluation_clip_options = ClipDecodeOptions(
        num_frames=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
        mode="benchmark",
        timeout_seconds=timeout_seconds,
        random_flip=False,
    )
    client_factory = RangeClientFactory(
        max_object_bytes=max(
            train_clip_options.max_media_bytes,
            evaluation_clip_options.max_media_bytes,
        ),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    loader_options = RemoteDataLoaderOptions(
        steps_per_epoch=(None if args.steps_per_epoch is None else int(args.steps_per_epoch)),
        global_batch_size=micro_batch_size * world_size,
        evaluation_batch_size=_positive_int(
            args.validation_batch_size,
            "validation_batch_size",
        ),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        max_sample_retries=max_retries,
        multiprocessing_context="spawn",
    )
    selected_builder: RemoteLoaderBuilder[Any] = (
        build_remote_dataloaders if builder is None else builder
    )
    dataset_roots = normalize_local_dataset_roots(
        read_dataset_sources(args.datasets_file, args.mirror_endpoint)
    )
    return selected_builder(
        partition=StablePartition(
            seed=int(args.split_seed),
            validation_size=int(args.validation_size),
            test_size=int(args.test_size),
            group_by_caption=str(args.dataset) == "seedance",
        ),
        train_clip_options=train_clip_options,
        evaluation_clip_options=evaluation_clip_options,
        loader_options=loader_options,
        rank=int(context.rank),
        world_size=world_size,
        endpoint=str(args.mirror_endpoint),
        client_factory=client_factory,
        dataset_roots=dataset_roots,
        catalog_record_limit=args.catalog_record_limit,
        openvid_archive_limit=args.openvid_archive_limit,
        record_ids_by_source=(
            {"dataverse": tuple(args.dataverse_record_ids)} if args.dataverse_record_ids else None
        ),
        selected_source=str(args.dataset),
    )


def clip_batch_to_training_batch(
    batch: ClipBatch,
    device: torch.device | str,
) -> tuple[TARDISTrainingBatch, tuple[str, ...]]:
    """Move one decoded microbatch while leaving prompts and ledger IDs untouched."""

    target = torch.device(device)
    return (
        TARDISTrainingBatch(
            prompts=batch.prompts,
            video=batch.video.to(device=target, non_blocking=True),
        ),
        batch.record_ids,
    )


@contextmanager
def ema_temporal_parameters(model: nn.Module, ema: ModelEMA) -> Iterator[None]:
    """Temporarily overlay validated EMA values on trainable temporal parameters."""

    temporal = ema_parameter_map(model)
    expected_names = set(temporal)
    shadow_names = set(ema.shadow)
    if shadow_names != expected_names:
        missing = sorted(expected_names - shadow_names)
        unexpected = sorted(shadow_names - expected_names)
        raise ValueError(
            f"EMA temporal parameter names mismatch; missing={missing}, extra={unexpected}"
        )
    for name, parameter in temporal.items():
        shadow = ema.shadow[name]
        if shadow.shape != parameter.shape:
            raise ValueError(f"EMA tensor shape mismatch for {name!r}")

    snapshot = {name: parameter.detach().clone() for name, parameter in temporal.items()}
    try:
        with torch.no_grad():
            for name, parameter in temporal.items():
                parameter.copy_(ema.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
        yield
    finally:
        with torch.no_grad():
            for name, parameter in temporal.items():
                parameter.copy_(snapshot[name])


@contextmanager
def validation_generation_autocast(
    model: nn.Module,
    device: torch.device,
) -> Iterator[None]:
    """Bridge compact frozen priors to FP32 temporal master weights."""

    priors = getattr(model, "priors", None)
    prior_dtype: torch.dtype | None = None
    if isinstance(priors, nn.Module):
        prior_dtype = next(
            (parameter.dtype for parameter in priors.parameters() if parameter.is_floating_point()),
            None,
        )
    enabled = prior_dtype in {torch.float16, torch.bfloat16} and (
        device.type == "cuda" or prior_dtype == torch.bfloat16
    )
    if not enabled or prior_dtype is None:
        yield
        return
    with torch.autocast(device_type=device.type, dtype=prior_dtype):
        yield


def evaluate_validation(
    model: nn.Module,
    loaders: RemoteDataLoaders,
    metric_suite: MetricSuite,
    fps: int,
    device: torch.device | str,
    seed: int,
    *,
    stop: CooperativeStop | None = None,
    context: DistributedContext | None = None,
    progress: ValidationProgress | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate exactly one selected dataset's validation stream."""

    validation_loaders = _validated_validation_loaders(loaders)
    if (stop is None) != (context is None):
        raise ValueError("validation stop and distributed context must be provided together")
    if fps <= 0:
        raise ValueError("validation fps must be positive")
    if seed < 0:
        raise ValueError("validation seed must be non-negative")

    target_device = torch.device(device)
    was_training = model.training
    results: dict[str, dict[str, float]] = {}
    model.eval()
    try:
        with torch.no_grad():
            for source, validated in validation_loaders.items():
                loader = validated.loader
                expected_record_ids = validated.expected_record_ids
                expected_records = len(expected_record_ids)
                observed_records = 0
                seen_record_ids: set[str] = set()
                repeated_record_ids: set[str] = set()
                try:
                    iterator = iter(loader)
                    while True:
                        batch = None
                        if stop is None or not stop.requested:
                            batch = next(iterator, None)
                        if batch is not None:
                            _validate_validation_batch(batch, source)
                            observed_records += len(batch.prompts)
                            batch_record_ids: set[str] = set()
                            for record_id in batch.record_ids:
                                if record_id in batch_record_ids or record_id in seen_record_ids:
                                    repeated_record_ids.add(record_id)
                                batch_record_ids.add(record_id)
                            unexpected_record_ids = batch_record_ids - expected_record_ids
                            if unexpected_record_ids:
                                raise ValueError(
                                    f"validation loader {source!r} yielded non-canonical record "
                                    f"IDs: {sorted(unexpected_record_ids)}"
                                )
                            seen_record_ids.update(batch_record_ids)
                            if isinstance(model, TARDISModel):
                                references = batch.video.to(
                                    device=target_device,
                                    non_blocking=True,
                                )
                                generators = tuple(
                                    torch.Generator(device=target_device).manual_seed(seed)
                                    for seed in batch.sample_seeds
                                )
                                with validation_generation_autocast(model, target_device):
                                    generated_batch = model.generate(
                                        list(batch.prompts),
                                        num_frames=int(references.shape[1]),
                                        fps=fps,
                                        generator=generators,
                                    ).video
                                if generated_batch.ndim != 5 or generated_batch.shape[0] != len(
                                    batch.prompts
                                ):
                                    raise RuntimeError(
                                        "validation generation must return video [B,T,3,H,W]"
                                    )
                                for index, prompt in enumerate(batch.prompts):
                                    metric_suite.update(
                                        generated_batch[index],
                                        references[index],
                                        prompt,
                                    )
                            else:
                                generate = getattr(model, "generate", None)
                                if not callable(generate):
                                    raise TypeError("validation model must expose generate()")
                                for index, prompt in enumerate(batch.prompts):
                                    if stop is not None and stop.requested:
                                        break
                                    reference = batch.video[index].to(
                                        device=target_device,
                                        non_blocking=True,
                                    )
                                    generator = torch.Generator(device=target_device).manual_seed(
                                        batch.sample_seeds[index]
                                    )
                                    with validation_generation_autocast(model, target_device):
                                        generated = generate(
                                            [prompt],
                                            num_frames=int(reference.shape[0]),
                                            fps=fps,
                                            generator=generator,
                                        ).video
                                    if generated.ndim != 5 or generated.shape[0] != 1:
                                        raise RuntimeError(
                                            "validation generation must return video [1,T,3,H,W]"
                                        )
                                    metric_suite.update(generated[0], reference, prompt)
                        cancelled, any_rank_has_batch, global_records = (
                            _synchronize_validation_round(
                                stop,
                                context,
                                local_has_batch=batch is not None,
                                local_records=0 if batch is None else len(batch.prompts),
                            )
                        )
                        if cancelled:
                            raise _ValidationInterrupted(
                                "validation interrupted at a synchronized batch boundary"
                            )
                        if progress is not None and global_records:
                            progress.update_validation(global_records, source)
                        if not any_rank_has_batch:
                            break
                    if observed_records != expected_records:
                        raise ValueError(
                            f"validation loader {source!r} expected {expected_records} records "
                            f"but observed {observed_records}"
                        )
                    if repeated_record_ids:
                        raise ValueError(
                            f"validation loader {source!r} repeated record IDs: "
                            f"{sorted(repeated_record_ids)}"
                        )
                    if seen_record_ids != expected_record_ids:
                        missing = sorted(expected_record_ids - seen_record_ids)
                        unexpected = sorted(seen_record_ids - expected_record_ids)
                        raise ValueError(
                            f"validation loader {source!r} did not consume its canonical shard; "
                            f"missing={missing}, unexpected={unexpected}"
                        )
                    metric_suite.all_reduce()
                    results[f"{source}_validation"] = dict(metric_suite.compute()["macro"])
                finally:
                    metric_suite.reset()
    finally:
        model.train(was_training)
    return results


def _validated_validation_loaders(loaders: RemoteDataLoaders) -> dict[str, _ValidatedLoader]:
    if not isinstance(loaders, RemoteDataLoaders):
        raise TypeError("evaluate_validation requires a RemoteDataLoaders bundle")
    selected_sources = tuple(loaders.validation)
    if len(selected_sources) != 1 or selected_sources[0] not in _VALIDATION_SOURCES:
        raise ValueError("validation loaders must contain exactly one canonical dataset")
    validation_splits = loaders.splits.get("validation")
    if not isinstance(validation_splits, Mapping) or set(validation_splits) != set(
        selected_sources
    ):
        raise ValueError("canonical validation split must contain exactly the selected dataset")

    validated: dict[str, _ValidatedLoader] = {}
    for source in selected_sources:
        raw_loader = loaders.validation[source]
        dataset = getattr(raw_loader, "dataset", None)
        if dataset is None:
            raise TypeError(f"validation loader {source!r} must expose its dataset")
        if getattr(dataset, "split", None) != "validation":
            raise ValueError(f"validation loader {source!r} dataset split must be 'validation'")
        if getattr(dataset, "source", None) != source:
            raise ValueError(f"validation loader {source!r} dataset source must equal {source!r}")
        dataset_records = getattr(dataset, "records", None)
        canonical_records = tuple(validation_splits[source])
        if dataset_records is None or tuple(dataset_records) != canonical_records:
            raise ValueError(
                f"validation loader {source!r} must contain exactly its canonical "
                "validation records"
            )
        rank = getattr(dataset, "rank", None)
        world_size = getattr(dataset, "world_size", None)
        if (
            not isinstance(rank, int)
            or not isinstance(world_size, int)
            or world_size <= 0
            or not 0 <= rank < world_size
        ):
            raise ValueError(f"validation loader {source!r} has invalid distributed geometry")
        canonical_ids = tuple(str(record.id) for record in canonical_records)
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError(f"canonical validation split {source!r} contains duplicate record IDs")
        expected_record_ids = frozenset(canonical_ids[rank::world_size])
        try:
            expected_records = len(dataset)
        except TypeError as error:
            raise TypeError(
                f"validation loader {source!r} dataset must have a local length"
            ) from error
        if expected_records < 0:
            raise ValueError(f"validation loader {source!r} dataset length cannot be negative")
        if expected_records != len(expected_record_ids):
            raise ValueError(
                f"validation loader {source!r} length does not match its canonical rank shard"
            )
        validated[source] = _ValidatedLoader(
            loader=cast(_ValidationLoader, raw_loader),
            expected_record_ids=expected_record_ids,
        )
    return validated


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _validate_validation_batch(batch: ClipBatch, source: str) -> None:
    batch_size = len(batch.prompts)
    if batch.video.ndim != 5 or batch.video.shape[0] != batch_size:
        raise ValueError("validation ClipBatch video and prompts must have matching batch size")
    if not (len(batch.sources) == len(batch.record_ids) == len(batch.sample_seeds) == batch_size):
        raise ValueError("validation ClipBatch metadata must match its batch size")
    if any(batch_source != source for batch_source in batch.sources):
        raise ValueError(f"validation loader {source!r} contains a foreign source")


RunStatus = Literal["running", "completed", "interrupted", "failed"]
ValidationRunner = Callable[..., dict[str, dict[str, float]]]
CheckpointWriter = Callable[[Mapping[str, Any], Path], None]
DDPFactory = Callable[..., nn.Module]


class _ValidationInterrupted(RuntimeError):
    """Raised on every rank after a synchronized cooperative validation stop."""


class TrainLoopRecorder(Protocol):
    """Rank-zero event and manifest sink used by the injectable epoch loop."""

    def record(self, event: Mapping[str, object]) -> None: ...

    def update(self, patch: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainLoopResult:
    status: Literal["completed", "interrupted"]
    position: tuple[int, int]
    microbatches_completed: int
    optimizer_steps_completed: int
    validation_epochs: tuple[int, ...]
    stop_reason: str | None


class TrainProgress(Protocol):
    """Exactly two progress surfaces per epoch: train and selected validation."""

    def start_epoch(self, epoch: int, epochs: int, steps: int, initial: int) -> None: ...

    def update(self, step: TrainStepResult) -> None: ...

    def begin_validation(self, epoch: int, epochs: int, total_records: int) -> None: ...

    def update_validation(self, records: int, source: str) -> None: ...

    def validation(
        self,
        epoch: int,
        epochs: int,
        score: ValidationScore,
        improved: bool,
    ) -> None: ...

    def finish_epoch(self) -> None: ...

    def close(self) -> None: ...


class TqdmTrainProgress:
    """Rank-zero tqdm progress with one bar for train and one for validation."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._bar: Any | None = None
        self._bar_kind: str | None = None

    def start_epoch(self, epoch: int, epochs: int, steps: int, initial: int) -> None:
        self.finish_epoch()
        if not self.enabled:
            return
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        self._bar = tqdm(
            total=steps,
            initial=initial,
            desc=f"Epoch {epoch}/{epochs} train",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        self._bar_kind = "train"

    def update(self, step: TrainStepResult) -> None:
        if self._bar is None:
            return
        postfix: dict[str, str] = {
            "loss": f"{step.total_loss:.4f}",
            "lr": f"{step.learning_rate:.2e}",
            "stage": step.stage,
        }
        if torch.cuda.is_available():
            postfix["vram"] = f"{torch.cuda.max_memory_reserved() / (1024**3):.1f}G"
        self._bar.set_postfix(postfix, refresh=False)
        self._bar.update(1)

    def begin_validation(self, epoch: int, epochs: int, total_records: int) -> None:
        self.finish_epoch()
        if not self.enabled:
            return
        from tqdm.auto import tqdm

        self._bar = tqdm(
            total=total_records,
            desc=f"Epoch {epoch}/{epochs} validation",
            unit="video",
            dynamic_ncols=True,
            leave=True,
        )
        self._bar_kind = "validation"

    def update_validation(self, records: int, source: str) -> None:
        if self._bar is None or self._bar_kind != "validation":
            return
        self._bar.set_postfix_str(f"source={_source_display_name(source)}", refresh=False)
        self._bar.update(records)

    def validation(
        self,
        epoch: int,
        epochs: int,
        score: ValidationScore,
        improved: bool,
    ) -> None:
        if not self.enabled:
            return
        from tqdm.auto import tqdm

        self.finish_epoch()
        tqdm.write(
            _format_validation_summary(
                epoch=epoch,
                epochs=epochs,
                score=score,
                improved=improved,
            )
        )

    def finish_epoch(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
            self._bar_kind = None

    def close(self) -> None:
        self.finish_epoch()


def _format_validation_summary(
    *,
    epoch: int,
    epochs: int,
    score: ValidationScore,
    improved: bool,
) -> str:
    labels = (
        ("tc", "TC"),
        ("lpips", "LPIPS"),
        ("fvd", "FVD"),
        ("fid", "FID"),
        ("clipscore", "CLIPScore"),
        ("ssim", "SSIM"),
    )
    if len(score.source_metrics) != 1:
        raise ValueError("validation summary requires exactly one selected dataset")
    source_key, metrics = next(iter(score.source_metrics.items()))
    source = source_key.removesuffix("_validation")
    checkpoint_status = "updated" if improved else "unchanged"
    lines = [
        (
            f"Epoch {epoch}/{epochs} {_source_display_name(source)} validation: "
            f"weighted_score={score.composite:.6f} "
            f"target_pass={'yes' if score.target_pass else 'no'} "
            f"best.pt={checkpoint_status}"
        ),
        f"metric     {_source_display_name(source):>14}",
    ]
    for name, label in labels:
        lines.append(f"{label:<10} {metrics[name]:>14.6f}")
    return "\n".join(lines)


def _source_display_name(source: str) -> str:
    return {
        "dataverse": "DataVerse",
        "openvid": "OpenVid",
        "seedance": "Seedance",
    }.get(source, source)


class CooperativeStop:
    """Thread-safe first-reason-wins stop token for SIGINT/SIGTERM handling."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str | None = None
        self._lock = threading.Lock()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def request(self, reason: str) -> None:
        normalized = reason.strip() or "requested"
        with self._lock:
            if self._reason is None:
                self._reason = normalized
            self._event.set()


@contextmanager
def cooperative_signal_handlers(stop: CooperativeStop) -> Iterator[None]:
    """Install cooperative handlers and restore the process's prior handlers."""

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def handle(signum: int, _frame: FrameType | None) -> None:
        stop.request(signal.Signals(signum).name)

    try:
        for signum in previous:
            signal.signal(signum, handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class TrainRunRecorder:
    """Atomic manifest plus append-only JSONL events for one rank-zero run."""

    def __init__(self, paths: RunPaths, initial: Mapping[str, object]) -> None:
        self.paths = paths
        self.manifest_path = paths.output_dir / "manifest.json"
        self.resources_path = paths.output_dir / "resources.json"
        self.events_path = paths.output_dir / "events.jsonl"
        self._closed = False
        existing = self._load_existing_manifest()
        for stale_field in ("finished_at", "resource_summary", "stop_reason", "error"):
            existing.pop(stale_field, None)
        self._manifest: dict[str, object] = existing
        self._manifest.update(dict(initial))
        self._manifest["run_id"] = paths.run_id
        self._events: TextIO = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._write_manifest()

    def record(self, event: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed train recorder")
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.paths.run_id,
            **dict(event),
        }
        self._events.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        self._events.flush()

    def update(self, patch: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("cannot update a closed train recorder")
        self._manifest.update(dict(patch))
        self._write_manifest()

    def finalize(
        self,
        *,
        status: RunStatus,
        resources: ResourceSummary,
        error: BaseException | None = None,
        stop_reason: str | None = None,
    ) -> None:
        resource_payload = cast(dict[str, object], _jsonable(resources))
        write_json_manifest(self.resources_path, resource_payload)
        patch: dict[str, object] = {
            "status": status,
            "finished_at": datetime.now(UTC),
            "resource_summary": resources,
        }
        if stop_reason is not None:
            patch["stop_reason"] = stop_reason
        if error is not None:
            patch["error"] = {"type": type(error).__name__, "message": str(error)}
        self.update(patch)
        self.record(
            {
                "type": "run_finished",
                "status": status,
                "stop_reason": stop_reason,
                "error_type": None if error is None else type(error).__name__,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._events.flush()
        self._events.close()
        self._closed = True

    def _load_existing_manifest(self) -> dict[str, object]:
        if not self.manifest_path.is_file():
            return {}
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("existing train manifest must be a JSON object")
        existing_run_id = raw.get("run_id")
        if existing_run_id not in (None, self.paths.run_id):
            raise ValueError("existing train manifest belongs to a different run")
        return {str(key): value for key, value in raw.items()}

    def _write_manifest(self) -> None:
        write_json_manifest(self.manifest_path, self._manifest)


def coordinate_run_paths(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    allocator: Callable[..., RunPaths] = create_run_paths,
    broadcaster: Callable[[RunPaths | None, DistributedContext], RunPaths] | None = None,
) -> RunPaths:
    """Allocate or recover run paths on rank zero and broadcast them to all ranks."""

    paths: RunPaths | None = None
    if context.is_main:
        resume = getattr(args, "resume", None)
        dataset = str(args.dataset)
        if resume is None:
            paths = allocator(
                args.output_root,
                Path(args.checkpoint_root) / dataset,
                f"train/{dataset}",
            )
        else:
            checkpoint = validate_dataset_checkpoint_path(
                resume,
                checkpoint_root=args.checkpoint_root,
                dataset=dataset,
                purpose="resume",
            )
            run_id = checkpoint.parent.name
            if not run_id:
                raise ValueError("resume checkpoint parent must define a run ID")
            output_dir = Path(args.output_root) / "train" / dataset / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = RunPaths(run_id, output_dir, checkpoint.parent)
    selected_broadcaster = _broadcast_run_paths if broadcaster is None else broadcaster
    return selected_broadcaster(paths, context)


def validate_dataset_checkpoint_path(
    checkpoint: Path | str,
    *,
    checkpoint_root: Path | str,
    dataset: str,
    purpose: str,
    allow_cross_dataset: bool = False,
) -> Path:
    """Validate a dataset-scoped checkpoint, with explicit cross-dataset opt-in."""

    resolved = Path(checkpoint).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{purpose} checkpoint does not exist: {resolved}")
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    expected_root = checkpoint_root / dataset
    try:
        relative = resolved.relative_to(expected_root)
    except ValueError as error:
        if not allow_cross_dataset:
            raise ValueError(
                f"{purpose} checkpoint must belong to dataset {dataset!r}: {resolved}"
            ) from error
        try:
            cross_relative = resolved.relative_to(checkpoint_root)
        except ValueError as root_error:
            raise ValueError(
                f"cross-dataset {purpose} checkpoint must be under checkpoint root: {resolved}"
            ) from root_error
        if len(cross_relative.parts) != 3 or cross_relative.parts[-1] != "best.pt":
            raise ValueError(
                f"cross-dataset {purpose} checkpoint must be <dataset>/<run>/best.pt: {resolved}"
            )
        source_dataset = cross_relative.parts[0]
        if source_dataset not in _VALIDATION_SOURCES:
            raise ValueError(
                f"cross-dataset {purpose} checkpoint has unknown source dataset "
                f"{source_dataset!r}: {resolved}"
            )
        return resolved
    if len(relative.parts) != 2:
        raise ValueError(
            f"{purpose} checkpoint must be stored in one timestamped run directory"
        )
    if not resolved.parent.name:
        raise ValueError(f"{purpose} checkpoint parent must define a run ID")
    return resolved


def wrap_distributed_model(
    model: nn.Module,
    context: DistributedContext,
    *,
    ddp_factory: DDPFactory = DistributedDataParallel,
) -> nn.Module:
    """Wrap the execution model exactly once for staged-loss distributed training."""

    if context.world_size <= 1:
        return model
    options: dict[str, object] = {"find_unused_parameters": True}
    if context.device.type == "cuda":
        options.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return ddp_factory(model, **options)


def prepare_amp_master_weights(model: nn.Module) -> None:
    """Keep trainable master parameters in FP32 while frozen priors stay compact."""

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point():
            parameter.data = parameter.data.float()


def run_train_epoch_loop(
    args: argparse.Namespace,
    *,
    context: DistributedContext,
    model: nn.Module,
    engine: TrainEngine,
    loaders: RemoteDataLoaders,
    metric_suite: MetricSuite,
    paths: RunPaths,
    training_signature: Mapping[str, object],
    start_position: tuple[int, int],
    stop: CooperativeStop,
    recorder: TrainLoopRecorder | None = None,
    progress: TrainProgress | None = None,
    validation_runner: ValidationRunner = evaluate_validation,
    checkpoint_writer: CheckpointWriter = atomic_torch_save,
) -> TrainLoopResult:
    """Execute exact complete epochs from a validated resumable batch position."""

    epochs = _positive_int(args.epochs, "epochs")
    steps_per_epoch = _positive_int(args.steps_per_epoch, "steps_per_epoch")
    validation_interval = _positive_int(args.validation_interval, "validation_interval")
    checkpoint_interval_steps = _positive_int(
        args.checkpoint_interval_steps,
        "checkpoint_interval_steps",
    )
    position = _validate_resume_position(
        start_position,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        micro_step=engine.micro_step,
    )
    validation_epochs: list[int] = []
    stop_seen = _distributed_stop_requested(stop, context)
    if stop_seen:
        _save_training_checkpoint(
            engine=engine,
            context=context,
            paths=paths,
            position=position,
            training_signature=training_signature,
            status="interrupted",
            validation_metrics=None,
            improved=False,
            checkpoint_writer=checkpoint_writer,
        )
        _record_interruption(recorder, context, engine, position, stop)
        context.barrier()
        return _loop_result(engine, position, validation_epochs, stop)

    start_epoch, first_batch_index = position
    for epoch_index in range(start_epoch, epochs):
        loaders.set_epoch(epoch_index)
        skip = first_batch_index if epoch_index == start_epoch else 0
        if progress is not None:
            progress.start_epoch(epoch_index + 1, epochs, steps_per_epoch, skip)
        set_start_batch = getattr(loaders, "set_start_batch", None)
        loader_seeked = callable(set_start_batch) and set_start_batch(skip) is not False
        iterator = iter(loaders.train)
        if not loader_seeked:
            for skipped_index in range(skip):
                _next_training_batch(iterator, epoch_index, skipped_index, steps_per_epoch)
        if _distributed_stop_requested(stop, context):
            position = (epoch_index, skip)
            _save_training_checkpoint(
                engine=engine,
                context=context,
                paths=paths,
                position=position,
                training_signature=training_signature,
                status="interrupted",
                validation_metrics=None,
                improved=False,
                checkpoint_writer=checkpoint_writer,
            )
            _record_interruption(recorder, context, engine, position, stop)
            context.barrier()
            return _loop_result(engine, position, validation_epochs, stop)

        for batch_index in range(skip, steps_per_epoch):
            clip_batch = _next_training_batch(
                iterator,
                epoch_index,
                batch_index,
                steps_per_epoch,
            )
            training_batch, record_ids = clip_batch_to_training_batch(clip_batch, context.device)
            step = engine.train_microbatch(training_batch, batch_ids=record_ids)
            position = (
                (epoch_index + 1, 0)
                if batch_index + 1 == steps_per_epoch
                else (epoch_index, batch_index + 1)
            )
            _record_microbatch(recorder, context, epoch_index, batch_index, record_ids, step)
            if progress is not None:
                progress.update(step)
            stop_seen = _distributed_stop_requested(stop, context)
            if stop_seen:
                _save_training_checkpoint(
                    engine=engine,
                    context=context,
                    paths=paths,
                    position=position,
                    training_signature=training_signature,
                    status="interrupted",
                    validation_metrics=None,
                    improved=False,
                    checkpoint_writer=checkpoint_writer,
                )
                _record_interruption(recorder, context, engine, position, stop)
                context.barrier()
                return _loop_result(engine, position, validation_epochs, stop)
            if (
                batch_index + 1 < steps_per_epoch
                and engine.micro_step % checkpoint_interval_steps == 0
                and engine.accumulation_index == 0
            ):
                _save_training_checkpoint(
                    engine=engine,
                    context=context,
                    paths=paths,
                    position=position,
                    training_signature=training_signature,
                    status="running",
                    validation_metrics=None,
                    improved=False,
                    checkpoint_writer=checkpoint_writer,
                )
                context.barrier()

        completed_epoch = epoch_index + 1
        should_validate = completed_epoch % validation_interval == 0 or completed_epoch == epochs
        validation_metrics: dict[str, dict[str, float]] | None = None
        improved = False
        if should_validate:
            if progress is not None:
                progress.begin_validation(
                    completed_epoch,
                    epochs,
                    _validation_record_count(loaders),
                )
            try:
                with ema_temporal_parameters(model, engine.ema):
                    validation_metrics = validation_runner(
                        model,
                        loaders,
                        metric_suite,
                        fps=int(args.fps),
                        device=context.device,
                        seed=int(args.seed),
                        stop=stop,
                        context=context,
                        progress=progress,
                    )
            except _ValidationInterrupted:
                if not stop.requested:
                    stop.request("validation_interrupted")
                position = (completed_epoch, 0)
                _save_training_checkpoint(
                    engine=engine,
                    context=context,
                    paths=paths,
                    position=position,
                    training_signature=training_signature,
                    status="interrupted",
                    validation_metrics=None,
                    improved=False,
                    checkpoint_writer=checkpoint_writer,
                )
                _record_interruption(recorder, context, engine, position, stop)
                context.barrier()
                return _loop_result(engine, position, validation_epochs, stop)
            if _distributed_stop_requested(stop, context):
                position = (completed_epoch, 0)
                _save_training_checkpoint(
                    engine=engine,
                    context=context,
                    paths=paths,
                    position=position,
                    training_signature=training_signature,
                    status="interrupted",
                    validation_metrics=None,
                    improved=False,
                    checkpoint_writer=checkpoint_writer,
                )
                _record_interruption(recorder, context, engine, position, stop)
                context.barrier()
                return _loop_result(engine, position, validation_epochs, stop)
            validation_score = score_validation_event(
                validation_metrics,
                baselines=engine.selector.baselines,
            )
            improved = engine.selector.update(validation_metrics, epoch=completed_epoch)
            validation_epochs.append(completed_epoch)
            _record_validation(
                recorder,
                context,
                completed_epoch,
                validation_metrics,
                validation_score,
                improved,
            )
            if progress is not None:
                progress.validation(
                    completed_epoch,
                    epochs,
                    validation_score,
                    improved,
                )

        stop_after_epoch = _distributed_stop_requested(stop, context)
        final_status: RunStatus
        if stop_after_epoch:
            final_status = "interrupted"
        elif completed_epoch == epochs:
            final_status = "completed"
        else:
            final_status = "running"
        position = (completed_epoch, 0)
        if final_status == "completed":
            completion_write_error: BaseException | None = None
            try:
                _save_training_checkpoint(
                    engine=engine,
                    context=context,
                    paths=paths,
                    position=position,
                    training_signature=training_signature,
                    status=final_status,
                    validation_metrics=validation_metrics,
                    improved=improved,
                    checkpoint_writer=checkpoint_writer,
                    candidate=True,
                )
            except BaseException as error:
                completion_write_error = error
            completion_write_failed = _distributed_completion_operation_failed(
                completion_write_error,
                context,
            )
            if completion_write_failed:
                _discard_completion_candidates(paths, context, improved=improved)
                if completion_write_error is not None:
                    raise completion_write_error
                raise RuntimeError("completion checkpoint writer failed on rank zero")
            stop_after_epoch = _distributed_stop_requested(stop, context)
            if stop_after_epoch:
                _discard_completion_candidates(paths, context, improved=improved)
                final_status = "interrupted"
                _save_training_checkpoint(
                    engine=engine,
                    context=context,
                    paths=paths,
                    position=position,
                    training_signature=training_signature,
                    status=final_status,
                    validation_metrics=validation_metrics,
                    improved=improved,
                    checkpoint_writer=checkpoint_writer,
                )
            else:
                completion_promotion_error: BaseException | None = None
                try:
                    _promote_completion_candidates(paths, context, improved=improved)
                except BaseException as error:
                    completion_promotion_error = error
                completion_promotion_failed = _distributed_completion_operation_failed(
                    completion_promotion_error,
                    context,
                )
                if completion_promotion_failed:
                    if completion_promotion_error is not None:
                        raise completion_promotion_error
                    raise RuntimeError("completion checkpoint promotion failed on rank zero")
        else:
            _save_training_checkpoint(
                engine=engine,
                context=context,
                paths=paths,
                position=position,
                training_signature=training_signature,
                status=final_status,
                validation_metrics=validation_metrics,
                improved=improved,
                checkpoint_writer=checkpoint_writer,
            )
        _record_epoch(
            recorder,
            context,
            engine,
            position,
            validation_metrics,
            final_status,
        )
        if progress is not None:
            progress.finish_epoch()
        context.barrier()
        if stop_after_epoch:
            _record_interruption(recorder, context, engine, position, stop)
            return _loop_result(engine, position, validation_epochs, stop)
        first_batch_index = 0

    return TrainLoopResult(
        status="completed",
        position=(epochs, 0),
        microbatches_completed=engine.micro_step,
        optimizer_steps_completed=engine.optimizer_step,
        validation_epochs=tuple(validation_epochs),
        stop_reason=None,
    )


def _validation_record_count(loaders: RemoteDataLoaders) -> int:
    """Return the selected dataset's global validation size."""

    validation = loaders.splits["validation"]
    if len(validation) != 1:
        raise ValueError("validation split must contain exactly one selected dataset")
    return len(next(iter(validation.values())))


def _validate_resume_position(
    position: tuple[int, int],
    *,
    epochs: int,
    steps_per_epoch: int,
    micro_step: int,
) -> tuple[int, int]:
    epoch, next_batch_index = position
    valid = (
        0 <= epoch <= epochs
        and 0 <= next_batch_index < steps_per_epoch
        and not (epoch == epochs and next_batch_index != 0)
    )
    if not valid:
        raise ValueError(
            f"resume position {(epoch, next_batch_index)} is outside "
            f"{epochs} epochs x {steps_per_epoch} batches"
        )
    expected_micro_step = epoch * steps_per_epoch + next_batch_index
    if micro_step != expected_micro_step:
        raise ValueError(
            f"resume position expects micro_step={expected_micro_step}, got {micro_step}"
        )
    return epoch, next_batch_index


def _next_training_batch(
    iterator: Iterator[ClipBatch],
    epoch: int,
    batch_index: int,
    steps_per_epoch: int,
) -> ClipBatch:
    try:
        return next(iterator)
    except StopIteration as error:
        raise RuntimeError(
            f"train loader ended at epoch={epoch}, batch={batch_index}; "
            f"expected {steps_per_epoch} microbatches"
        ) from error


def _loop_result(
    engine: TrainEngine,
    position: tuple[int, int],
    validation_epochs: Sequence[int],
    stop: CooperativeStop,
) -> TrainLoopResult:
    return TrainLoopResult(
        status="interrupted",
        position=position,
        microbatches_completed=engine.micro_step,
        optimizer_steps_completed=engine.optimizer_step,
        validation_epochs=tuple(validation_epochs),
        stop_reason=stop.reason or "distributed_stop",
    )


def _save_training_checkpoint(
    *,
    engine: TrainEngine,
    context: DistributedContext,
    paths: RunPaths,
    position: tuple[int, int],
    training_signature: Mapping[str, object],
    status: RunStatus,
    validation_metrics: Mapping[str, Mapping[str, float]] | None,
    improved: bool,
    checkpoint_writer: CheckpointWriter,
    candidate: bool = False,
) -> None:
    rank_random_states = _collect_rank_random_states(engine, context)
    if not context.is_main:
        return
    epoch, next_batch_index = position
    payload = engine.state_dict(epoch=epoch, next_batch_index=next_batch_index)
    payload.update(
        {
            "run_id": paths.run_id,
            "run_status": status,
            "world_size": context.world_size,
            "training_signature": _jsonable(training_signature),
            "rank_random_states": rank_random_states,
            "validation_metrics": (
                None if validation_metrics is None else _jsonable(validation_metrics)
            ),
            "validation_score": _validation_score_payload(engine.selector.best_score),
        }
    )
    suffix = ".candidate" if candidate else ""
    checkpoint_writer(payload, paths.checkpoint_dir / f"latest.pt{suffix}")
    if improved:
        checkpoint_writer(payload, paths.checkpoint_dir / f"best.pt{suffix}")


def _discard_completion_candidates(
    paths: RunPaths,
    context: DistributedContext,
    *,
    improved: bool,
) -> None:
    if not context.is_main:
        return
    (paths.checkpoint_dir / "latest.pt.candidate").unlink(missing_ok=True)
    if improved:
        (paths.checkpoint_dir / "best.pt.candidate").unlink(missing_ok=True)


def _promote_completion_candidates(
    paths: RunPaths,
    context: DistributedContext,
    *,
    improved: bool,
) -> None:
    if not context.is_main:
        return
    candidates = [
        (
            paths.checkpoint_dir / "latest.pt.candidate",
            paths.checkpoint_dir / "latest.pt",
        )
    ]
    if improved:
        candidates.insert(
            0,
            (paths.checkpoint_dir / "best.pt.candidate", paths.checkpoint_dir / "best.pt"),
        )
    missing = [str(candidate) for candidate, _ in candidates if not candidate.is_file()]
    if missing:
        raise RuntimeError(f"completion checkpoint writer did not create candidates: {missing}")
    rollback_entries: list[tuple[Path, Path, bool]] = []
    try:
        for _, destination in candidates:
            rollback = destination.with_name(f".{destination.name}.promotion-rollback")
            rollback.unlink(missing_ok=True)
            existed = destination.is_file()
            rollback_entries.append((destination, rollback, existed))
            if existed:
                os.link(destination, rollback)
        for candidate, destination in candidates:
            candidate.replace(destination)
    except BaseException as promotion_error:
        rollback_error: BaseException | None = None
        for destination, rollback, existed in reversed(rollback_entries):
            try:
                if existed:
                    rollback.replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            except BaseException as error:
                if rollback_error is None:
                    rollback_error = error
        for candidate, _ in candidates:
            candidate.unlink(missing_ok=True)
        for _, rollback, _ in rollback_entries:
            rollback.unlink(missing_ok=True)
        if rollback_error is not None:
            raise RuntimeError(
                "completion checkpoint promotion rollback failed"
            ) from promotion_error
        raise
    for _, rollback, _ in rollback_entries:
        rollback.unlink(missing_ok=True)


def _collect_rank_random_states(
    engine: TrainEngine,
    context: DistributedContext,
) -> dict[str, object] | None:
    local_state = engine.stochastic_state_dict()
    if context.world_size <= 1:
        return {str(context.rank): local_state} if context.is_main else None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "distributed checkpoint collection requires an initialized process group"
        )
    gathered: list[object] | None = [None] * context.world_size if context.is_main else None
    dist.gather_object(local_state, gathered, dst=0)
    if not context.is_main:
        return None
    if gathered is None or any(state is None for state in gathered):
        raise RuntimeError("distributed checkpoint did not collect every rank random state")
    return {str(rank): state for rank, state in enumerate(gathered)}


def _distributed_stop_requested(
    stop: CooperativeStop,
    context: DistributedContext,
) -> bool:
    if context.world_size <= 1:
        return stop.requested
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed stop synchronization requires an initialized process group")
    flag = torch.tensor(int(stop.requested), device=context.device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    requested = bool(flag.item())
    if requested and not stop.requested:
        stop.request("distributed_stop")
    return requested


def _distributed_completion_operation_failed(
    error: BaseException | None,
    context: DistributedContext,
) -> bool:
    if context.world_size <= 1:
        return error is not None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "distributed completion synchronization requires an initialized process group"
        )
    failed = torch.tensor(int(error is not None), device=context.device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    return bool(failed.item())


def _synchronize_validation_round(
    stop: CooperativeStop | None,
    context: DistributedContext | None,
    *,
    local_has_batch: bool,
    local_records: int,
) -> tuple[bool, bool, int]:
    if local_records < 0:
        raise ValueError("local validation records cannot be negative")
    if stop is None and context is None:
        return False, local_has_batch, local_records
    if stop is None or context is None:
        raise ValueError("validation stop and distributed context must be provided together")
    if context.world_size <= 1:
        return stop.requested, local_has_batch, local_records
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "distributed validation cancellation requires an initialized process group"
        )
    flags = torch.tensor(
        [int(stop.requested), int(local_has_batch), local_records],
        device=context.device,
        dtype=torch.int32,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    cancelled = bool(flags[0].item())
    if cancelled and not stop.requested:
        stop.request("distributed_stop")
    return cancelled, bool(flags[1].item()), int(flags[2].item())


def _record_microbatch(
    recorder: TrainLoopRecorder | None,
    context: DistributedContext,
    epoch_index: int,
    batch_index: int,
    record_ids: tuple[str, ...],
    step: TrainStepResult,
) -> None:
    _record_rank_zero(
        recorder,
        context,
        {
            "type": "microbatch",
            "epoch": epoch_index + 1,
            "batch_index": batch_index,
            "record_ids": record_ids,
            "micro_step": step.micro_step,
            "optimizer_step": step.optimizer_step,
            "optimizer_updated": step.optimizer_updated,
            "skipped_nonfinite": step.skipped_nonfinite,
            "total_loss": step.total_loss,
            "losses": step.losses,
            "method_metrics": step.metrics,
            "gradient_norm": step.gradient_norm,
            "learning_rate": step.learning_rate,
            "stage": step.stage,
        },
    )


def _record_validation(
    recorder: TrainLoopRecorder | None,
    context: DistributedContext,
    epoch: int,
    metrics: Mapping[str, Mapping[str, float]],
    score: ValidationScore | None,
    improved: bool,
) -> None:
    _record_rank_zero(
        recorder,
        context,
        {
            "type": "validation",
            "epoch": epoch,
            "metrics": metrics,
            "score": _validation_score_payload(score),
            "improved": improved,
        },
    )


def _record_epoch(
    recorder: TrainLoopRecorder | None,
    context: DistributedContext,
    engine: TrainEngine,
    position: tuple[int, int],
    validation_metrics: Mapping[str, Mapping[str, float]] | None,
    status: RunStatus,
) -> None:
    if recorder is None or not context.is_main:
        return
    progress = {
        "epoch": position[0],
        "next_batch_index": position[1],
        "micro_step": engine.micro_step,
        "optimizer_step": engine.optimizer_step,
    }
    recorder.record(
        {
            "type": "epoch_completed",
            "status": status,
            "progress": progress,
            "validation_metrics": validation_metrics,
        }
    )
    recorder.update(
        {
            "status": status,
            "progress": progress,
            "last_validation_metrics": validation_metrics,
            "best_validation_score": _validation_score_payload(engine.selector.best_score),
            "best_epoch": engine.selector.best_epoch,
            "nonfinite_ledger": engine.nonfinite_ledger,
        }
    )


def _record_interruption(
    recorder: TrainLoopRecorder | None,
    context: DistributedContext,
    engine: TrainEngine,
    position: tuple[int, int],
    stop: CooperativeStop,
) -> None:
    if recorder is None or not context.is_main:
        return
    progress = {
        "epoch": position[0],
        "next_batch_index": position[1],
        "micro_step": engine.micro_step,
        "optimizer_step": engine.optimizer_step,
    }
    recorder.record(
        {
            "type": "interrupted",
            "reason": stop.reason or "distributed_stop",
            "progress": progress,
        }
    )
    recorder.update(
        {
            "status": "interrupted",
            "stop_reason": stop.reason or "distributed_stop",
            "progress": progress,
            "nonfinite_ledger": engine.nonfinite_ledger,
        }
    )


def _record_rank_zero(
    recorder: TrainLoopRecorder | None,
    context: DistributedContext,
    event: Mapping[str, object],
) -> None:
    if recorder is not None and context.is_main:
        recorder.record(event)


def _validation_score_payload(score: ValidationScore | None) -> dict[str, object] | None:
    if score is None:
        return None
    return {
        "source_metrics": score.source_metrics,
        "average_metrics": score.average_metrics,
        "normalized_metrics": score.normalized_metrics,
        "composite": score.composite,
        "target_pass": score.target_pass,
    }


class _RuntimeLike(Protocol):
    @property
    def model(self) -> TARDISModel: ...

    @property
    def metric_suite(self) -> MetricSuite: ...

    @property
    def dataset_sources(self) -> tuple[str, ...]: ...

    @property
    def factory_options(self) -> object: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def torch_dtype(self) -> torch.dtype: ...


def _default_context_factory(device_type: str | None) -> DistributedContext:
    return DistributedContext.from_environment(device_type=device_type)


def _default_runtime_builder(
    args: argparse.Namespace,
    *,
    restore_checkpoint: bool,
) -> _RuntimeLike:
    from tardis.cli.runtime import build_production_runtime

    return build_production_runtime(args, restore_checkpoint=restore_checkpoint)


def _default_warm_start_loader(
    model: TARDISModel,
    checkpoint: Path | str,
    *,
    use_ema: bool,
) -> _WarmStartIdentity:
    from tardis.cli.runtime import load_temporal_checkpoint

    return load_temporal_checkpoint(model, checkpoint, use_ema=use_ema)


def _default_monitor_factory(device: torch.device) -> ResourceMonitor:
    return ResourceMonitor(sample_fn=lambda: sample_resources(device))


def _default_engine_builder(
    args: argparse.Namespace,
    runtime: _RuntimeLike,
    execution_model: nn.Module,
    rank_seed: int,
) -> TrainEngine:
    from tardis.training.losses import LossWeights
    from tardis.training.objective import TARDISKeyframeObjective, TARDISObjective

    options = train_engine_options_from_args(args)
    objective: TARDISKeyframeObjective | TARDISObjective
    if str(args.train_mode) == "keyframe_only":
        objective = TARDISKeyframeObjective(
            perceptual_metric=runtime.metric_suite.lpips.feature,
            keyframe_weight=float(args.keyframe_loss_weight),
            lpips_weight=float(args.lpips_loss_weight),
            lpips_frame_chunk_size=int(args.lpips_frame_chunk_size),
        )
    else:
        objective = TARDISObjective(
            perceptual_metric=runtime.metric_suite.lpips.feature,
            weights=LossWeights(
                diffusion=float(args.diffusion_loss_weight),
                keyframe=float(args.keyframe_loss_weight),
                residual=float(args.residual_loss_weight),
                tc=float(args.tc_loss_weight),
                lpips=float(args.lpips_loss_weight),
                transport=float(args.transport_loss_weight),
                flow=float(args.flow_loss_weight),
                visibility=float(args.visibility_loss_weight),
                router=float(args.router_loss_weight),
                survival=float(args.survival_loss_weight),
                lite=float(args.lite_loss_weight),
                budget=float(args.budget_loss_weight),
                warp=float(args.warp_loss_weight),
                drift=float(args.drift_loss_weight),
                crcd=float(args.crcd_loss_weight),
                text=float(args.text_loss_weight),
            ),
            lpips_frame_chunk_size=int(args.lpips_frame_chunk_size),
        )
    return TrainEngine(
        execution_model,
        objective=objective,
        options=options,
        curriculum=CurriculumSchedule(
            durations=curriculum_durations_for_profile(
                options.total_optimizer_steps,
                str(args.curriculum_profile),
            )
        ),
        generator=make_generator(rank_seed, runtime.device),
        selector=ValidationCheckpointSelector(
            baselines=cast(
                Mapping[ValidationMetric | str, float],
                selection_scales_for_source(f"{args.dataset}_validation"),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class TrainServices:
    """Injection boundaries for production assembly and orchestration tests."""

    context_factory: Callable[[str | None], DistributedContext] = _default_context_factory
    runtime_builder: Callable[..., _RuntimeLike] = _default_runtime_builder
    loader_builder: Callable[[argparse.Namespace, DistributedContext], RemoteDataLoaders] = (
        build_train_dataloaders
    )
    model_wrapper: Callable[[nn.Module, DistributedContext], nn.Module] = wrap_distributed_model
    engine_builder: Callable[[argparse.Namespace, _RuntimeLike, nn.Module, int], TrainEngine] = (
        _default_engine_builder
    )
    monitor_factory: Callable[[torch.device], ResourceMonitor] = _default_monitor_factory
    path_allocator: Callable[..., RunPaths] = create_run_paths
    path_broadcaster: Callable[[RunPaths | None, DistributedContext], RunPaths] | None = None
    recorder_factory: Callable[[RunPaths, Mapping[str, object]], TrainRunRecorder] = (
        TrainRunRecorder
    )
    checkpoint_loader: Callable[..., dict[str, Any]] = load_checkpoint
    warm_start_loader: WarmStartLoader = _default_warm_start_loader
    checkpoint_writer: CheckpointWriter = atomic_torch_save
    validation_runner: ValidationRunner = evaluate_validation


def run_training(
    args: argparse.Namespace,
    *,
    services: TrainServices | None = None,
    stop: CooperativeStop | None = None,
) -> TrainLoopResult:
    """Assemble and execute one production train run with unconditional cleanup."""

    selected = TrainServices() if services is None else services
    requested_device = torch.device(str(args.device))
    context = selected.context_factory(requested_device.type)
    stop_token = CooperativeStop() if stop is None else stop
    recorder: TrainRunRecorder | None = None
    monitor: ResourceMonitor | None = None
    progress: TqdmTrainProgress | None = None
    result: TrainLoopResult | None = None
    status: RunStatus = "failed"
    run_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        context.initialize()
        progress = TqdmTrainProgress(enabled=context.is_main)
        local_args = argparse.Namespace(**vars(args))
        local_args.device = str(context.device)
        rank_seed = seed_everything(
            int(local_args.seed),
            rank=int(context.rank),
            deterministic=bool(local_args.deterministic),
        )
        _enable_cuda_training_defaults(context.device)
        paths = coordinate_run_paths(
            local_args,
            context,
            allocator=selected.path_allocator,
            broadcaster=selected.path_broadcaster,
        )
        if context.is_main:
            recorder = selected.recorder_factory(
                paths,
                _initial_manifest(local_args, context, paths, rank_seed),
            )
            monitor = selected.monitor_factory(context.device)
            monitor.sample_once()
            monitor.start()
            recorder.record(
                {
                    "type": "run_started",
                    "resume": None if local_args.resume is None else str(local_args.resume),
                }
            )

        with cooperative_signal_handlers(stop_token):
            runtime = selected.runtime_builder(local_args, restore_checkpoint=False)
            warm_start_identity: _WarmStartIdentity | None = None
            if local_args.warm_start is not None:
                warm_start_path = validate_dataset_checkpoint_path(
                    local_args.warm_start,
                    checkpoint_root=local_args.checkpoint_root,
                    dataset=str(local_args.dataset),
                    purpose="warm-start",
                    allow_cross_dataset=bool(local_args.allow_cross_dataset_warm_start),
                )
                warm_start_identity = selected.warm_start_loader(
                    runtime.model,
                    warm_start_path,
                    use_ema=bool(local_args.warm_start_use_ema),
                )
                if recorder is not None:
                    recorder.record(
                        {
                            "type": "weights_only_warm_start",
                            "checkpoint": str(warm_start_identity.path),
                            "checkpoint_sha256": warm_start_identity.sha256,
                            "used_ema": warm_start_identity.used_ema,
                            "cross_dataset": warm_start_path.parent.parent.name
                            != str(local_args.dataset),
                        }
                    )
            from tardis.training.modes import configure_train_mode

            train_mode_summary = configure_train_mode(runtime.model, str(local_args.train_mode))
            loaders = selected.loader_builder(local_args, context)
            resolved_steps = getattr(loaders, "steps_per_epoch", None)
            if resolved_steps is not None:
                resolved_steps = _positive_int(resolved_steps, "resolved steps_per_epoch")
                if (
                    local_args.steps_per_epoch is not None
                    and int(local_args.steps_per_epoch) != resolved_steps
                ):
                    raise ValueError(
                        "steps_per_epoch does not match the assembled training loader: "
                        f"requested={local_args.steps_per_epoch}, resolved={resolved_steps}"
                    )
                local_args.steps_per_epoch = resolved_steps
            if local_args.steps_per_epoch is None:
                raise ValueError("training loader did not expose resolved steps_per_epoch")
            prepare_amp_master_weights(runtime.model)
            execution_model: nn.Module = runtime.model
            if bool(local_args.compile_model):
                execution_model = cast(nn.Module, torch.compile(execution_model))
            execution_model = selected.model_wrapper(execution_model, context)
            engine = selected.engine_builder(
                local_args,
                runtime,
                execution_model,
                rank_seed,
            )
            source_identity = _source_identity(
                loaders,
                runtime.dataset_sources,
                dataset=str(local_args.dataset),
            )
            training_signature = _training_signature(
                local_args,
                runtime,
                source_identity,
                context.world_size,
            )
            if recorder is not None:
                recorder.update(
                    {
                        "status": "running",
                        "sources": source_identity,
                        "model_provenance": _model_provenance(local_args, runtime.model),
                        "train_mode": dataclasses.asdict(train_mode_summary),
                        "metric_provenance": runtime.metric_suite.provenance_ids,
                        "training_signature": training_signature,
                        "weights_only_warm_start": (
                            None
                            if warm_start_identity is None
                            else {
                                "enabled": True,
                                "checkpoint": str(warm_start_identity.path),
                                "checkpoint_sha256": warm_start_identity.sha256,
                                "used_ema": warm_start_identity.used_ema,
                                "cross_dataset": warm_start_path.parent.parent.name
                                != str(local_args.dataset),
                                "optimizer_restored": False,
                                "scheduler_restored": False,
                                "curriculum_restored": False,
                            }
                        ),
                    }
                )

            start_position = (0, 0)
            if local_args.resume is not None:
                checkpoint = selected.checkpoint_loader(Path(local_args.resume), map_location="cpu")
                _validate_resume_checkpoint(
                    checkpoint,
                    paths=paths,
                    signature=training_signature,
                    world_size=context.world_size,
                )
                start_position = engine.load_state_dict(checkpoint)
                _restore_rank_random_state(engine, checkpoint, context)
                if recorder is not None:
                    recorder.record(
                        {
                            "type": "resumed",
                            "checkpoint": str(local_args.resume),
                            "position": start_position,
                            "micro_step": engine.micro_step,
                            "optimizer_step": engine.optimizer_step,
                        }
                    )

            result = run_train_epoch_loop(
                local_args,
                context=context,
                model=runtime.model,
                engine=engine,
                loaders=loaders,
                metric_suite=runtime.metric_suite,
                paths=paths,
                training_signature=training_signature,
                start_position=start_position,
                stop=stop_token,
                recorder=recorder,
                progress=progress,
                validation_runner=selected.validation_runner,
                checkpoint_writer=selected.checkpoint_writer,
            )
            status = result.status
            return result
    except BaseException as error:
        run_error = error
        status = "failed"
        raise
    finally:
        if progress is not None:
            progress.close()
        if monitor is not None:
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_errors.append(error)
        summary = ResourceSummary(0, 0.0, 0.0, 0.0, 0.0) if monitor is None else monitor.summary()
        if recorder is not None:
            try:
                recorder.finalize(
                    status=status,
                    resources=summary,
                    error=run_error,
                    stop_reason=None if result is None else result.stop_reason,
                )
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                try:
                    recorder.close()
                except BaseException as error:
                    cleanup_errors.append(error)
        try:
            context.close()
        except BaseException as error:
            cleanup_errors.append(error)
        if run_error is None and cleanup_errors:
            raise cleanup_errors[0]


def _initial_manifest(
    args: argparse.Namespace,
    context: DistributedContext,
    paths: RunPaths,
    rank_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "tardis_train_run",
        "run_id": paths.run_id,
        "status": "initializing",
        "started_at": datetime.now(UTC),
        "args": vars(args),
        "distributed": {
            "world_size": context.world_size,
            "rank_zero_seed": effective_seed(int(args.seed), rank=0),
            "rank_seed": rank_seed,
            "device": str(context.device),
        },
        "progress": {
            "epoch": 0,
            "next_batch_index": 0,
            "micro_step": 0,
            "optimizer_step": 0,
        },
    }


def _source_identity(
    loaders: RemoteDataLoaders,
    dataset_sources: Sequence[str],
    *,
    dataset: str = "dataverse",
) -> dict[str, object]:
    records_by_source = getattr(loaders.catalog, "records_by_source", None)
    if not isinstance(records_by_source, Mapping):
        raise ValueError("train loader catalog must expose records_by_source")
    if dataset not in _VALIDATION_SOURCES:
        raise ValueError(f"unknown training dataset: {dataset!r}")
    if dataset not in records_by_source:
        raise ValueError("train loader catalog does not contain the selected dataset")
    if len(dataset_sources) not in (1, len(_VALIDATION_SOURCES)):
        raise ValueError("runtime dataset source paths must contain one or three entries")
    paths_by_source: dict[str, str] = {}
    for source_path in dataset_sources:
        matches = [
            source
            for source, directory in _SOURCE_DIRECTORIES.items()
            if Path(source_path).name == directory
        ]
        if len(matches) != 1 or matches[0] in paths_by_source:
            raise ValueError(
                f"runtime dataset source path is not uniquely canonical: {source_path}"
            )
        paths_by_source[matches[0]] = str(Path(source_path).resolve())
    if dataset not in paths_by_source:
        raise ValueError("runtime dataset source paths do not identify the selected dataset")
    result: dict[str, object] = {}
    for source in (dataset,):
        records = tuple(records_by_source[source])
        if not records:
            raise ValueError(f"train loader source {source!r} is empty")
        metadata = getattr(records[0], "metadata", None)
        revision = metadata.get("revision") if isinstance(metadata, Mapping) else None
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"train loader source {source!r} has no revision identity")
        split_counts = {
            split: len(loaders.splits[cast(Any, split)][source])
            for split in ("train", "validation", "test")
        }
        result[source] = {
            "path": paths_by_source[source],
            "revision": revision,
            "catalog_records": len(records),
            "split_records": split_counts,
        }
    return result


def _training_signature(
    args: argparse.Namespace,
    runtime: _RuntimeLike,
    source_identity: Mapping[str, object],
    world_size: int,
) -> dict[str, object]:
    argument_names = (
        "pretrained_model",
        "dataset",
        "train_mode",
        "height",
        "width",
        "num_frames",
        "fps",
        "latent_channels",
        "patch_size",
        "hidden_size",
        "num_layers",
        "num_heads",
        "active_ratio",
        "transport_quotient",
        "quotient_regularization",
        "quotient_rank_threshold",
        "innovation_proper_time",
        "proper_time_maximum_hazard",
        "diffusion_steps",
        "diffusion_time_sampling",
        "sampler_trajectory_alignment",
        "gradient_checkpointing",
        "compile_model",
        "seed",
        "precision",
        "deterministic",
        "epochs",
        "steps_per_epoch",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "validation_interval",
        "gradient_clip_norm",
        "ema_decay",
        "tc_loss_weight",
        "lpips_loss_weight",
        "diffusion_loss_weight",
        "keyframe_loss_weight",
        "residual_loss_weight",
        "transport_loss_weight",
        "flow_loss_weight",
        "visibility_loss_weight",
        "router_loss_weight",
        "survival_loss_weight",
        "lite_loss_weight",
        "budget_loss_weight",
        "warp_loss_weight",
        "drift_loss_weight",
        "crcd_loss_weight",
        "text_loss_weight",
        "lpips_frame_chunk_size",
        "curriculum_profile",
        "validation_size",
        "test_size",
        "split_seed",
    )
    return cast(
        dict[str, object],
        _jsonable(
            {
                "args": {name: getattr(args, name) for name in argument_names},
                "world_size": world_size,
                "factory_options": runtime.factory_options,
                "sources": source_identity,
                "metric_provenance": runtime.metric_suite.provenance_ids,
            }
        ),
    )


def _model_provenance(args: argparse.Namespace, model: nn.Module) -> dict[str, object]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "method": "TARDIS: Transport-Aligned Residual Diffusion in Innovation Subspaces",
        "theory": "motion-advected quotient renewal process",
        "core_operator": "transport-orbit quotient diffusion with innovation proper time",
        "transport_quotient": bool(args.transport_quotient),
        "innovation_proper_time": bool(args.innovation_proper_time),
        "proper_time_maximum_hazard": float(args.proper_time_maximum_hazard),
        "pretrained_model": str(args.pretrained_model),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
    }


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    paths: RunPaths,
    signature: Mapping[str, object],
    world_size: int,
) -> None:
    if checkpoint.get("run_id") != paths.run_id:
        raise ValueError("resume checkpoint run ID does not match its run directory")
    if int(cast(int, checkpoint.get("world_size", -1))) != world_size:
        raise ValueError("resume checkpoint world_size does not match the current torchrun")
    checkpoint_signature = checkpoint.get("training_signature")
    if checkpoint_signature != _jsonable(signature):
        raise ValueError("resume checkpoint training signature is incompatible")
    rank_states = checkpoint.get("rank_random_states")
    if not isinstance(rank_states, Mapping) or set(rank_states) != {
        str(rank) for rank in range(world_size)
    }:
        raise ValueError("resume checkpoint does not contain every rank random state")


def _restore_rank_random_state(
    engine: TrainEngine,
    checkpoint: Mapping[str, object],
    context: DistributedContext,
) -> None:
    rank_states = cast(Mapping[object, object], checkpoint["rank_random_states"])
    rank_state = rank_states[str(context.rank)]
    if not isinstance(rank_state, Mapping):
        raise ValueError(f"resume checkpoint rank {context.rank} random state is invalid")
    engine.load_stochastic_state_dict(cast(Mapping[str, object], rank_state))


def _broadcast_run_paths(
    paths: RunPaths | None,
    context: DistributedContext,
) -> RunPaths:
    if context.world_size <= 1:
        if paths is None:
            raise RuntimeError("rank zero did not allocate train run paths")
        return paths
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("run path broadcast requires an initialized process group")
    payload: list[object] = [paths]
    dist.broadcast_object_list(payload, src=0)
    received = payload[0]
    if not isinstance(received, RunPaths):
        raise RuntimeError("rank zero broadcast an invalid train run path payload")
    return received


def _enable_cuda_training_defaults(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, torch.device | torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=str)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the production train command under Python or torchrun."""

    run_training(parse_args(argv))
    return 0


__all__ = [
    "build_train_dataloaders",
    "clip_batch_to_training_batch",
    "curriculum_durations",
    "ema_temporal_parameters",
    "evaluate_validation",
    "train_engine_options_from_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
