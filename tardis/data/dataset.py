"""Deterministic bounded-memory clip datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import IterableDataset, get_worker_info

from tardis.data.adapters import read_record_media, stream_oversized_local_media
from tardis.data.contracts import ObjectTooLargeError, VideoRecord
from tardis.data.http_range import RangeClient
from tardis.data.sampler import (
    BalancedVirtualEpoch,
    CyclicTrainingEpoch,
    FullTrainingEpoch,
    SampleRequest,
)
from tardis.data.splits import StablePartition
from tardis.data.video import (
    DecodedVideo,
    decode_sampled_video_bytes,
    decode_sampled_video_path,
    prepare_clip,
)

SplitName = Literal["train", "validation", "test"]
TemporalDecodeMode = Literal["window", "uniform"]


@dataclass(frozen=True, slots=True)
class ClipExample:
    record: VideoRecord
    video: torch.Tensor
    sample_seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    """One exhausted benchmark media load that must not stop the source stream."""

    record: VideoRecord
    sample_seed: int
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ClipBatch:
    prompts: list[str]
    video: torch.Tensor
    sources: tuple[str, ...]
    record_ids: tuple[str, ...]
    sample_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClipDecodeOptions:
    num_frames: int
    height: int
    width: int
    mode: Literal["train", "benchmark"]
    max_media_bytes: int = 128 * 1024 * 1024
    max_decoded_bytes: int = 512 * 1024 * 1024
    timeout_seconds: float = 120.0
    random_flip: bool = True

    def __post_init__(self) -> None:
        if min(self.num_frames, self.height, self.width) <= 0:
            raise ValueError("clip dimensions must be positive")
        if min(self.max_media_bytes, self.max_decoded_bytes) <= 0:
            raise ValueError("clip byte limits must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("clip timeout must be positive")


@dataclass(frozen=True, slots=True)
class RemoteClipLoader:
    """Fetch and decode exactly one selected record in bounded RAM."""

    options: ClipDecodeOptions

    def __call__(
        self,
        client: RangeClient,
        record: VideoRecord,
        request: SampleRequest,
    ) -> torch.Tensor:
        temporal_mode: TemporalDecodeMode = "window"
        try:
            payload = read_record_media(
                client,
                record,
                max_media_bytes=self.options.max_media_bytes,
            )
        except ObjectTooLargeError:
            with stream_oversized_local_media(
                client,
                record,
                max_media_bytes=self.options.max_media_bytes,
            ) as media_path:
                decoded = self._decode_path(media_path, temporal_mode, request.sample_seed)
        else:
            decoded = decode_sampled_video_bytes(
                payload,
                num_frames=self.options.num_frames,
                mode=temporal_mode,
                seed=request.sample_seed,
                max_decoded_bytes=self.options.max_decoded_bytes,
                timeout_seconds=self.options.timeout_seconds,
            )
        return prepare_clip(
            decoded,
            num_frames=self.options.num_frames,
            height=self.options.height,
            width=self.options.width,
            mode=self.options.mode,
            seed=request.sample_seed,
            random_flip=self.options.random_flip and self.options.mode == "train",
        )

    def _decode_path(
        self,
        path: Path,
        temporal_mode: TemporalDecodeMode,
        sample_seed: int,
    ) -> DecodedVideo:
        return decode_sampled_video_path(
            path,
            num_frames=self.options.num_frames,
            mode=temporal_mode,
            seed=sample_seed,
            max_decoded_bytes=self.options.max_decoded_bytes,
            timeout_seconds=self.options.timeout_seconds,
        )


ClipLoader = Callable[[RangeClient, VideoRecord, SampleRequest], torch.Tensor]


class RemoteClipIterableDataset(IterableDataset[ClipExample]):
    """Resolve a rank-sharded training schedule into decoded prompt-video clips."""

    def __init__(
        self,
        records_by_source: Mapping[str, Sequence[VideoRecord]],
        *,
        schedule: BalancedVirtualEpoch | CyclicTrainingEpoch | FullTrainingEpoch,
        epoch: int,
        rank: int,
        world_size: int,
        client_factory: Callable[[], RangeClient],
        clip_loader: ClipLoader,
        max_retries: int = 2,
    ) -> None:
        super().__init__()
        if set(records_by_source) != set(schedule.source_sizes):
            raise ValueError("dataset sources must match the training schedule")
        size_mismatch = any(
            len(records) != schedule.source_sizes[source]
            for source, records in records_by_source.items()
        )
        if size_mismatch:
            raise ValueError("dataset source lengths must match the training schedule")
        if epoch < 0 or max_retries < 0:
            raise ValueError("epoch and max_retries cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if schedule.global_batch_size % world_size:
            raise ValueError("global batch size must be divisible by world size")
        self.records_by_source = {
            source: tuple(records) for source, records in records_by_source.items()
        }
        self.schedule = schedule
        self._epoch = torch.tensor(epoch, dtype=torch.int64)
        self._epoch.share_memory_()  # type: ignore[no-untyped-call]
        self._start_batch = torch.tensor(0, dtype=torch.int64)
        self._start_batch.share_memory_()  # type: ignore[no-untyped-call]
        self.rank = rank
        self.world_size = world_size
        self.local_batch_size = schedule.global_batch_size // world_size
        self.client_factory = client_factory
        self.clip_loader = clip_loader
        self.max_retries = max_retries

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self._epoch.fill_(epoch)

    def set_start_batch(self, batch_index: int) -> None:
        if not 0 <= batch_index <= self.schedule.steps_per_epoch:
            raise ValueError("start batch must be within the training epoch")
        self._start_batch.fill_(batch_index)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    @property
    def start_batch(self) -> int:
        return int(self._start_batch.item())

    def __len__(self) -> int:
        total = self.schedule.steps_per_epoch * self.schedule.global_batch_size
        if total % self.world_size:
            raise ValueError("training epoch size must be divisible by world size")
        return total // self.world_size

    def __iter__(self) -> Iterator[ClipExample]:
        client = self.client_factory()
        try:
            for request in self._worker_requests():
                yield self._load_with_replacement(client, request)
        finally:
            client.close()

    def _worker_requests(self) -> Iterator[SampleRequest]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        requests = self.schedule.requests(
            epoch=self.epoch,
            rank=self.rank,
            world_size=self.world_size,
        )
        start_batch = self.start_batch
        for local_position, request in enumerate(requests):
            local_batch_index = local_position // self.local_batch_size
            if local_batch_index < start_batch:
                continue
            if local_batch_index % worker_count == worker_id:
                yield request

    def _load_with_replacement(
        self,
        client: RangeClient,
        request: SampleRequest,
    ) -> ClipExample:
        last_error: Exception | None = None
        candidate = request
        for attempt in range(self.max_retries + 1):
            if attempt:
                candidate = self.schedule.replacement(request, attempt=attempt)
            record = self.records_by_source[candidate.source][candidate.source_index]
            try:
                clip = self.clip_loader(client, record, candidate)
                _validate_clip(clip)
                return ClipExample(record, clip, candidate.sample_seed)
            except Exception as error:
                last_error = error
        raise RuntimeError(
            f"failed to load record for global position {request.global_position} "
            f"after {self.max_retries + 1} attempts"
        ) from last_error


class RemoteSourceClipIterableDataset(IterableDataset[ClipExample]):
    """Iterate one benchmark source exactly once across DDP ranks and workers."""

    def __init__(
        self,
        records: Sequence[VideoRecord],
        *,
        source: str,
        split: SplitName,
        seed: int,
        rank: int,
        world_size: int,
        client_factory: Callable[[], RangeClient],
        clip_loader: ClipLoader,
        max_retries: int = 2,
    ) -> None:
        super().__init__()
        if split not in ("validation", "test"):
            raise ValueError("source benchmark split must be validation or test")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        materialized = tuple(records)
        if any(record.source != source for record in materialized):
            raise ValueError(f"benchmark source {source!r} contains a foreign record")
        self.records = materialized
        self.source = source
        self.split = split
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.client_factory = client_factory
        self.clip_loader = clip_loader
        self.max_retries = max_retries
        self._excluded_record_ids: frozenset[str] = frozenset()

    def exclude_record_ids(self, record_ids: Sequence[str] | set[str]) -> None:
        """Skip records already completed by a resumable evaluator."""

        excluded = frozenset(str(record_id) for record_id in record_ids)
        known = {record.id for record in self.records}
        unknown = excluded - known
        if unknown:
            raise ValueError(f"cannot exclude unknown benchmark record IDs: {sorted(unknown)}")
        self._excluded_record_ids = excluded

    def __len__(self) -> int:
        return sum(
            self.records[index].id not in self._excluded_record_ids
            for index in range(self.rank, len(self.records), self.world_size)
        )

    def __iter__(self) -> Iterator[ClipExample]:
        client = self.client_factory()
        try:
            for source_index in self._worker_source_indices():
                yield self._load_with_retries(client, source_index)
        finally:
            client.close()

    def _worker_source_indices(self) -> Iterator[int]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rank_indices = (
            index
            for index in range(self.rank, len(self.records), self.world_size)
            if self.records[index].id not in self._excluded_record_ids
        )
        for local_position, source_index in enumerate(rank_indices):
            if local_position % worker_count == worker_id:
                yield source_index

    def _load_with_retries(self, client: RangeClient, source_index: int) -> ClipExample:
        record = self.records[source_index]
        sample_seed = _benchmark_sample_seed(
            self.seed,
            self.split,
            self.source,
            record.id,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = SampleRequest(
                source=self.source,
                source_index=source_index,
                sample_seed=sample_seed,
                global_position=source_index,
                epoch=0,
                replacement_attempt=attempt,
            )
            try:
                clip = self.clip_loader(client, record, request)
                _validate_clip(clip)
                return ClipExample(record, clip, sample_seed)
            except Exception as error:
                last_error = error
        raise RuntimeError(
            f"failed to load {self.split} record {record.id!r} "
            f"after {self.max_retries + 1} attempts"
        ) from last_error


class ResilientRemoteSourceClipIterableDataset(IterableDataset[ClipExample | BenchmarkFailure]):
    """Materialize exhausted benchmark loads as failure records and continue."""

    def __init__(self, dataset: RemoteSourceClipIterableDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[ClipExample | BenchmarkFailure]:
        client = self.dataset.client_factory()
        try:
            for source_index in self.dataset._worker_source_indices():
                try:
                    yield self.dataset._load_with_retries(client, source_index)
                except Exception as error:
                    record = self.dataset.records[source_index]
                    yield BenchmarkFailure(
                        record=record,
                        sample_seed=_benchmark_sample_seed(
                            self.dataset.seed,
                            self.dataset.split,
                            self.dataset.source,
                            record.id,
                        ),
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
        finally:
            client.close()


def build_split_records(
    records_by_source: Mapping[str, Sequence[VideoRecord]],
    partition: StablePartition,
) -> dict[SplitName, dict[str, tuple[VideoRecord, ...]]]:
    """Build exact source-local splits from compact in-memory metadata."""

    result: dict[SplitName, dict[str, tuple[VideoRecord, ...]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for source, source_records in sorted(records_by_source.items()):
        records = tuple(source_records)
        if any(record.source != source for record in records):
            raise ValueError(f"catalog for {source!r} contains a foreign record")
        selection = partition.select(iter(records))
        for split in result:
            result[split][source] = tuple(partition.iter_split(iter(records), selection, split))
    return result


def collate_clip_examples(examples: Sequence[ClipExample]) -> ClipBatch:
    if not examples:
        raise ValueError("cannot collate an empty clip batch")
    shape = examples[0].video.shape
    if any(example.video.shape != shape for example in examples):
        raise ValueError("all clips in a batch must share shape")
    return ClipBatch(
        prompts=[example.record.caption for example in examples],
        video=torch.stack([example.video for example in examples]),
        sources=tuple(example.record.source for example in examples),
        record_ids=tuple(example.record.id for example in examples),
        sample_seeds=tuple(example.sample_seed for example in examples),
    )


def collate_benchmark_items(
    items: Sequence[ClipExample | BenchmarkFailure],
) -> ClipBatch | BenchmarkFailure:
    """Collate successful clips while preserving one structured load failure."""

    if not items:
        raise ValueError("cannot collate an empty benchmark batch")
    failures = [item for item in items if isinstance(item, BenchmarkFailure)]
    if failures:
        if len(items) != 1:
            raise ValueError("benchmark failures require evaluation_batch_size=1")
        return failures[0]
    return collate_clip_examples([item for item in items if isinstance(item, ClipExample)])


def _validate_clip(clip: torch.Tensor) -> None:
    if clip.ndim != 4 or clip.shape[1] != 3:
        raise ValueError("clip loader must return [T,3,H,W]")
    if not clip.is_floating_point() or not bool(torch.isfinite(clip).all().item()):
        raise ValueError("clip loader must return finite floating-point values")
    if float(clip.min().item()) < -1 or float(clip.max().item()) > 1:
        raise ValueError("clip loader output must be normalized to [-1,1]")


def _benchmark_sample_seed(seed: int, split: str, source: str, record_id: str) -> int:
    payload = "\x1f".join((str(seed), split, source, record_id)).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"TARDISEval").digest()
    return int.from_bytes(digest, "big")
