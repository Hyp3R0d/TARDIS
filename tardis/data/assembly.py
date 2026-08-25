"""Production assembly for run-selected video DataLoaders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from torch.utils.data import DataLoader, IterableDataset

from tardis.data.adapters import DatasetAdapter
from tardis.data.catalog import (
    DEFAULT_SOURCE_SPECS,
    SourceSpec,
    build_adapters,
    build_local_adapters,
)
from tardis.data.contracts import VideoRecord
from tardis.data.dataset import (
    ClipBatch,
    ClipDecodeOptions,
    ClipExample,
    ClipLoader,
    RemoteClipIterableDataset,
    RemoteClipLoader,
    RemoteSourceClipIterableDataset,
    SplitName,
    build_split_records,
    collate_clip_examples,
)
from tardis.data.http_range import RangeClient
from tardis.data.sampler import CyclicTrainingEpoch, FullTrainingEpoch
from tardis.data.splits import StablePartition

SOURCE_NAMES = ("dataverse", "openvid", "seedance")
DEFAULT_CATALOG_CHUNK_BYTES = 16 * 1024 * 1024
ClientFactory = Callable[[], RangeClient]
AdapterBuilder = Callable[[RangeClient], Mapping[str, DatasetAdapter]]
ClipDataLoader = DataLoader[ClipBatch]
SplitRecords = dict[SplitName, dict[str, tuple[VideoRecord, ...]]]


@dataclass(frozen=True, slots=True)
class RangeClientFactory:
    """Picklable factory for one bounded byte-range client per iterator worker."""

    max_object_bytes: int = 128 * 1024 * 1024
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_object_bytes <= 0 or self.timeout_seconds <= 0:
            raise ValueError("remote byte and timeout limits must be positive")
        if self.max_retries < 0 or self.backoff_base_seconds < 0:
            raise ValueError("remote retries and backoff cannot be negative")

    def __call__(self) -> RangeClient:
        return RangeClient(
            max_object_bytes=self.max_object_bytes,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            backoff_base_seconds=self.backoff_base_seconds,
        )


@dataclass(frozen=True, slots=True)
class RemoteDataLoaderOptions:
    """Sampling and worker geometry shared by the assembled loaders."""

    # ``None`` is the production mode: the loader derives a complete epoch from all
    # training records. A positive value remains available for small deterministic tests.
    steps_per_epoch: int | None = None
    global_batch_size: int = 1
    evaluation_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 3407
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    max_sample_retries: int = 2
    multiprocessing_context: Literal["fork", "forkserver", "spawn"] | None = None

    def __post_init__(self) -> None:
        if self.steps_per_epoch is not None and self.steps_per_epoch <= 0:
            raise ValueError("loader steps must be positive when explicitly provided")
        if min(self.global_batch_size, self.evaluation_batch_size) <= 0:
            raise ValueError("loader batch sizes must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient accumulation steps must be positive")
        if self.num_workers < 0 or self.max_sample_retries < 0:
            raise ValueError("worker and sample retry counts cannot be negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")


@dataclass(frozen=True, slots=True)
class RemoteCatalog:
    """Canonical metadata for all sources or one explicitly selected source."""

    records_by_source: Mapping[str, tuple[VideoRecord, ...]]

    def __post_init__(self) -> None:
        received = set(self.records_by_source)
        if received != set(SOURCE_NAMES) and not (
            len(received) == 1 and received.issubset(SOURCE_NAMES)
        ):
            raise ValueError(
                "remote catalog must contain all canonical sources or exactly one "
                f"selected source from {list(SOURCE_NAMES)}"
            )
        normalized: dict[str, tuple[VideoRecord, ...]] = {}
        for source in SOURCE_NAMES:
            if source not in received:
                continue
            records = tuple(self.records_by_source[source])
            if not records:
                raise ValueError(f"remote catalog source {source!r} cannot be empty")
            if any(record.source != source for record in records):
                raise ValueError(f"remote catalog source {source!r} contains a foreign record")
            normalized[source] = records
        object.__setattr__(self, "records_by_source", MappingProxyType(normalized))


@dataclass(slots=True)
class RemoteDataLoaders:
    """Train and benchmark loaders for one run-local source selection."""

    catalog: RemoteCatalog
    splits: SplitRecords
    train_dataset: RemoteClipIterableDataset
    train: ClipDataLoader
    validation: dict[str, ClipDataLoader]
    test: dict[str, ClipDataLoader]
    selected_source: str | None = None

    @property
    def steps_per_epoch(self) -> int:
        """Resolved number of local training batches in one complete epoch."""

        return int(self.train_dataset.schedule.steps_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.train_dataset.set_epoch(epoch)

    def set_start_batch(self, batch_index: int) -> bool:
        setter = getattr(self.train_dataset, "set_start_batch", None)
        if not callable(setter):
            return False
        setter(batch_index)
        return True


def build_remote_catalog(
    *,
    client_factory: ClientFactory,
    endpoint: str = "https://hf-mirror.com",
    specs: Mapping[str, SourceSpec] = DEFAULT_SOURCE_SPECS,
    chunk_bytes: int = DEFAULT_CATALOG_CHUNK_BYTES,
    openvid_max_index_entries: int = 2_000_000,
    max_records_per_source: int | None = None,
    openvid_archive_limit: int | None = None,
    record_ids_by_source: Mapping[str, Sequence[str]] | None = None,
    dataset_roots: Mapping[str, Path | str] | None = None,
    adapter_builder: AdapterBuilder | None = None,
    selected_source: str | None = None,
) -> RemoteCatalog:
    """Materialize canonical metadata for all sources or one selected source."""

    if max_records_per_source is not None and max_records_per_source <= 0:
        raise ValueError("max_records_per_source must be positive when provided")
    if dataset_roots is not None and adapter_builder is not None:
        raise ValueError("dataset_roots and adapter_builder cannot be supplied together")
    selected_sources = _selected_sources(selected_source)
    requested_ids = _normalize_record_ids(record_ids_by_source)
    foreign_allowlists = set(requested_ids) - set(selected_sources)
    if selected_source is not None and foreign_allowlists:
        raise ValueError(
            f"record allowlists cannot target an unselected source: {sorted(foreign_allowlists)}"
        )
    client = client_factory()
    try:
        if adapter_builder is not None:
            adapters = adapter_builder(client)
        elif dataset_roots is not None:
            adapters = build_local_adapters(
                client,
                dataset_roots=dataset_roots,
                chunk_bytes=chunk_bytes,
                openvid_max_index_entries=openvid_max_index_entries,
                openvid_archive_limit=openvid_archive_limit,
            )
        else:
            adapters = build_adapters(
                client,
                endpoint=endpoint,
                specs=specs,
                chunk_bytes=chunk_bytes,
                openvid_max_index_entries=openvid_max_index_entries,
                openvid_archive_limit=openvid_archive_limit,
            )
        if set(adapters) != set(SOURCE_NAMES):
            raise ValueError(f"adapter catalog must contain exactly {list(SOURCE_NAMES)}")
        records: dict[str, tuple[VideoRecord, ...]] = {}
        for source in selected_sources:
            adapter = adapters[source]
            if adapter.source != source:
                raise ValueError(f"adapter key {source!r} does not match source {adapter.source!r}")
            stream = adapter.iter_records()
            source_ids = requested_ids.get(source)
            if source_ids is None:
                records[source] = tuple(
                    stream
                    if max_records_per_source is None
                    else islice(stream, max_records_per_source)
                )
                continue
            if max_records_per_source is not None and len(source_ids) > max_records_per_source:
                raise ValueError(f"record allowlist for {source!r} exceeds max_records_per_source")
            selected = tuple(
                islice(
                    (record for record in stream if record.id in source_ids),
                    len(source_ids),
                )
            )
            found = {record.id for record in selected}
            if found != source_ids:
                missing = sorted(source_ids - found)
                raise ValueError(
                    f"record allowlist for {source!r} was not found; missing={missing}"
                )
            records[source] = selected
        return RemoteCatalog(records)
    finally:
        client.close()


def build_remote_dataloaders(
    *,
    partition: StablePartition,
    train_clip_options: ClipDecodeOptions,
    evaluation_clip_options: ClipDecodeOptions,
    loader_options: RemoteDataLoaderOptions,
    rank: int = 0,
    world_size: int = 1,
    endpoint: str = "https://hf-mirror.com",
    specs: Mapping[str, SourceSpec] = DEFAULT_SOURCE_SPECS,
    catalog: RemoteCatalog | None = None,
    client_factory: ClientFactory | None = None,
    adapter_builder: AdapterBuilder | None = None,
    train_clip_loader: ClipLoader | None = None,
    evaluation_clip_loader: ClipLoader | None = None,
    catalog_chunk_bytes: int = DEFAULT_CATALOG_CHUNK_BYTES,
    openvid_max_index_entries: int = 2_000_000,
    catalog_record_limit: int | None = None,
    openvid_archive_limit: int | None = None,
    record_ids_by_source: Mapping[str, Sequence[str]] | None = None,
    dataset_roots: Mapping[str, Path | str] | None = None,
    selected_source: str | None = None,
) -> RemoteDataLoaders:
    """Assemble train/validation/test loaders over bounded local or remote reads."""

    _validate_geometry(loader_options, rank=rank, world_size=world_size)
    selected_sources = _selected_sources(selected_source)
    if train_clip_options.mode != "train":
        raise ValueError("train_clip_options must use train mode")
    if evaluation_clip_options.mode != "benchmark":
        raise ValueError("evaluation_clip_options must use benchmark mode")
    if catalog is not None and (adapter_builder is not None or dataset_roots is not None):
        raise ValueError(
            "adapter_builder and dataset_roots cannot be supplied with an existing catalog"
        )

    if client_factory is None:
        client_factory = RangeClientFactory(
            max_object_bytes=max(
                train_clip_options.max_media_bytes,
                evaluation_clip_options.max_media_bytes,
                catalog_chunk_bytes,
            ),
            timeout_seconds=max(
                train_clip_options.timeout_seconds,
                evaluation_clip_options.timeout_seconds,
            ),
        )
    if catalog is None:
        catalog = build_remote_catalog(
            client_factory=client_factory,
            endpoint=endpoint,
            specs=specs,
            chunk_bytes=catalog_chunk_bytes,
            openvid_max_index_entries=openvid_max_index_entries,
            max_records_per_source=catalog_record_limit,
            openvid_archive_limit=openvid_archive_limit,
            record_ids_by_source=record_ids_by_source,
            dataset_roots=dataset_roots,
            adapter_builder=adapter_builder,
            selected_source=selected_source,
        )

    missing_sources = set(selected_sources) - set(catalog.records_by_source)
    if missing_sources:
        raise ValueError(f"selected source is absent from the catalog: {sorted(missing_sources)}")
    active_records = {source: catalog.records_by_source[source] for source in selected_sources}
    splits = build_split_records(active_records, partition)
    train_records = splits["train"]
    if any(not train_records[source] for source in selected_sources):
        raise ValueError("every selected source must retain at least one training record")
    source_sizes = {source: len(train_records[source]) for source in selected_sources}
    schedule: FullTrainingEpoch | CyclicTrainingEpoch
    if loader_options.steps_per_epoch is None:
        schedule = FullTrainingEpoch(
            source_sizes,
            global_batch_size=loader_options.global_batch_size,
            gradient_accumulation_steps=loader_options.gradient_accumulation_steps,
            seed=loader_options.seed,
        )
    else:
        schedule = CyclicTrainingEpoch(
            source_sizes,
            steps_per_epoch=loader_options.steps_per_epoch,
            global_batch_size=loader_options.global_batch_size,
            seed=loader_options.seed,
        )
    train_dataset = RemoteClipIterableDataset(
        train_records,
        schedule=schedule,
        epoch=0,
        rank=rank,
        world_size=world_size,
        client_factory=client_factory,
        clip_loader=(
            RemoteClipLoader(train_clip_options) if train_clip_loader is None else train_clip_loader
        ),
        max_retries=loader_options.max_sample_retries,
    )
    local_train_batch = loader_options.global_batch_size // world_size
    train_loader = _make_loader(
        train_dataset,
        batch_size=local_train_batch,
        options=loader_options,
        drop_last=True,
    )

    benchmark_loader = (
        RemoteClipLoader(evaluation_clip_options)
        if evaluation_clip_loader is None
        else evaluation_clip_loader
    )
    validation = _build_evaluation_loaders(
        splits["validation"],
        split="validation",
        rank=rank,
        world_size=world_size,
        client_factory=client_factory,
        clip_loader=benchmark_loader,
        options=loader_options,
        source_names=selected_sources,
    )
    test = _build_evaluation_loaders(
        splits["test"],
        split="test",
        rank=rank,
        world_size=world_size,
        client_factory=client_factory,
        clip_loader=benchmark_loader,
        options=loader_options,
        source_names=selected_sources,
    )
    return RemoteDataLoaders(
        catalog,
        splits,
        train_dataset,
        train_loader,
        validation,
        test,
        selected_source,
    )


def _selected_sources(selected_source: str | None) -> tuple[str, ...]:
    if selected_source is None:
        return SOURCE_NAMES
    if selected_source not in SOURCE_NAMES:
        raise ValueError(
            f"selected_source must be one of {list(SOURCE_NAMES)}; got {selected_source!r}"
        )
    return (selected_source,)


def _normalize_record_ids(
    values: Mapping[str, Sequence[str]] | None,
) -> dict[str, frozenset[str]]:
    if values is None:
        return {}
    unknown = set(values) - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"record allowlists contain unknown sources: {sorted(unknown)}")
    normalized: dict[str, frozenset[str]] = {}
    for source, raw_ids in values.items():
        record_ids = tuple(str(record_id).strip() for record_id in raw_ids)
        if not record_ids or any(not record_id for record_id in record_ids):
            raise ValueError(f"record allowlist for {source!r} must be non-empty")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError(f"record allowlist for {source!r} contains duplicates")
        normalized[source] = frozenset(record_ids)
    return normalized


def _build_evaluation_loaders(
    records_by_source: Mapping[str, tuple[VideoRecord, ...]],
    *,
    split: SplitName,
    rank: int,
    world_size: int,
    client_factory: ClientFactory,
    clip_loader: ClipLoader,
    options: RemoteDataLoaderOptions,
    source_names: Sequence[str],
) -> dict[str, ClipDataLoader]:
    result: dict[str, ClipDataLoader] = {}
    for source in source_names:
        dataset = RemoteSourceClipIterableDataset(
            records_by_source[source],
            source=source,
            split=split,
            seed=options.seed,
            rank=rank,
            world_size=world_size,
            client_factory=client_factory,
            clip_loader=clip_loader,
            max_retries=options.max_sample_retries,
        )
        result[source] = _make_loader(
            dataset,
            batch_size=options.evaluation_batch_size,
            options=options,
            drop_last=False,
        )
    return result


def _make_loader(
    dataset: IterableDataset[ClipExample],
    *,
    batch_size: int,
    options: RemoteDataLoaderOptions,
    drop_last: bool,
) -> ClipDataLoader:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=options.num_workers,
        collate_fn=collate_clip_examples,
        pin_memory=options.pin_memory,
        drop_last=drop_last,
        prefetch_factor=options.prefetch_factor if options.num_workers else None,
        persistent_workers=options.persistent_workers and options.num_workers > 0,
        multiprocessing_context=(
            options.multiprocessing_context if options.num_workers > 0 else None
        ),
    )
    return cast(ClipDataLoader, loader)


def _validate_geometry(
    options: RemoteDataLoaderOptions,
    *,
    rank: int,
    world_size: int,
) -> None:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if options.global_batch_size % world_size:
        raise ValueError("global_batch_size must be divisible by world_size")
