from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from tardis.data.contracts import VideoRecord
from tardis.data.dataset import ClipDecodeOptions
from tardis.data.splits import StablePartition

SOURCES = ("dataverse", "openvid", "seedance")


def _assembly() -> ModuleType:
    return importlib.import_module("tardis.data.assembly")


def _record(source: str, index: int) -> VideoRecord:
    return VideoRecord(
        id=f"{source}-{index}",
        caption=f"prompt {source} {index}",
        media_locator=f"https://media.test/{source}/{index}.mp4",
        source=source,
        metadata={"revision": "1" * 40},
    )


def _records(count: int = 8) -> dict[str, tuple[VideoRecord, ...]]:
    return {source: tuple(_record(source, index) for index in range(count)) for source in SOURCES}


class _Adapter:
    def __init__(self, source: str, records: Sequence[VideoRecord]) -> None:
        self.source = source
        self.revision = "1" * 40
        self._records = tuple(records)

    def iter_records(self) -> Iterator[VideoRecord]:
        yield from self._records


class _Client:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ClientFactory:
    def __init__(self) -> None:
        self.clients: list[_Client] = []

    def __call__(self) -> _Client:
        client = _Client()
        self.clients.append(client)
        return client


class _ClipLoader:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, _client: object, record: VideoRecord, request: Any) -> torch.Tensor:
        self.calls.append((record.id, request.global_position, request.replacement_attempt))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("synthetic media failure")
        value = (request.sample_seed % 101) / 100.0
        return torch.full((2, 3, 4, 4), value)


def _adapter_builder(
    records: Mapping[str, Sequence[VideoRecord]],
) -> Any:
    def build(_client: object) -> dict[str, _Adapter]:
        return {
            source: _Adapter(source, source_records) for source, source_records in records.items()
        }

    return build


def _clip_options(mode: str) -> ClipDecodeOptions:
    return ClipDecodeOptions(
        num_frames=2,
        height=4,
        width=4,
        mode=mode,  # type: ignore[arg-type]
        max_media_bytes=1024,
        max_decoded_bytes=4096,
        random_flip=mode == "train",
    )


def _loader_options(
    module: ModuleType,
    *,
    steps: int = 2,
    global_batch_size: int = 3,
    max_sample_retries: int = 2,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    persistent_workers: bool = True,
) -> Any:
    return module.RemoteDataLoaderOptions(
        steps_per_epoch=steps,
        global_batch_size=global_batch_size,
        evaluation_batch_size=2,
        seed=3407,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        max_sample_retries=max_sample_retries,
        multiprocessing_context=multiprocessing_context,
    )


def _build(
    module: ModuleType,
    *,
    records: Mapping[str, Sequence[VideoRecord]] | None = None,
    catalog: Any = None,
    rank: int = 0,
    world_size: int = 1,
    options: Any = None,
    client_factory: Any = None,
    clip_loader: Any = None,
    selected_source: str | None = None,
) -> Any:
    records = _records() if records is None else records
    client_factory = _ClientFactory() if client_factory is None else client_factory
    clip_loader = _ClipLoader() if clip_loader is None else clip_loader
    options = _loader_options(module) if options is None else options
    return module.build_remote_dataloaders(
        partition=StablePartition(seed=19, validation_size=2, test_size=2),
        train_clip_options=_clip_options("train"),
        evaluation_clip_options=_clip_options("benchmark"),
        loader_options=options,
        rank=rank,
        world_size=world_size,
        catalog=catalog,
        client_factory=client_factory,
        adapter_builder=None if catalog is not None else _adapter_builder(records),
        train_clip_loader=clip_loader,
        evaluation_clip_loader=clip_loader,
        selected_source=selected_source,
    )


@pytest.mark.unit
def test_catalog_materialization_is_three_source_metadata_only_and_closes_client() -> None:
    module = _assembly()
    clients = _ClientFactory()
    media_loader = _ClipLoader()

    bundle = _build(module, client_factory=clients, clip_loader=media_loader)

    assert tuple(bundle.catalog.records_by_source) == SOURCES
    assert all(len(bundle.catalog.records_by_source[source]) == 8 for source in SOURCES)
    assert media_loader.calls == []
    assert len(clients.clients) == 1
    assert clients.clients[0].closed

    batches = list(bundle.train)
    assert sum(len(batch.record_ids) for batch in batches) == 6
    assert len(media_loader.calls) == 6
    assert all(client.closed for client in clients.clients)


@pytest.mark.unit
def test_selected_source_builds_single_dataset_train_validation_and_test_loaders() -> None:
    module = _assembly()

    bundle = _build(module, selected_source="openvid")

    assert tuple(bundle.train_dataset.records_by_source) == ("openvid",)
    assert tuple(bundle.validation) == ("openvid",)
    assert tuple(bundle.test) == ("openvid",)
    assert (
        set(bundle.splits["train"])
        == set(bundle.splits["validation"])
        == set(bundle.splits["test"])
        == {"openvid"}
    )
    assert {source for batch in bundle.train for source in batch.sources} == {"openvid"}


@pytest.mark.unit
def test_selected_source_rejects_unknown_dataset() -> None:
    module = _assembly()

    with pytest.raises(ValueError, match="selected_source"):
        _build(module, selected_source="unknown")


@pytest.mark.unit
def test_default_train_loader_is_one_complete_shuffled_training_epoch() -> None:
    module = _assembly()
    options = module.RemoteDataLoaderOptions(
        steps_per_epoch=None,
        global_batch_size=3,
        evaluation_batch_size=2,
        gradient_accumulation_steps=2,
        seed=3407,
        num_workers=0,
        pin_memory=False,
    )

    bundle = _build(module, options=options)
    record_ids = [record_id for batch in bundle.train for record_id in batch.record_ids]
    expected = {
        record.id for source_records in bundle.splits["train"].values() for record in source_records
    }

    assert bundle.steps_per_epoch == 4
    assert len(record_ids) == 12
    assert len(set(record_ids)) == 12
    assert set(record_ids) == expected


@pytest.mark.unit
def test_catalog_materialization_can_bound_each_source_before_media_reads() -> None:
    module = _assembly()
    clients = _ClientFactory()

    catalog = module.build_remote_catalog(
        client_factory=clients,
        adapter_builder=_adapter_builder(_records(8)),
        max_records_per_source=5,
    )

    assert all(len(catalog.records_by_source[source]) == 5 for source in SOURCES)
    assert len(clients.clients) == 1
    assert clients.clients[0].closed


@pytest.mark.unit
def test_catalog_materialization_can_select_explicit_source_record_ids() -> None:
    module = _assembly()

    catalog = module.build_remote_catalog(
        client_factory=_ClientFactory(),
        adapter_builder=_adapter_builder(_records(8)),
        max_records_per_source=5,
        record_ids_by_source={"dataverse": ("dataverse-7", "dataverse-2")},
    )

    assert [record.id for record in catalog.records_by_source["dataverse"]] == [
        "dataverse-2",
        "dataverse-7",
    ]
    assert len(catalog.records_by_source["openvid"]) == 5
    assert len(catalog.records_by_source["seedance"]) == 5


@pytest.mark.unit
def test_cyclic_train_loader_ddp_shards_reconstruct_serial_deterministically() -> None:
    module = _assembly()
    records = _records(10)
    catalog = module.build_remote_catalog(
        client_factory=_ClientFactory(),
        adapter_builder=_adapter_builder(records),
    )
    options = _loader_options(module, steps=4, global_batch_size=6)

    serial = _build(module, catalog=catalog, options=options, world_size=1)
    rank_zero = _build(module, catalog=catalog, options=options, rank=0, world_size=2)
    rank_one = _build(module, catalog=catalog, options=options, rank=1, world_size=2)
    repeated_rank_zero = _build(module, catalog=catalog, options=options, rank=0, world_size=2)

    serial_batches = list(serial.train)
    zero_batches = list(rank_zero.train)
    one_batches = list(rank_one.train)
    repeated_batches = list(repeated_rank_zero.train)
    serial_seeds = [seed for batch in serial_batches for seed in batch.sample_seeds]
    zero_seeds = [seed for batch in zero_batches for seed in batch.sample_seeds]
    one_seeds = [seed for batch in one_batches for seed in batch.sample_seeds]
    reconstructed = [seed for pair in zip(zero_seeds, one_seeds, strict=True) for seed in pair]
    serial_ids = [record_id for batch in serial_batches for record_id in batch.record_ids]
    expected_ids = {
        record.id for source_records in serial.splits["train"].values() for record in source_records
    }

    assert reconstructed == serial_seeds
    assert zero_seeds == [seed for batch in repeated_batches for seed in batch.sample_seeds]
    assert len(set(serial_ids[:18])) == 18
    assert set(serial_ids[:18]) == expected_ids
    assert len(serial_batches) == len(zero_batches) == len(one_batches) == 4


@pytest.mark.unit
def test_persistent_train_workers_observe_set_epoch_deterministically() -> None:
    module = _assembly()
    options = _loader_options(
        module,
        steps=4,
        global_batch_size=3,
        num_workers=2,
    )
    bundle = _build(module, options=options)

    epoch_zero = sorted(seed for batch in bundle.train for seed in batch.sample_seeds)
    bundle.set_epoch(1)
    epoch_one = sorted(seed for batch in bundle.train for seed in batch.sample_seeds)
    expected = sorted(
        request.sample_seed
        for request in bundle.train_dataset.schedule.requests(epoch=1, rank=0, world_size=1)
    )

    assert epoch_one == expected
    assert epoch_one != epoch_zero


def test_persistent_train_workers_seek_to_resume_batch() -> None:
    module = _assembly()
    options = _loader_options(
        module,
        steps=4,
        global_batch_size=3,
        num_workers=2,
    )
    bundle = _build(module, options=options)
    bundle.set_epoch(1)
    bundle.set_start_batch(2)

    resumed = [seed for batch in bundle.train for seed in batch.sample_seeds]
    expected = [
        request.sample_seed
        for request in bundle.train_dataset.schedule.requests(epoch=1, rank=0, world_size=1)
    ][6:]

    assert resumed == expected
    assert len(resumed) == 6


@pytest.mark.unit
@pytest.mark.parametrize(
    ("steps", "global_batch_size", "world_size", "rank", "num_workers"),
    [
        (2, 2, 1, 0, 8),
        (5, 4, 2, 1, 3),
    ],
)
def test_spawn_train_workers_receive_complete_rank_local_batches(
    steps: int,
    global_batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    module = _assembly()
    records = _records(10)
    catalog = module.build_remote_catalog(
        client_factory=_ClientFactory(),
        adapter_builder=_adapter_builder(records),
    )
    options = _loader_options(
        module,
        steps=steps,
        global_batch_size=global_batch_size,
        num_workers=num_workers,
        multiprocessing_context="spawn",
        persistent_workers=False,
    )
    bundle = _build(
        module,
        catalog=catalog,
        options=options,
        rank=rank,
        world_size=world_size,
    )
    expected = [
        request.sample_seed
        for request in bundle.train_dataset.schedule.requests(
            epoch=0,
            rank=rank,
            world_size=world_size,
        )
    ]

    batches = list(bundle.train)
    actual = [seed for batch in batches for seed in batch.sample_seeds]

    assert len(batches) == steps
    assert all(len(batch.sample_seeds) == global_batch_size // world_size for batch in batches)
    assert actual == expected
    assert len(actual) == len(set(actual))


@pytest.mark.unit
def test_validation_and_test_loaders_cover_each_source_once_across_ddp_ranks() -> None:
    module = _assembly()
    records = _records(9)
    catalog = module.build_remote_catalog(
        client_factory=_ClientFactory(),
        adapter_builder=_adapter_builder(records),
    )
    options = _loader_options(module, global_batch_size=4)
    rank_zero = _build(module, catalog=catalog, rank=0, world_size=2, options=options)
    rank_one = _build(module, catalog=catalog, rank=1, world_size=2, options=options)

    for split_name in ("validation", "test"):
        expected_by_source = rank_zero.splits[split_name]
        loaders_zero = getattr(rank_zero, split_name)
        loaders_one = getattr(rank_one, split_name)
        assert tuple(loaders_zero) == tuple(loaders_one) == SOURCES
        for source in SOURCES:
            ids_zero = [
                record_id for batch in loaders_zero[source] for record_id in batch.record_ids
            ]
            ids_one = [record_id for batch in loaders_one[source] for record_id in batch.record_ids]
            expected = [record.id for record in expected_by_source[source]]
            assert set(ids_zero).isdisjoint(ids_one)
            assert sorted((*ids_zero, *ids_one)) == sorted(expected)


@pytest.mark.unit
def test_train_loader_uses_existing_deterministic_same_source_replacement() -> None:
    module = _assembly()
    loader = _ClipLoader(fail_first=True)
    options = _loader_options(
        module,
        steps=1,
        global_batch_size=1,
        max_sample_retries=1,
    )

    bundle = _build(module, options=options, clip_loader=loader)
    batches = list(bundle.train)

    assert len(batches) == 1
    assert len(loader.calls) == 2
    assert loader.calls[0][0] != loader.calls[1][0]
    assert loader.calls[0][2] == 0
    assert loader.calls[1][2] == 1
    assert loader.calls[0][0].split("-", maxsplit=1)[0] == batches[0].sources[0]


@pytest.mark.unit
def test_loader_assembly_rejects_invalid_ddp_and_noncanonical_catalogs() -> None:
    module = _assembly()
    options = _loader_options(module, global_batch_size=3)

    with pytest.raises(ValueError, match="divisible"):
        _build(module, options=options, rank=0, world_size=2)

    incomplete = {source: records for source, records in _records().items() if source != "seedance"}
    with pytest.raises(ValueError, match="exactly"):
        module.build_remote_catalog(
            client_factory=_ClientFactory(),
            adapter_builder=_adapter_builder(incomplete),
        )


@pytest.mark.unit
def test_public_data_package_exports_production_assembly() -> None:
    import tardis.data as data

    module = _assembly()

    assert data.build_remote_dataloaders is module.build_remote_dataloaders
    assert data.RemoteDataLoaders is module.RemoteDataLoaders
