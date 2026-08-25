from __future__ import annotations

from collections import Counter

from tardis.data.sampler import (
    BalancedVirtualEpoch,
    CyclicTrainingEpoch,
    FullTrainingEpoch,
    SampleRequest,
)

SOURCE_SIZES = {"dataverse": 10_000, "openvid": 20_000, "seedance": 8_100}


def test_virtual_epoch_has_fixed_steps_and_balanced_global_sources() -> None:
    schedule = BalancedVirtualEpoch(
        SOURCE_SIZES,
        steps_per_epoch=120,
        global_batch_size=8,
        seed=3407,
    )

    requests = list(schedule.requests(epoch=3, rank=0, world_size=1))

    assert len(requests) == 120 * 8
    counts = Counter(request.source for request in requests)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(0 <= request.source_index < SOURCE_SIZES[request.source] for request in requests)


def test_ddp_rank_shards_reconstruct_serial_schedule() -> None:
    schedule = BalancedVirtualEpoch(
        SOURCE_SIZES,
        steps_per_epoch=7,
        global_batch_size=8,
        seed=3407,
    )

    serial = list(schedule.requests(epoch=5, rank=0, world_size=1))
    rank_shards = [list(schedule.requests(epoch=5, rank=rank, world_size=4)) for rank in range(4)]
    reconstructed = sorted(
        (request for shard in rank_shards for request in shard),
        key=lambda request: request.global_position,
    )

    assert reconstructed == serial
    assert all(len(shard) == len(serial) // 4 for shard in rank_shards)


def test_schedule_is_repeatable_and_epoch_specific() -> None:
    schedule = BalancedVirtualEpoch(
        SOURCE_SIZES,
        steps_per_epoch=4,
        global_batch_size=3,
        seed=3407,
    )

    first = list(schedule.requests(epoch=0, rank=0, world_size=1))
    repeated = list(schedule.requests(epoch=0, rank=0, world_size=1))
    next_epoch = list(schedule.requests(epoch=1, rank=0, world_size=1))

    assert first == repeated
    assert first != next_epoch


def test_schedule_uses_tardis_hash_namespace_fixed_vector() -> None:
    schedule = BalancedVirtualEpoch(
        SOURCE_SIZES,
        steps_per_epoch=1,
        global_batch_size=1,
        seed=3407,
    )

    request = next(schedule.requests(epoch=0, rank=0, world_size=1))

    assert request.source == "seedance"
    assert request.source_index == 1452
    assert request.sample_seed == 3_029_689_747_061_361_044


def test_failure_replacement_is_deterministic_and_changes_source_index() -> None:
    schedule = BalancedVirtualEpoch(
        SOURCE_SIZES,
        steps_per_epoch=1,
        global_batch_size=3,
        seed=3407,
    )
    request = next(schedule.requests(epoch=0, rank=0, world_size=1))

    replacement = schedule.replacement(request, attempt=1)

    assert isinstance(replacement, SampleRequest)
    assert replacement == schedule.replacement(request, attempt=1)
    assert replacement.source == request.source
    assert replacement.source_index != request.source_index
    assert replacement.sample_seed != request.sample_seed


def test_invalid_ddp_or_batch_geometry_is_rejected() -> None:
    schedule = BalancedVirtualEpoch(
        SOURCE_SIZES,
        steps_per_epoch=2,
        global_batch_size=3,
        seed=3407,
    )

    try:
        list(schedule.requests(epoch=0, rank=0, world_size=2))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-divisible global batch must fail")


def test_full_training_epoch_consumes_each_record_once_before_minimal_padding() -> None:
    schedule = FullTrainingEpoch(
        {"dataverse": 3, "openvid": 2, "seedance": 2},
        global_batch_size=2,
        gradient_accumulation_steps=2,
        seed=3407,
    )

    requests = list(schedule.requests(epoch=0, rank=0, world_size=1))
    identities = [(request.source, request.source_index) for request in requests]

    assert schedule.steps_per_epoch == 4
    assert schedule.total_records == 8
    assert schedule.padding_records == 1
    assert len(identities) == 8
    assert set(identities[:7]) == {
        (source, index)
        for source, size in {"dataverse": 3, "openvid": 2, "seedance": 2}.items()
        for index in range(size)
    }
    assert len(set(identities[:7])) == 7
    assert identities[7] in set(identities[:7])


def test_full_training_epoch_ddp_shards_reconstruct_complete_schedule() -> None:
    schedule = FullTrainingEpoch(
        {"dataverse": 5, "openvid": 4},
        global_batch_size=4,
        gradient_accumulation_steps=1,
        seed=17,
    )

    serial = list(schedule.requests(epoch=2, rank=0, world_size=1))
    rank_shards = [list(schedule.requests(epoch=2, rank=rank, world_size=2)) for rank in range(2)]
    reconstructed = sorted(
        (request for shard in rank_shards for request in shard),
        key=lambda request: request.global_position,
    )

    assert reconstructed == serial
    assert len(serial) == 12
    assert len(rank_shards[0]) == len(rank_shards[1]) == 6


def test_cyclic_training_epochs_do_not_repeat_before_complete_dataset_coverage() -> None:
    source_sizes = {"dataverse": 5, "openvid": 3, "seedance": 2}
    schedule = CyclicTrainingEpoch(
        source_sizes,
        steps_per_epoch=2,
        global_batch_size=2,
        seed=3407,
    )

    first = list(schedule.requests(epoch=0, rank=0, world_size=1))
    second = list(schedule.requests(epoch=1, rank=0, world_size=1))
    third = list(schedule.requests(epoch=2, rank=0, world_size=1))
    first_cycle = first + second + third[:2]
    identities = [(request.source, request.source_index) for request in first_cycle]

    assert len(identities) == schedule.records_per_cycle == 10
    assert len(set(identities)) == 10
    assert set(identities) == {
        (source, index) for source, size in source_sizes.items() for index in range(size)
    }
    assert (third[2].source, third[2].source_index) in set(identities)


def test_cyclic_training_epoch_ddp_shards_reconstruct_serial_window() -> None:
    schedule = CyclicTrainingEpoch(
        {"dataverse": 7, "openvid": 5},
        steps_per_epoch=3,
        global_batch_size=4,
        seed=17,
    )

    serial = list(schedule.requests(epoch=3, rank=0, world_size=1))
    rank_shards = [list(schedule.requests(epoch=3, rank=rank, world_size=2)) for rank in range(2)]
    reconstructed = sorted(
        (request for shard in rank_shards for request in shard),
        key=lambda request: request.global_position,
    )

    assert reconstructed == serial
