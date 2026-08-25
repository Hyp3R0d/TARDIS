from __future__ import annotations

import random
import unicodedata

import pytest

from tardis.data.contracts import VideoRecord
from tardis.data.splits import StablePartition


def records(count: int) -> list[VideoRecord]:
    return [
        VideoRecord(
            id=f"record-{index:05d}",
            caption=f"caption {index}",
            media_locator=f"https://example.test/{index}.mp4",
            source="fixture",
            metadata={"revision": "rev-1"},
        )
        for index in range(count)
    ]


def test_exact_size_partitions_are_disjoint_and_exhaustive() -> None:
    partition = StablePartition(seed=3407, validation_size=512, test_size=2048)
    result = partition.partition(records(4000))

    train_ids = {record.id for record in result.train}
    validation_ids = {record.id for record in result.validation}
    test_ids = {record.id for record in result.test}
    assert len(validation_ids) == 512
    assert len(test_ids) == 2048
    assert train_ids.isdisjoint(validation_ids | test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert len(train_ids | validation_ids | test_ids) == 4000


def test_partitions_are_order_and_worker_independent() -> None:
    source = records(3200)
    shuffled = list(source)
    random.Random(99).shuffle(shuffled)
    partition = StablePartition(seed=3407, validation_size=512, test_size=2048)

    ordered = partition.partition(source)
    reordered = partition.partition(shuffled)

    assert [record.id for record in ordered.validation] == [
        record.id for record in reordered.validation
    ]
    assert [record.id for record in ordered.test] == [record.id for record in reordered.test]


def test_partition_hash_includes_revision_and_source() -> None:
    partition = StablePartition(seed=3407, validation_size=2, test_size=2)
    base = records(10)
    changed = [
        VideoRecord(
            id=record.id,
            caption=record.caption,
            media_locator=record.media_locator,
            source="different",
            metadata={"revision": "rev-2"},
        )
        for record in base
    ]

    base_result = partition.partition(base)
    changed_result = partition.partition(changed)

    assert {record.id for record in base_result.test} != {
        record.id for record in changed_result.test
    }


def test_partition_rejects_sources_too_small_for_benchmark() -> None:
    partition = StablePartition(seed=3407, validation_size=5, test_size=6)
    with pytest.raises(ValueError, match="at least 11"):
        partition.partition(records(10))


def test_streaming_selection_retains_only_compact_holdout_ids() -> None:
    source = records(4000)
    partition = StablePartition(seed=3407, validation_size=512, test_size=2048)

    selection = partition.select(iter(source))

    assert selection.total_records == 4000
    assert len(selection.validation_ids) == 512
    assert len(selection.test_ids) == 2048
    assert not hasattr(selection, "train")
    validation = list(partition.iter_split(iter(source), selection, "validation"))
    test = list(partition.iter_split(iter(source), selection, "test"))
    train = list(partition.iter_split(iter(source), selection, "train"))
    assert len(validation) == 512
    assert len(test) == 2048
    assert len(train) == 1440


def test_caption_group_partition_is_exact_disjoint_and_order_independent() -> None:
    captions = (
        "group-a",
        "group-a",
        "group-a",
        "group-b",
        "group-b",
        "group-c",
        "group-c",
        "group-d",
        "group-e",
        "group-f",
        "caf\u00e9",
        "cafe\u0301",
    )
    source = [
        VideoRecord(
            id=f"seedance-{index:02d}",
            caption=caption,
            media_locator=f"file:///videos/{index}.mp4",
            source="seedance",
            metadata={"revision": "rev-1"},
        )
        for index, caption in enumerate(captions)
    ]
    shuffled = list(source)
    random.Random(99).shuffle(shuffled)
    partition = StablePartition(
        seed=3407,
        validation_size=3,
        test_size=4,
        group_by_caption=True,
    )

    ordered = partition.partition(source)
    reordered = partition.partition(shuffled)

    assert len(ordered.validation) == 3
    assert len(ordered.test) == 4
    assert len(ordered.train) == 5
    assert [record.id for record in ordered.validation] == [
        record.id for record in reordered.validation
    ]
    assert [record.id for record in ordered.test] == [record.id for record in reordered.test]
    def caption_set(split: tuple[VideoRecord, ...]) -> set[str]:
        return {unicodedata.normalize("NFC", record.caption).strip() for record in split}

    train_captions = caption_set(ordered.train)
    validation_captions = caption_set(ordered.validation)
    test_captions = caption_set(ordered.test)
    assert train_captions.isdisjoint(validation_captions | test_captions)
    assert validation_captions.isdisjoint(test_captions)


def test_caption_group_partition_rejects_an_unreachable_exact_size() -> None:
    source = [
        VideoRecord(
            id=f"record-{index}",
            caption=f"group-{index // 2}",
            media_locator=f"file:///videos/{index}.mp4",
            source="seedance",
            metadata={"revision": "rev-1"},
        )
        for index in range(6)
    ]
    partition = StablePartition(
        seed=3407,
        validation_size=1,
        test_size=1,
        group_by_caption=True,
    )

    with pytest.raises(ValueError, match="cannot satisfy exact holdout size"):
        partition.partition(source)
