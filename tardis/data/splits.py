"""Exact-size, order-independent benchmark partitions."""

from __future__ import annotations

import hashlib
import heapq
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from tardis.data.contracts import VideoRecord


@dataclass(frozen=True, slots=True)
class PartitionedRecords:
    train: tuple[VideoRecord, ...]
    validation: tuple[VideoRecord, ...]
    test: tuple[VideoRecord, ...]


@dataclass(frozen=True, slots=True)
class PartitionSelection:
    """Compact holdout identity sets derived from one metadata pass."""

    validation_ids: frozenset[str]
    test_ids: frozenset[str]
    total_records: int


@dataclass(frozen=True, slots=True)
class StablePartition:
    """Select exact validation/test sizes with streaming bottom-k hashes."""

    seed: int
    validation_size: int
    test_size: int
    group_by_caption: bool = False

    def __post_init__(self) -> None:
        if self.validation_size < 0 or self.test_size < 0:
            raise ValueError("partition sizes cannot be negative")

    def partition(self, records: Iterable[VideoRecord]) -> PartitionedRecords:
        materialized = tuple(records)
        selection = self.select(iter(materialized))
        train = tuple(self.iter_split(iter(materialized), selection, "train"))
        validation = tuple(self.iter_split(iter(materialized), selection, "validation"))
        test = tuple(self.iter_split(iter(materialized), selection, "test"))
        return PartitionedRecords(
            train=tuple(sorted(train, key=_identity)),
            validation=tuple(sorted(validation, key=_identity)),
            test=tuple(sorted(test, key=_identity)),
        )

    def select(self, records: Iterable[VideoRecord]) -> PartitionSelection:
        """Make one bounded-memory pass to select exact holdout identities."""

        if self.group_by_caption:
            return self._select_caption_groups(records)

        required = self.validation_size + self.test_size
        selected: list[tuple[int, str]] = []
        seen_identities: set[str] = set()
        total_records = 0
        for record in records:
            identity = _identity(record)
            if identity in seen_identities:
                raise ValueError(f"duplicate partition identity {identity!r}")
            seen_identities.add(identity)
            total_records += 1
            score = _hash_score(identity, self.seed, "holdout")
            entry = (-score, identity)
            if len(selected) < required:
                heapq.heappush(selected, entry)
            elif entry > selected[0]:
                heapq.heapreplace(selected, entry)
        if total_records < required:
            raise ValueError(f"partition requires at least {required} records; got {total_records}")

        holdout_ids = [entry[1] for entry in selected]
        holdout_ids.sort(key=lambda identity: (_hash_score(identity, self.seed, "split"), identity))
        return PartitionSelection(
            validation_ids=frozenset(holdout_ids[: self.validation_size]),
            test_ids=frozenset(holdout_ids[self.validation_size :]),
            total_records=total_records,
        )

    def _select_caption_groups(self, records: Iterable[VideoRecord]) -> PartitionSelection:
        """Select exact holdouts without splitting identical normalized captions."""

        grouped: dict[str, list[VideoRecord]] = {}
        seen_identities: set[str] = set()
        total_records = 0
        for record in records:
            identity = _identity(record)
            if identity in seen_identities:
                raise ValueError(f"duplicate partition identity {identity!r}")
            seen_identities.add(identity)
            caption_key = _caption_key(record.caption)
            grouped.setdefault(caption_key, []).append(record)
            total_records += 1

        required = self.validation_size + self.test_size
        if total_records < required:
            raise ValueError(f"partition requires at least {required} records; got {total_records}")

        validation_groups = _choose_exact_groups(
            grouped,
            self.validation_size,
            seed=self.seed,
            salt="validation",
        )
        remaining = {
            key: records_for_caption
            for key, records_for_caption in grouped.items()
            if key not in validation_groups
        }
        test_groups = _choose_exact_groups(
            remaining,
            self.test_size,
            seed=self.seed,
            salt="test",
        )
        validation_ids = frozenset(
            _identity(record) for key in validation_groups for record in grouped[key]
        )
        test_ids = frozenset(
            _identity(record) for key in test_groups for record in grouped[key]
        )
        if len(validation_ids) != self.validation_size:
            raise ValueError(
                "caption-group validation selection cannot satisfy the requested exact size: "
                f"requested {self.validation_size}, got {len(validation_ids)}"
            )
        if len(test_ids) != self.test_size:
            raise ValueError(
                "caption-group test selection cannot satisfy the requested exact size: "
                f"requested {self.test_size}, got {len(test_ids)}"
            )
        return PartitionSelection(
            validation_ids=validation_ids,
            test_ids=test_ids,
            total_records=total_records,
        )

    def iter_split(
        self,
        records: Iterable[VideoRecord],
        selection: PartitionSelection,
        split: Literal["train", "validation", "test"],
    ) -> Iterator[VideoRecord]:
        """Classify a repeatable metadata stream using compact selected identities."""

        for record in records:
            identity = _identity(record)
            is_validation = split == "validation" and identity in selection.validation_ids
            is_test = split == "test" and identity in selection.test_ids
            is_train = (
                split == "train"
                and identity not in selection.validation_ids
                and identity not in selection.test_ids
            )
            if is_validation or is_test or is_train:
                yield record


def _identity(record: VideoRecord) -> str:
    revision = record.metadata.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"record {record.id!r} has no pinned revision")
    return "\x1f".join((revision, record.source, record.id))


def _hash_score(identity: str, seed: int, salt: str) -> int:
    digest = hashlib.blake2b(
        f"{identity}\x1f{seed}\x1f{salt}".encode(),
        digest_size=16,
        person=b"TARDIS-v1",
    ).digest()
    return int.from_bytes(digest, "big")


def _caption_key(caption: str) -> str:
    normalized = unicodedata.normalize("NFC", caption).strip()
    if not normalized:
        raise ValueError("caption-group partition does not allow an empty caption")
    return normalized


def _choose_exact_groups(
    grouped: dict[str, list[VideoRecord]],
    target_size: int,
    *,
    seed: int,
    salt: str,
) -> frozenset[str]:
    """Choose a deterministic exact-size subset of whole caption groups."""

    if target_size == 0:
        return frozenset()
    ordered = sorted(
        grouped,
        key=lambda caption: (_hash_score(caption, seed, f"caption-group:{salt}"), caption),
    )
    predecessor: list[tuple[int, int] | None] = [None] * (target_size + 1)
    predecessor[0] = (-1, -1)
    for group_index, caption in enumerate(ordered):
        group_size = len(grouped[caption])
        if group_size > target_size:
            continue
        for current_size in range(target_size - group_size, -1, -1):
            if predecessor[current_size] is None:
                continue
            next_size = current_size + group_size
            if predecessor[next_size] is None:
                predecessor[next_size] = (current_size, group_index)
    if predecessor[target_size] is None:
        available = sum(len(grouped[caption]) for caption in ordered)
        raise ValueError(
            "caption-group partition cannot satisfy exact holdout size "
            f"{target_size}; available records={available}, groups={len(ordered)}"
        )

    selected: set[str] = set()
    current_size = target_size
    while current_size:
        previous = predecessor[current_size]
        if previous is None or previous[1] < 0:
            raise RuntimeError("caption-group predecessor chain is corrupted")
        previous_size, group_index = previous
        selected.add(ordered[group_index])
        current_size = previous_size
    return frozenset(selected)
