"""Deterministic, byte-accounted curation primitives for local video datasets."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tardis.data.contracts import VideoRecord
from tardis.data.splits import PartitionSelection, StablePartition


@dataclass(frozen=True, slots=True)
class CurationTarget:
    """One source's exact cardinality, byte budget, and frozen holdout sizes."""

    record_count: int
    target_bytes: int
    min_bytes: int
    max_bytes: int
    validation_size: int
    test_size: int
    split_seed: int = 3407

    def __post_init__(self) -> None:
        if self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if self.min_bytes <= 0:
            raise ValueError("min_bytes must be positive")
        if not self.min_bytes <= self.target_bytes <= self.max_bytes:
            raise ValueError("target_bytes must lie inside the byte budget")
        if self.validation_size < 0 or self.test_size < 0:
            raise ValueError("split sizes cannot be negative")
        if self.validation_size + self.test_size > self.record_count:
            raise ValueError("validation and test sizes exceed record_count")


@dataclass(frozen=True, slots=True)
class CurationCandidate:
    """A canonical record augmented with materialized-media accounting."""

    record: VideoRecord
    media_bytes: int
    quality_score: float = 0.0
    content_sha256: str | None = None
    source_archive: str | None = None
    source_member: str | None = None

    def __post_init__(self) -> None:
        if self.media_bytes <= 0:
            raise ValueError("media_bytes must be positive")
        if not math.isfinite(self.quality_score):
            raise ValueError("quality_score must be finite")
        if self.content_sha256 is not None and not _is_sha256(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")


def select_candidates(
    candidates: tuple[CurationCandidate, ...],
    *,
    target: CurationTarget,
    seed: int,
) -> tuple[CurationCandidate, ...]:
    """Select a fixed count near the target bytes with a quality-size Lagrangian."""

    ordered = tuple(sorted(candidates, key=_candidate_identity))
    identities = [_candidate_identity(candidate) for candidate in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate candidate identity")
    if len(ordered) < target.record_count:
        raise ValueError(
            f"curation requires {target.record_count} candidates; got {len(ordered)}"
        )

    smallest = sum(
        candidate.media_bytes
        for candidate in sorted(ordered, key=_size_order)[: target.record_count]
    )
    largest = sum(
        candidate.media_bytes
        for candidate in sorted(ordered, key=_size_order, reverse=True)[: target.record_count]
    )
    if smallest > target.max_bytes or largest < target.min_bytes:
        raise ValueError(
            "candidate pool cannot satisfy byte budget: "
            f"reachable range is {smallest}-{largest}, target is "
            f"{target.min_bytes}-{target.max_bytes}"
        )

    average_bytes = target.target_bytes / target.record_count
    evaluations: dict[tuple[str, ...], tuple[CurationCandidate, ...]] = {}

    def evaluate(penalty: float) -> tuple[CurationCandidate, ...]:
        ranked = sorted(
            ordered,
            key=lambda candidate: (
                -(
                    candidate.quality_score
                    + _stable_unit(_candidate_identity(candidate), seed) * 1e-9
                    - penalty * candidate.media_bytes / average_bytes
                ),
                _candidate_identity(candidate),
            ),
        )[: target.record_count]
        selected = tuple(sorted(ranked, key=_candidate_identity))
        evaluations[tuple(_candidate_identity(item) for item in selected)] = selected
        return selected

    neutral = evaluate(0.0)
    neutral_bytes = _total_bytes(neutral)
    if target.min_bytes <= neutral_bytes <= target.max_bytes:
        return neutral

    direction = 1.0 if neutral_bytes > target.target_bytes else -1.0
    low = 0.0
    high = direction
    for _ in range(64):
        current = evaluate(high)
        current_bytes = _total_bytes(current)
        crossed = (
            current_bytes <= target.target_bytes
            if direction > 0
            else current_bytes >= target.target_bytes
        )
        if crossed:
            break
        low = high
        high *= 2.0
    else:
        raise ValueError("failed to bracket the curation byte target")

    lower = min(low, high)
    upper = max(low, high)
    for _ in range(72):
        midpoint = (lower + upper) / 2.0
        current = evaluate(midpoint)
        current_bytes = _total_bytes(current)
        if current_bytes > target.target_bytes:
            lower = midpoint
        else:
            upper = midpoint

    feasible = [
        selected
        for selected in evaluations.values()
        if target.min_bytes <= _total_bytes(selected) <= target.max_bytes
    ]
    if feasible:
        return min(feasible, key=lambda selected: _selection_order(selected, target))

    closest = min(evaluations.values(), key=lambda selected: _selection_order(selected, target))
    repaired = _repair_byte_budget(ordered, closest, target)
    if not target.min_bytes <= _total_bytes(repaired) <= target.max_bytes:
        raise ValueError("candidate selection did not converge inside the byte budget")
    return repaired


def build_curation_report(
    selected: tuple[CurationCandidate, ...],
    *,
    target: CurationTarget,
    policy_version: str,
) -> dict[str, Any]:
    """Build a JSON-serializable accounting report and validate frozen split sizes."""

    policy_version = policy_version.strip()
    if not policy_version:
        raise ValueError("policy_version cannot be empty")
    _validate_selection(selected, target)
    split_by_identity = _split_assignments(selected, target)
    split_counts = {split: 0 for split in ("train", "validation", "test")}
    split_bytes = {split: 0 for split in ("train", "validation", "test")}
    for candidate in selected:
        split = split_by_identity[_candidate_identity(candidate)]
        split_counts[split] += 1
        split_bytes[split] += candidate.media_bytes
    expected_train = target.record_count - target.validation_size - target.test_size
    expected = {
        "train": expected_train,
        "validation": target.validation_size,
        "test": target.test_size,
    }
    if split_counts != expected:
        raise ValueError(f"stable partition mismatch: expected {expected}, got {split_counts}")
    sources = sorted({candidate.record.source for candidate in selected})
    return {
        "policy_version": policy_version,
        "sources": sources,
        "record_count": len(selected),
        "media_bytes": _total_bytes(selected),
        "target_bytes": target.target_bytes,
        "min_bytes": target.min_bytes,
        "max_bytes": target.max_bytes,
        "split_seed": target.split_seed,
        "splits": split_counts,
        "split_media_bytes": split_bytes,
    }


def write_curated_manifest(
    destination: Path,
    selected: tuple[CurationCandidate, ...],
    *,
    target: CurationTarget,
    policy_version: str,
) -> None:
    """Atomically publish a curated manifest with byte and split provenance."""

    _validate_selection(selected, target)
    split_by_identity = _split_assignments(selected, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for candidate in sorted(selected, key=_candidate_identity):
                metadata = dict(candidate.record.metadata)
                metadata.update(
                    {
                        "curation_policy": policy_version,
                        "curation_split": split_by_identity[_candidate_identity(candidate)],
                        "media_bytes": candidate.media_bytes,
                        "quality_score": candidate.quality_score,
                    }
                )
                if candidate.content_sha256 is not None:
                    metadata["content_sha256"] = candidate.content_sha256
                if candidate.source_archive is not None:
                    metadata["source_archive"] = candidate.source_archive
                if candidate.source_member is not None:
                    metadata["source_member"] = candidate.source_member
                payload = {
                    "id": candidate.record.id,
                    "caption": candidate.record.caption,
                    "media_locator": candidate.record.media_locator,
                    "source": candidate.record.source,
                    "metadata": metadata,
                }
                stream.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_selection(
    selected: tuple[CurationCandidate, ...], target: CurationTarget
) -> None:
    if len(selected) != target.record_count:
        raise ValueError(
            f"selected record count must be {target.record_count}; got {len(selected)}"
        )
    identities = [_candidate_identity(candidate) for candidate in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("selected records contain duplicate identities")
    media_bytes = _total_bytes(selected)
    if not target.min_bytes <= media_bytes <= target.max_bytes:
        raise ValueError(
            f"selected media bytes {media_bytes} lie outside "
            f"{target.min_bytes}-{target.max_bytes}"
        )


def _split_assignments(
    selected: tuple[CurationCandidate, ...], target: CurationTarget
) -> dict[str, str]:
    partition = StablePartition(
        seed=target.split_seed,
        validation_size=target.validation_size,
        test_size=target.test_size,
        group_by_caption=bool(selected and selected[0].record.source == "seedance"),
    )
    records = tuple(candidate.record for candidate in selected)
    selection = partition.select(records)
    return {
        _candidate_identity(candidate): _record_split(candidate.record, selection)
        for candidate in selected
    }


def _record_split(record: VideoRecord, selection: PartitionSelection) -> str:
    identity = _record_identity(record)
    if identity in selection.validation_ids:
        return "validation"
    if identity in selection.test_ids:
        return "test"
    return "train"


def _repair_byte_budget(
    candidates: tuple[CurationCandidate, ...],
    selected: tuple[CurationCandidate, ...],
    target: CurationTarget,
) -> tuple[CurationCandidate, ...]:
    selected_by_id = {_candidate_identity(item): item for item in selected}
    total = _total_bytes(selected)
    for _ in range(len(candidates)):
        if target.min_bytes <= total <= target.max_bytes:
            return tuple(sorted(selected_by_id.values(), key=_candidate_identity))
        unselected = [
            item for item in candidates if _candidate_identity(item) not in selected_by_id
        ]
        current = sorted(selected_by_id.values(), key=_size_order)
        current_sizes = [item.media_bytes for item in current]
        best: tuple[tuple[float, float, str, str], CurationCandidate, CurationCandidate] | None = (
            None
        )
        increasing = total < target.min_bytes
        for addition in unselected:
            desired_delta = target.target_bytes - total
            desired_removal = addition.media_bytes - desired_delta
            index = bisect.bisect_left(current_sizes, desired_removal)
            for candidate_index in {index - 1, index, index + 1}:
                if not 0 <= candidate_index < len(current):
                    continue
                removal = current[candidate_index]
                delta = addition.media_bytes - removal.media_bytes
                if increasing != (delta > 0) or delta == 0:
                    continue
                updated_total = total + delta
                if abs(updated_total - target.target_bytes) >= abs(
                    total - target.target_bytes
                ):
                    continue
                quality_delta = addition.quality_score - removal.quality_score
                order = (
                    abs(updated_total - target.target_bytes),
                    -quality_delta,
                    _candidate_identity(addition),
                    _candidate_identity(removal),
                )
                if best is None or order < best[0]:
                    best = (order, addition, removal)
        if best is None:
            break
        _, addition, removal = best
        del selected_by_id[_candidate_identity(removal)]
        selected_by_id[_candidate_identity(addition)] = addition
        total += addition.media_bytes - removal.media_bytes
    return tuple(sorted(selected_by_id.values(), key=_candidate_identity))


def _selection_order(
    selected: tuple[CurationCandidate, ...], target: CurationTarget
) -> tuple[float, float, tuple[str, ...]]:
    return (
        abs(_total_bytes(selected) - target.target_bytes),
        -sum(candidate.quality_score for candidate in selected),
        tuple(_candidate_identity(candidate) for candidate in selected),
    )


def _candidate_identity(candidate: CurationCandidate) -> str:
    return _record_identity(candidate.record)


def _record_identity(record: VideoRecord) -> str:
    revision = record.metadata.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"record {record.id!r} has no pinned revision")
    return "\x1f".join((revision, record.source, record.id))


def _size_order(candidate: CurationCandidate) -> tuple[int, str]:
    return candidate.media_bytes, _candidate_identity(candidate)


def _total_bytes(candidates: tuple[CurationCandidate, ...]) -> int:
    return sum(candidate.media_bytes for candidate in candidates)


def _stable_unit(identity: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{identity}\x1f{seed}".encode(),
        digest_size=8,
        person=b"TARDIS-curate",
    ).digest()
    return int.from_bytes(digest, "big") / ((1 << 64) - 1)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
