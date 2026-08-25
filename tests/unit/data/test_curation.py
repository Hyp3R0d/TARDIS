from __future__ import annotations

import json
from pathlib import Path

import pytest

from tardis.data.contracts import VideoRecord
from tardis.data.curation import (
    CurationCandidate,
    CurationTarget,
    build_curation_report,
    select_candidates,
    write_curated_manifest,
)


def _candidate(index: int, media_bytes: int, quality: float = 0.0) -> CurationCandidate:
    return CurationCandidate(
        record=VideoRecord(
            id=f"clip-{index:03d}.mp4",
            caption=f"caption {index}",
            media_locator=f"file:///videos/clip-{index:03d}.mp4",
            source="openvid",
            metadata={"revision": "a" * 40},
        ),
        media_bytes=media_bytes,
        quality_score=quality,
    )


def test_select_candidates_is_deterministic_and_meets_cardinality_and_byte_budget() -> None:
    candidates = tuple(
        _candidate(index, media_bytes=1_000 + index * 37, quality=(index % 7) / 7)
        for index in range(30)
    )
    target = CurationTarget(
        record_count=20,
        target_bytes=30_000,
        min_bytes=28_000,
        max_bytes=32_000,
        validation_size=3,
        test_size=5,
        split_seed=3407,
    )

    first = select_candidates(candidates, target=target, seed=19)
    second = select_candidates(tuple(reversed(candidates)), target=target, seed=19)

    assert [item.record.id for item in first] == [item.record.id for item in second]
    assert len(first) == 20
    assert target.min_bytes <= sum(item.media_bytes for item in first) <= target.max_bytes


def test_select_candidates_rejects_duplicate_record_identity() -> None:
    duplicate = _candidate(1, 1_000)
    target = CurationTarget(
        record_count=1,
        target_bytes=1_000,
        min_bytes=900,
        max_bytes=1_100,
        validation_size=0,
        test_size=0,
    )

    with pytest.raises(ValueError, match="duplicate candidate identity"):
        select_candidates((duplicate, duplicate), target=target, seed=1)


def test_build_curation_report_uses_exact_stable_split_counts() -> None:
    selected = tuple(_candidate(index, 1_000) for index in range(20))
    target = CurationTarget(
        record_count=20,
        target_bytes=20_000,
        min_bytes=19_000,
        max_bytes=21_000,
        validation_size=3,
        test_size=5,
        split_seed=3407,
    )

    report = build_curation_report(selected, target=target, policy_version="test-v1")

    assert report["record_count"] == 20
    assert report["media_bytes"] == 20_000
    assert report["splits"] == {"train": 12, "validation": 3, "test": 5}
    assert report["policy_version"] == "test-v1"


def test_write_curated_manifest_persists_media_bytes_and_split(tmp_path: Path) -> None:
    selected = tuple(_candidate(index, 1_000 + index) for index in range(8))
    target = CurationTarget(
        record_count=8,
        target_bytes=sum(item.media_bytes for item in selected),
        min_bytes=8_000,
        max_bytes=9_000,
        validation_size=2,
        test_size=2,
        split_seed=11,
    )
    destination = tmp_path / "tardis_manifest.jsonl"

    write_curated_manifest(
        destination,
        selected,
        target=target,
        policy_version="test-v1",
    )

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert {row["metadata"]["curation_split"] for row in rows} == {
        "train",
        "validation",
        "test",
    }
    assert all(row["metadata"]["media_bytes"] >= 1_000 for row in rows)
    assert all(row["metadata"]["curation_policy"] == "test-v1" for row in rows)


def test_seedance_manifest_keeps_duplicate_captions_in_one_split(tmp_path: Path) -> None:
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
        "group-g",
        "group-h",
    )
    selected = tuple(
        CurationCandidate(
            record=VideoRecord(
                id=f"seedance-{index:02d}.mp4",
                caption=caption,
                media_locator=f"file:///videos/seedance-{index:02d}.mp4",
                source="seedance",
                metadata={"revision": "a" * 40},
            ),
            media_bytes=1_000,
        )
        for index, caption in enumerate(captions)
    )
    target = CurationTarget(
        record_count=12,
        target_bytes=12_000,
        min_bytes=11_000,
        max_bytes=13_000,
        validation_size=3,
        test_size=4,
        split_seed=3407,
    )
    destination = tmp_path / "tardis_manifest.jsonl"

    write_curated_manifest(destination, selected, target=target, policy_version="test-v1")

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    split_by_caption: dict[str, set[str]] = {}
    for row in rows:
        split_by_caption.setdefault(row["caption"], set()).add(
            row["metadata"]["curation_split"]
        )
    assert all(len(splits) == 1 for splits in split_by_caption.values())
    counts = {split: 0 for split in ("train", "validation", "test")}
    for row in rows:
        counts[row["metadata"]["curation_split"]] += 1
    assert counts == {"train": 5, "validation": 3, "test": 4}
