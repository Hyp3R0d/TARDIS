from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from tardis.data.curate_local import (
    ArchiveSpec,
    curate_dataverse_dataset,
    curate_openvid_dataset,
    curate_seedance_dataset,
    verify_curated_dataset,
)
from tardis.data.curation import CurationTarget


def _target(count: int, media_bytes: int) -> CurationTarget:
    return CurationTarget(
        record_count=count,
        target_bytes=media_bytes,
        min_bytes=media_bytes,
        max_bytes=media_bytes,
        validation_size=1,
        test_size=1,
        split_seed=17,
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_curate_seedance_publishes_manifest_before_removing_unselected_media(
    tmp_path: Path,
) -> None:
    root = tmp_path / "seedance-2-prompts-datasets"
    videos = root / "seedance-2/videos"
    videos.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index in range(6):
        media = videos / f"SD2_{index:05d}.mp4"
        media.write_bytes(bytes([index]) * 10)
        rows.append(
            {
                "id": f"SD2_{index:05d}",
                "caption": f"prompt {index}",
                "media_locator": media.as_uri(),
                "source": "seedance",
                "metadata": {"revision": "s" * 40, "category": "test"},
            }
        )
    (videos / "orphan.webm").write_bytes(b"not referenced by the source manifest")
    _write_manifest(root / "tardis_manifest.jsonl", rows)

    report = curate_seedance_dataset(
        root,
        target=_target(4, 40),
        policy_version="test-v1",
        selection_seed=9,
        commit=True,
    )

    assert report["record_count"] == 4
    assert len(list(videos.glob("*.mp4"))) == 4
    assert len((root / "tardis_manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads((root / "curation_report.json").read_text(encoding="utf-8")) == report
    verified = verify_curated_dataset(
        root,
        source="seedance",
        target=_target(4, 40),
        policy_version="test-v1",
    )
    assert verified["record_count"] == 4


def test_verify_seedance_rejects_persisted_split_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "seedance-2-prompts-datasets"
    videos = root / "seedance-2/videos"
    videos.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index in range(6):
        media = videos / f"SD2_{index:05d}.mp4"
        media.write_bytes(bytes([index]) * 10)
        rows.append(
            {
                "id": f"SD2_{index:05d}",
                "caption": f"prompt {index}",
                "media_locator": media.as_uri(),
                "source": "seedance",
                "metadata": {"revision": "s" * 40, "category": "test"},
            }
        )
    _write_manifest(root / "tardis_manifest.jsonl", rows)
    target = _target(4, 40)
    curate_seedance_dataset(root, target=target, policy_version="test-v1", commit=True)

    persisted = [
        json.loads(line)
        for line in (root / "tardis_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_row = next(row for row in persisted if row["metadata"]["curation_split"] == "train")
    validation_row = next(
        row for row in persisted if row["metadata"]["curation_split"] == "validation"
    )
    train_row["metadata"]["curation_split"], validation_row["metadata"]["curation_split"] = (
        validation_row["metadata"]["curation_split"],
        train_row["metadata"]["curation_split"],
    )
    _write_manifest(root / "tardis_manifest.jsonl", persisted)

    with pytest.raises(ValueError, match="does not match runtime partition"):
        verify_curated_dataset(root, source="seedance", target=target, policy_version="test-v1")


def test_curate_seedance_can_bootstrap_manifest_from_raw_metadata(tmp_path: Path) -> None:
    root = tmp_path / "seedance-2-prompts-datasets"
    videos = root / "seedance-2/videos"
    videos.mkdir(parents=True)
    metadata_rows = []
    for index in range(4):
        relative = f"seedance-2/videos/SD2_{index:05d}.mp4"
        (root / relative).write_bytes(bytes([index]) * 10)
        metadata_rows.append(
            {
                "id": f"SD2_{index:05d}",
                "raw_p": f"prompt {index}",
                "file_name": relative,
                "category": "test",
            }
        )
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )

    report = curate_seedance_dataset(
        root,
        revision="r" * 40,
        target=_target(2, 20),
        policy_version="test-v1",
        selection_seed=9,
        commit=True,
    )

    assert report["candidate_count"] == 4
    assert (root / "tardis_manifest.jsonl").is_file()
    assert len(list(videos.glob("*.mp4"))) == 2


def test_curate_dataverse_indexes_selected_tar_members_and_removes_old_tar(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Vchitect_T2V_DataVerse"
    root.mkdir()
    selected_tar = root / "00000/000000.tar"
    old_tar = root / "00000/000001.tar"
    _write_tar(
        selected_tar,
        {"./0000000000.mp4": b"a" * 11, "./0000000001.mp4": b"b" * 13},
    )
    _write_tar(old_tar, {"./0000001000.mp4": b"old"})
    (root / "annotation.json").write_text(
        json.dumps(
            [
                {"video": "0000000000.mp4", "text": "first"},
                {"video": "0000000001.mp4", "text": "second"},
                {"video": "0000001000.mp4", "text": "old"},
            ]
        ),
        encoding="utf-8",
    )

    report = curate_dataverse_dataset(
        root,
        archive_specs=(
            ArchiveSpec("00000/000000.tar", selected_tar.stat().st_size),
        ),
        revision="d" * 40,
        target=_target(2, 24),
        policy_version="test-v1",
        commit=True,
    )

    assert report["media_bytes"] == 24
    assert selected_tar.is_file()
    assert not old_tar.exists()
    rows = [
        json.loads(line)
        for line in (root / "tardis_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["metadata"]["media_bytes"] for row in rows} == {11, 13}
    assert verify_curated_dataset(
        root,
        source="dataverse",
        target=_target(2, 24),
        policy_version="test-v1",
    )["media_bytes"] == 24


def test_curate_openvid_repacks_selected_zip_members_and_removes_source_zips(
    tmp_path: Path,
) -> None:
    root = tmp_path / "OpenVid-1M"
    metadata = root / "data/train/OpenVid-1M.csv"
    metadata.parent.mkdir(parents=True)
    fieldnames = [
        "video",
        "caption",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "camera motion",
        "frame",
        "fps",
        "seconds",
    ]
    archives: list[ArchiveSpec] = []
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for archive_index in range(2):
            archive_path = root / f"OpenVid_part{archive_index}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for member_index in range(3):
                    name = f"clip-{archive_index}-{member_index}.mp4"
                    archive.writestr(f"videos/{name}", bytes([member_index]) * 10)
                    writer.writerow(
                        {
                            "video": name,
                            "caption": f"caption {name}",
                            "aesthetic score": "6.0",
                            "motion score": "5.0",
                            "temporal consistency score": "0.99",
                            "camera motion": "static",
                            "frame": "30",
                            "fps": "30",
                            "seconds": "1",
                        }
                    )
            archives.append(ArchiveSpec(archive_path.name, archive_path.stat().st_size))

    report = curate_openvid_dataset(
        root,
        archive_specs=tuple(archives),
        revision="o" * 40,
        endpoint="https://unused.invalid",
        repository="unused/repository",
        target=_target(4, 40),
        policy_version="test-v1",
        selection_seed=5,
        commit=True,
        download_missing=False,
    )

    assert report["record_count"] == 4
    assert not list(root.glob("*.zip"))
    assert sum(1 for _ in (root / "curated").glob("*.tar")) == 2
    rows = [
        json.loads(line)
        for line in (root / "tardis_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert all(row["media_locator"].startswith("tar+file://") for row in rows)
    assert verify_curated_dataset(
        root,
        source="openvid",
        target=_target(4, 40),
        policy_version="test-v1",
    )["record_count"] == 4
