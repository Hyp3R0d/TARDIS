"""Build balanced, byte-accounted local subsets for the three TARDIS sources."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tarfile
import time
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from tqdm import tqdm  # type: ignore[import-untyped]

from tardis.data.adapters import DataVerseAdapter, SeedanceAdapter
from tardis.data.archives import ArchiveMember, RemoteZipReader
from tardis.data.catalog import (
    DATAVERSE_REVISION,
    OPENVID_REVISION,
    SEEDANCE_REVISION,
)
from tardis.data.contracts import MetadataValue, VideoRecord
from tardis.data.curation import (
    CurationCandidate,
    CurationTarget,
    build_curation_report,
    select_candidates,
    write_curated_manifest,
)
from tardis.data.http_range import RangeClient
from tardis.data.splits import StablePartition

POLICY_VERSION = "tardis-balanced-45gb-v1"
DEFAULT_TARGET = CurationTarget(
    record_count=8_000,
    target_bytes=45_000_000_000,
    min_bytes=44_000_000_000,
    max_bytes=46_000_000_000,
    validation_size=256,
    test_size=512,
    split_seed=3407,
)
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/TARDIS/datasets")
DATAVERSE_REPOSITORY = "Vchitect/Vchitect_T2V_DataVerse"
OPENVID_REPOSITORY = "nkp37/OpenVid-1M"
SEEDANCE_DIRECTORY = "seedance-2-prompts-datasets"
DATAVERSE_DIRECTORY = "Vchitect_T2V_DataVerse"
OPENVID_DIRECTORY = "OpenVid-1M"
OPENVID_METADATA_RECORDS = 1_019_957


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    """A revision-pinned archive path and its exact repository byte count."""

    path: str
    expected_bytes: int

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.path)
        if normalized.is_absolute() or ".." in normalized.parts or str(normalized) == ".":
            raise ValueError(f"archive path must be a safe repository-relative path: {self.path}")
        if self.expected_bytes <= 0:
            raise ValueError("archive expected_bytes must be positive")


DATAVERSE_ARCHIVES = (
    ArchiveSpec("00000/000127.tar", 4_304_732_160),
    ArchiveSpec("00001/000397.tar", 6_506_752_000),
    ArchiveSpec("00002/000599.tar", 6_231_316_480),
    ArchiveSpec("00003/000720.tar", 5_568_133_120),
    ArchiveSpec("00004/000807.tar", 5_558_456_320),
    ArchiveSpec("00005/001038.tar", 5_625_569_280),
    ArchiveSpec("00006/001353.tar", 5_566_095_360),
    ArchiveSpec("00007/001496.tar", 5_639_680_000),
)

OPENVID_ARCHIVES = (
    ArchiveSpec("OpenVid_part100.zip", 19_173_859_423),
    ArchiveSpec("OpenVid_part84.zip", 19_799_110_192),
    ArchiveSpec("OpenVid_part108.zip", 19_958_035_287),
    ArchiveSpec("OpenVid_part114.zip", 20_138_925_501),
    ArchiveSpec("OpenVid_part85.zip", 20_216_549_049),
)


@dataclass(frozen=True, slots=True)
class _OpenVidMember:
    archive: ArchiveSpec
    archive_url: str
    member: ArchiveMember


def curate_seedance_dataset(
    root: Path,
    *,
    revision: str = SEEDANCE_REVISION,
    target: CurationTarget = DEFAULT_TARGET,
    policy_version: str = POLICY_VERSION,
    selection_seed: int = 3407,
    commit: bool,
) -> dict[str, Any]:
    """Select direct Seedance media and atomically remove only unreferenced videos."""

    root = root.expanduser().resolve()
    manifest = root / "tardis_manifest.jsonl"
    records = (
        _read_manifest(manifest, expected_source="seedance")
        if manifest.is_file()
        else _seedance_records_from_raw(root, revision=revision)
    )
    candidates: list[CurationCandidate] = []
    for record in tqdm(records, desc="Seedance 候选审计", unit="条"):
        media_path = _direct_media_path(record.media_locator)
        _require_below(media_path, root)
        if not media_path.is_file():
            raise FileNotFoundError(f"Seedance media does not exist: {media_path}")
        featured = record.metadata.get("is_featured")
        candidates.append(
            CurationCandidate(
                record=record,
                media_bytes=media_path.stat().st_size,
                quality_score=1.0 if featured is True else 0.0,
            )
        )
    selected = select_candidates(tuple(candidates), target=target, seed=selection_seed)
    report = build_curation_report(selected, target=target, policy_version=policy_version)
    report.update(
        {
            "dataset": "seedance",
            "candidate_count": len(candidates),
            "selection_seed": selection_seed,
        }
    )
    if not commit:
        return report

    selected_paths = {
        _direct_media_path(candidate.record.media_locator).resolve() for candidate in selected
    }
    write_curated_manifest(
        manifest,
        selected,
        target=target,
        policy_version=policy_version,
    )
    _write_json_atomic(root / "curation_report.json", report)
    for media_path in root.rglob("*"):
        if (
            media_path.is_file()
            and media_path.suffix.lower() in {".mp4", ".webm", ".mov"}
            and media_path.resolve() not in selected_paths
        ):
            media_path.unlink()
    _verify_direct_media(selected, root)
    return report


def curate_dataverse_dataset(
    root: Path,
    *,
    archive_specs: tuple[ArchiveSpec, ...] = DATAVERSE_ARCHIVES,
    revision: str = DATAVERSE_REVISION,
    target: CurationTarget = DEFAULT_TARGET,
    policy_version: str = POLICY_VERSION,
    commit: bool,
) -> dict[str, Any]:
    """Index selected DataVerse TARs, publish exactly their records, then remove old TARs."""

    root = root.expanduser().resolve()
    selected_paths = _validate_local_archives(root, archive_specs)
    member_sizes: dict[str, int] = {}
    for path in tqdm(selected_paths, desc="DataVerse TAR 索引", unit="包"):
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                basename = PurePosixPath(member.name).name
                if not member.isfile() or not basename.lower().endswith((".mp4", ".webm", ".mov")):
                    continue
                if basename in member_sizes:
                    raise ValueError(f"duplicate DataVerse member basename: {basename}")
                member_sizes[basename] = member.size

    shards = frozenset(int(Path(spec.path).stem) for spec in archive_specs)
    with RangeClient(max_object_bytes=64 * 1024 * 1024) as client:
        adapter = DataVerseAdapter(
            client,
            metadata_url=(root / "annotation.json").as_uri(),
            resolve_root=root.as_uri(),
            revision=revision,
            chunk_bytes=16 * 1024 * 1024,
            available_archive_shards=shards,
        )
        candidates = tuple(
            CurationCandidate(
                record=record,
                media_bytes=_required_member_size(member_sizes, record.id),
                source_archive=_archive_path_for_dataverse(record),
                source_member=f"./{record.id}",
            )
            for record in tqdm(
                adapter.iter_records(),
                total=target.record_count,
                desc="DataVerse 元数据连接",
                unit="条",
            )
        )
    selected = select_candidates(candidates, target=target, seed=target.split_seed)
    report = build_curation_report(selected, target=target, policy_version=policy_version)
    report.update(
        {
            "dataset": "dataverse",
            "candidate_count": len(candidates),
            "archives": [spec.path for spec in archive_specs],
            "archive_bytes": sum(spec.expected_bytes for spec in archive_specs),
        }
    )
    if not commit:
        return report

    write_curated_manifest(
        root / "tardis_manifest.jsonl",
        selected,
        target=target,
        policy_version=policy_version,
    )
    _write_json_atomic(root / "curation_report.json", report)
    selected_resolved = {path.resolve() for path in selected_paths}
    for path in root.rglob("*.tar"):
        if path.resolve() not in selected_resolved:
            path.unlink()
    _verify_tar_media(selected)
    return report


def curate_openvid_dataset(
    root: Path,
    *,
    archive_specs: tuple[ArchiveSpec, ...] = OPENVID_ARCHIVES,
    revision: str = OPENVID_REVISION,
    endpoint: str = "https://hf-mirror.com",
    repository: str = OPENVID_REPOSITORY,
    target: CurationTarget = DEFAULT_TARGET,
    policy_version: str = POLICY_VERSION,
    selection_seed: int = 3407,
    commit: bool,
    download_missing: bool = True,
) -> dict[str, Any]:
    """Select OpenVid members, repack selected bytes into TARs, and remove source ZIPs."""

    root = root.expanduser().resolve()
    members = _index_openvid_archives(
        root,
        archive_specs=archive_specs,
        endpoint=endpoint,
        repository=repository,
        revision=revision,
    )
    candidates = _join_openvid_metadata(root, members=members, revision=revision)
    selected = select_candidates(candidates, target=target, seed=selection_seed)
    initial_report = build_curation_report(
        selected,
        target=target,
        policy_version=policy_version,
    )
    archive_counts: dict[str, int] = defaultdict(int)
    for candidate in selected:
        if candidate.source_archive is None:
            raise ValueError("OpenVid candidate lacks source archive provenance")
        archive_counts[candidate.source_archive] += 1
    initial_report.update(
        {
            "dataset": "openvid",
            "candidate_count": len(candidates),
            "selection_seed": selection_seed,
            "source_archives": dict(sorted(archive_counts.items())),
        }
    )
    if not commit:
        return initial_report

    grouped: dict[str, list[CurationCandidate]] = defaultdict(list)
    for candidate in selected:
        assert candidate.source_archive is not None
        grouped[candidate.source_archive].append(candidate)
    materialized: list[CurationCandidate] = []
    staging_root = root / ".curation_sources"
    for spec in archive_specs:
        group = tuple(grouped.get(spec.path, ()))
        if not group:
            continue
        output = root / "curated" / f"{Path(spec.path).stem}.tar"
        if output.is_file():
            materialized.extend(_repack_openvid_archive(root / spec.path, output, group))
            continue
        source_path = root / spec.path
        temporary_source = False
        if not source_path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"OpenVid source archive is missing: {source_path}")
            source_path = staging_root / Path(spec.path).name
            _download_archive(
                source_path,
                url=_archive_url(endpoint, repository, revision, spec.path),
                expected_bytes=spec.expected_bytes,
            )
            temporary_source = True
        _validate_archive_size(source_path, spec.expected_bytes)
        materialized.extend(_repack_openvid_archive(source_path, output, group))
        if temporary_source:
            source_path.unlink(missing_ok=True)
    selected_local = tuple(sorted(materialized, key=lambda item: item.record.id))
    report = build_curation_report(
        selected_local,
        target=target,
        policy_version=policy_version,
    )
    report.update(initial_report)
    report["media_bytes"] = sum(item.media_bytes for item in selected_local)
    write_curated_manifest(
        root / "tardis_manifest.jsonl",
        selected_local,
        target=target,
        policy_version=policy_version,
    )
    _write_json_atomic(root / "curation_report.json", report)
    _verify_tar_media(selected_local)
    for spec in archive_specs:
        (root / spec.path).unlink(missing_ok=True)
        (staging_root / Path(spec.path).name).unlink(missing_ok=True)
    if staging_root.is_dir() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    return report


def download_dataverse_archives(
    root: Path,
    *,
    archive_specs: tuple[ArchiveSpec, ...] = DATAVERSE_ARCHIVES,
    endpoint: str = "https://hf-mirror.com",
    repository: str = DATAVERSE_REPOSITORY,
    revision: str = DATAVERSE_REVISION,
) -> None:
    """Download the pinned DataVerse metadata and selected full TARs with HF resume support."""

    root.mkdir(parents=True, exist_ok=True)
    files = ["annotation.json", ".gitattributes", "README.md"]
    files.extend(spec.path for spec in archive_specs)
    environment = dict(os.environ)
    environment["HF_ENDPOINT"] = endpoint
    environment.setdefault("HF_HUB_DISABLE_XET", "1")
    environment.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    environment.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    command = [
        "hf",
        "download",
        repository,
        *files,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(root),
        "--max-workers",
        "4",
    ]
    for attempt in range(1, 31):
        print(f"DataVerse 下载尝试 {attempt}/30", flush=True)
        result = subprocess.run(command, check=False, env=environment)
        if result.returncode == 0:
            break
        if attempt == 30:
            raise subprocess.CalledProcessError(result.returncode, command)
        delay = min(120, attempt * 10)
        print(f"镜像瞬时失败，{delay} 秒后从断点续传。", flush=True)
        time.sleep(delay)
    _validate_local_archives(root, archive_specs)


def verify_curated_dataset(
    root: Path,
    *,
    source: str,
    target: CurationTarget = DEFAULT_TARGET,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Verify manifest accounting, frozen split counts, and exact local media closure."""

    if source not in {"seedance", "dataverse", "openvid"}:
        raise ValueError(f"unsupported curated dataset source: {source}")
    root = root.expanduser().resolve()
    records = _read_manifest(root / "tardis_manifest.jsonl", expected_source=source)
    candidates: list[CurationCandidate] = []
    persisted_splits = {split: 0 for split in ("train", "validation", "test")}
    for record in records:
        metadata = record.metadata
        if metadata.get("curation_policy") != policy_version:
            raise ValueError(f"record {record.id!r} has the wrong curation policy")
        media_bytes = metadata.get("media_bytes")
        if not isinstance(media_bytes, int) or isinstance(media_bytes, bool):
            raise ValueError(f"record {record.id!r} has invalid media_bytes")
        quality = metadata.get("quality_score", 0.0)
        if not isinstance(quality, int | float) or isinstance(quality, bool):
            raise ValueError(f"record {record.id!r} has invalid quality_score")
        split = metadata.get("curation_split")
        if not isinstance(split, str) or split not in persisted_splits:
            raise ValueError(f"record {record.id!r} has invalid curation_split")
        persisted_splits[split] += 1
        source_archive = metadata.get("source_archive")
        source_member = metadata.get("source_member")
        candidates.append(
            CurationCandidate(
                record=record,
                media_bytes=media_bytes,
                quality_score=float(quality),
                source_archive=source_archive if isinstance(source_archive, str) else None,
                source_member=source_member if isinstance(source_member, str) else None,
            )
        )
    selected = tuple(candidates)
    report = build_curation_report(selected, target=target, policy_version=policy_version)
    if persisted_splits != report["splits"]:
        raise ValueError(
            f"persisted split counts {persisted_splits} do not match {report['splits']}"
        )
    if source == "seedance":
        runtime_splits = StablePartition(
            seed=target.split_seed,
            validation_size=target.validation_size,
            test_size=target.test_size,
            group_by_caption=True,
        ).partition(records)
        runtime_split_by_id = {
            record.id: split
            for split, split_records in (
                ("train", runtime_splits.train),
                ("validation", runtime_splits.validation),
                ("test", runtime_splits.test),
            )
            for record in split_records
        }
        mismatches = [
            record.id
            for record in records
            if record.metadata.get("curation_split") != runtime_split_by_id[record.id]
        ]
        if mismatches:
            raise ValueError(
                "persisted Seedance caption-group split does not match runtime partition; "
                f"mismatched records={mismatches[:10]}"
            )
        _verify_direct_media(selected, root)
    else:
        _verify_tar_media(selected)
        referenced_archives = {
            _local_tar_locator(candidate.record.media_locator)[0].resolve()
            for candidate in selected
        }
        actual_archives = {path.resolve() for path in root.rglob("*.tar") if path.is_file()}
        if actual_archives != referenced_archives:
            raise ValueError("dataset contains unreferenced or missing TAR archives")
        if source == "openvid" and any(root.rglob("*.zip")):
            raise ValueError("curated OpenVid dataset still contains source ZIP archives")
    stored_report = root / "curation_report.json"
    if not stored_report.is_file():
        raise FileNotFoundError(f"curation report does not exist: {stored_report}")
    stored = json.loads(stored_report.read_text(encoding="utf-8"))
    for key in ("record_count", "media_bytes", "splits", "split_seed", "policy_version"):
        if stored.get(key) != report[key]:
            raise ValueError(f"stored curation report field {key!r} does not match manifest")
    return report


def _index_openvid_archives(
    root: Path,
    *,
    archive_specs: tuple[ArchiveSpec, ...],
    endpoint: str,
    repository: str,
    revision: str,
) -> dict[str, _OpenVidMember]:
    catalog: dict[str, _OpenVidMember] = {}
    with RangeClient(
        max_object_bytes=64 * 1024 * 1024,
        timeout_seconds=120,
        max_retries=5,
    ) as client:
        for spec in tqdm(archive_specs, desc="OpenVid 远程 ZIP 索引", unit="包"):
            local = root / spec.path
            url = local.as_uri() if local.is_file() else _archive_url(
                endpoint, repository, revision, spec.path
            )
            remote = client.inspect(url)
            if remote.size != spec.expected_bytes:
                raise ValueError(
                    f"OpenVid archive {spec.path} has {remote.size} bytes; "
                    f"expected {spec.expected_bytes}"
                )
            reader = RemoteZipReader(client, url, max_member_bytes=512 * 1024 * 1024)
            for member in reader.iter_members():
                basename = PurePosixPath(member.name).name
                if not basename.lower().endswith((".mp4", ".webm", ".mov")):
                    continue
                if basename in catalog:
                    raise ValueError(f"duplicate OpenVid member basename: {basename}")
                catalog[basename] = _OpenVidMember(spec, url, member)
    return catalog


def _join_openvid_metadata(
    root: Path,
    *,
    members: Mapping[str, _OpenVidMember],
    revision: str,
) -> tuple[CurationCandidate, ...]:
    metadata_path = root / "data/train/OpenVid-1M.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OpenVid metadata does not exist: {metadata_path}")
    csv.field_size_limit(8 * 1024 * 1024)
    candidates: list[CurationCandidate] = []
    seen: set[str] = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in tqdm(
            reader,
            total=OPENVID_METADATA_RECORDS,
            desc="OpenVid 元数据连接",
            unit="条",
        ):
            video_name = PurePosixPath((row.get("video") or "").strip()).name
            member = members.get(video_name)
            if member is None:
                continue
            if video_name in seen:
                raise ValueError(f"duplicate OpenVid metadata row: {video_name}")
            seen.add(video_name)
            caption = (row.get("caption") or "").strip()
            if not caption:
                raise ValueError(f"OpenVid row has an empty caption: {video_name}")
            metadata: dict[str, MetadataValue] = {
                str(key): str(value)
                for key, value in row.items()
                if key not in {"video", "caption"} and value is not None
            }
            metadata["revision"] = revision
            record = VideoRecord(
                id=video_name,
                caption=caption,
                media_locator=(
                    f"zip+{member.archive_url}#member="
                    f"{quote(member.member.name, safe='/._-')}"
                ),
                source="openvid",
                metadata=metadata,
            )
            candidates.append(
                CurationCandidate(
                    record=record,
                    media_bytes=member.member.uncompressed_size,
                    quality_score=_openvid_quality(row),
                    source_archive=member.archive.path,
                    source_member=member.member.name,
                )
            )
    if len(candidates) != len(members):
        missing = sorted(set(members) - seen)
        raise ValueError(
            f"OpenVid metadata matched {len(candidates)}/{len(members)} indexed members; "
            f"first missing IDs: {missing[:5]}"
        )
    return tuple(candidates)


def _repack_openvid_archive(
    source: Path,
    destination: Path,
    selected: tuple[CurationCandidate, ...],
) -> tuple[CurationCandidate, ...]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = {candidate.record.id: candidate for candidate in selected}
    if destination.is_file():
        _validate_repacked_tar(destination, expected)
        return tuple(_localize_openvid_candidate(item, destination) for item in selected)

    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with zipfile.ZipFile(source) as input_archive:
        infos = {info.filename: info for info in input_archive.infolist() if not info.is_dir()}
        with partial.open("wb") as raw_output:
            with tarfile.open(fileobj=raw_output, mode="w", format=tarfile.USTAR_FORMAT) as output:
                for candidate in tqdm(
                    sorted(selected, key=lambda item: item.record.id),
                    desc=f"重打包 {source.name}",
                    unit="条",
                ):
                    if candidate.source_member is None:
                        raise ValueError("OpenVid candidate lacks source member provenance")
                    info = infos.get(candidate.source_member)
                    if info is None:
                        raise KeyError(
                            f"OpenVid member {candidate.source_member!r} not found in {source}"
                        )
                    if info.file_size != candidate.media_bytes:
                        raise ValueError(
                            f"OpenVid member {candidate.record.id} size changed: "
                            f"{info.file_size} != {candidate.media_bytes}"
                        )
                    tar_info = tarfile.TarInfo(candidate.record.id)
                    tar_info.size = info.file_size
                    tar_info.mode = 0o644
                    tar_info.mtime = 0
                    with input_archive.open(info) as payload:
                        output.addfile(tar_info, payload)
            raw_output.flush()
            os.fsync(raw_output.fileno())
    os.replace(partial, destination)
    _validate_repacked_tar(destination, expected)
    return tuple(_localize_openvid_candidate(item, destination) for item in selected)


def _localize_openvid_candidate(
    candidate: CurationCandidate, destination: Path
) -> CurationCandidate:
    record = VideoRecord(
        id=candidate.record.id,
        caption=candidate.record.caption,
        media_locator=(
            f"tar+{destination.resolve().as_uri()}#member="
            f"{quote(candidate.record.id, safe='._-')}"
        ),
        source=candidate.record.source,
        metadata=dict(candidate.record.metadata),
    )
    return CurationCandidate(
        record=record,
        media_bytes=candidate.media_bytes,
        quality_score=candidate.quality_score,
        source_archive=candidate.source_archive,
        source_member=candidate.source_member,
    )


def _validate_repacked_tar(
    path: Path, expected: Mapping[str, CurationCandidate]
) -> None:
    actual: dict[str, int] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if member.isfile():
                actual[member.name] = member.size
    expected_sizes = {record_id: item.media_bytes for record_id, item in expected.items()}
    if actual != expected_sizes:
        raise ValueError(f"repacked TAR {path} does not match its selected member plan")


def _download_archive(destination: Path, *, url: str, expected_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        _validate_archive_size(destination, expected_bytes)
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    command = [
        "aria2c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--check-integrity=true",
        "--connect-timeout=60",
        "--continue=true",
        "--file-allocation=none",
        "--max-connection-per-server=8",
        "--max-tries=30",
        "--min-split-size=16M",
        "--retry-wait=5",
        "--split=8",
        "--timeout=120",
        f"--dir={partial.parent}",
        f"--out={partial.name}",
        url,
    ]
    for attempt in range(1, 31):
        print(f"{destination.name} 下载尝试 {attempt}/30", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            break
        if attempt == 30:
            raise subprocess.CalledProcessError(result.returncode, command)
        delay = min(120, attempt * 10)
        print(f"镜像瞬时失败，{delay} 秒后从断点续传。", flush=True)
        time.sleep(delay)
    _validate_archive_size(partial, expected_bytes)
    os.replace(partial, destination)


def _validate_local_archives(
    root: Path, archive_specs: tuple[ArchiveSpec, ...]
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for spec in archive_specs:
        path = root / spec.path
        if not path.is_file():
            raise FileNotFoundError(f"selected archive does not exist: {path}")
        _validate_archive_size(path, spec.expected_bytes)
        paths.append(path)
    return tuple(paths)


def _validate_archive_size(path: Path, expected_bytes: int) -> None:
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(f"archive {path} has {actual} bytes; expected {expected_bytes}")


def _verify_direct_media(selected: tuple[CurationCandidate, ...], root: Path) -> None:
    selected_paths = set()
    for candidate in selected:
        path = _direct_media_path(candidate.record.media_locator).resolve()
        _require_below(path, root)
        if path.stat().st_size != candidate.media_bytes:
            raise ValueError(f"direct media size changed after curation: {path}")
        selected_paths.add(path)
    actual_paths = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov"}
    }
    if actual_paths != selected_paths:
        raise ValueError("Seedance directory contains unreferenced or missing media")


def _verify_tar_media(selected: tuple[CurationCandidate, ...]) -> None:
    grouped: dict[Path, dict[str, int]] = defaultdict(dict)
    for candidate in selected:
        archive, member_name = _local_tar_locator(candidate.record.media_locator)
        grouped[archive][member_name] = candidate.media_bytes
    for archive, expected in grouped.items():
        actual: dict[str, int] = {}
        with tarfile.open(archive, mode="r:") as stream:
            for tar_member in stream:
                if tar_member.isfile() and tar_member.name.lower().endswith(
                    (".mp4", ".webm", ".mov")
                ):
                    actual[tar_member.name] = tar_member.size
        if actual != expected:
            raise ValueError(f"TAR media verification failed: {archive}")


def _read_manifest(path: Path, *, expected_source: str) -> tuple[VideoRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {path}")
    records: list[VideoRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if value.get("source") != expected_source:
                raise ValueError(f"manifest line {line_number} has the wrong source")
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"manifest line {line_number} metadata is not an object")
            records.append(
                VideoRecord(
                    id=str(value["id"]),
                    caption=str(value["caption"]),
                    media_locator=str(value["media_locator"]),
                    source=expected_source,
                    metadata={str(key): _metadata_value(item) for key, item in metadata.items()},
                )
            )
    return tuple(records)


def _seedance_records_from_raw(root: Path, *, revision: str) -> tuple[VideoRecord, ...]:
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Seedance metadata does not exist: {metadata_path}")
    media_paths = frozenset(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov"}
    )
    if not media_paths:
        raise FileNotFoundError(f"Seedance media does not exist below {root}")
    with RangeClient(max_object_bytes=64 * 1024 * 1024) as client:
        adapter = SeedanceAdapter(
            client,
            metadata_url=metadata_path.as_uri(),
            resolve_root=root.as_uri(),
            revision=revision,
            chunk_bytes=16 * 1024 * 1024,
            available_media_paths=media_paths,
        )
        return tuple(adapter.iter_records())


def _direct_media_path(locator: str) -> Path:
    parsed = urlsplit(locator)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"expected a local direct file URI: {locator}")
    return Path(unquote(parsed.path))


def _local_tar_locator(locator: str) -> tuple[Path, str]:
    if not locator.startswith("tar+") or "#member=" not in locator:
        raise ValueError(f"expected a local TAR locator: {locator}")
    archive_uri, member = locator.removeprefix("tar+").split("#member=", maxsplit=1)
    return _direct_media_path(archive_uri), unquote(member)


def _require_below(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"media path is outside its dataset root: {path}") from error


def _required_member_size(member_sizes: Mapping[str, int], record_id: str) -> int:
    try:
        return member_sizes[record_id]
    except KeyError as error:
        raise KeyError(
            f"DataVerse metadata record has no selected TAR member: {record_id}"
        ) from error


def _archive_path_for_dataverse(record: VideoRecord) -> str:
    shard = record.metadata.get("archive_shard")
    if not isinstance(shard, int):
        raise ValueError(f"DataVerse record {record.id} has no integer archive_shard")
    return f"{shard // 200:05d}/{shard:06d}.tar"


def _openvid_quality(row: Mapping[str, str]) -> float:
    aesthetic = _bounded_float(row.get("aesthetic score"), lower=0.0, upper=10.0) / 10.0
    temporal = _bounded_float(
        row.get("temporal consistency score"), lower=0.0, upper=1.0
    )
    motion = max(0.0, _bounded_float(row.get("motion score"), lower=0.0, upper=1_000.0))
    motion_preference = 1.0 / (1.0 + abs(motion - 8.0) / 8.0)
    return 0.55 * aesthetic + 0.35 * temporal + 0.10 * motion_preference


def _bounded_float(value: str | None, *, lower: float, upper: float) -> float:
    try:
        parsed = float(value) if value is not None else lower
    except ValueError:
        parsed = lower
    return min(upper, max(lower, parsed))


def _archive_url(endpoint: str, repository: str, revision: str, path: str) -> str:
    return (
        f"{endpoint.rstrip('/')}/datasets/{repository}/resolve/{revision}/"
        f"{quote(path, safe='/._-')}"
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_value(value: object) -> MetadataValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 TARDIS 三源 8,000 条 / 45 GB 平衡数据集")
    parser.add_argument("source", choices=("seedance", "dataverse", "openvid", "all"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = ("seedance", "dataverse", "openvid") if args.source == "all" else (args.source,)
    for source in sources:
        directory = {
            "seedance": SEEDANCE_DIRECTORY,
            "dataverse": DATAVERSE_DIRECTORY,
            "openvid": OPENVID_DIRECTORY,
        }[source]
        if args.verify_only:
            report = verify_curated_dataset(args.data_root / directory, source=source)
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            continue
        if source == "seedance":
            report = curate_seedance_dataset(
                args.data_root / SEEDANCE_DIRECTORY,
                commit=not args.dry_run,
            )
        elif source == "dataverse":
            root = args.data_root / DATAVERSE_DIRECTORY
            if args.download_missing:
                download_dataverse_archives(root)
            report = curate_dataverse_dataset(root, commit=not args.dry_run)
        else:
            report = curate_openvid_dataset(
                args.data_root / OPENVID_DIRECTORY,
                commit=not args.dry_run,
                download_missing=args.download_missing,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
