"""Revision-pinned remote and local source adapter construction.

The catalog contains metadata URLs and revisions only. Video bytes are never
materialized while discovering a source or selecting a split.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import httpx

from tardis.data.adapters import (
    DatasetAdapter,
    DataVerseAdapter,
    LocalManifestAdapter,
    OpenVidAdapter,
    SeedanceAdapter,
)
from tardis.data.http_range import RangeClient

_DEFAULT_METADATA_CHUNK_BYTES = 16 * 1024 * 1024
LOCAL_MANIFEST_NAME = "tardis_manifest.jsonl"


class SourceName(StrEnum):
    DATAVERSE = "dataverse"
    OPENVID = "openvid"
    SEEDANCE = "seedance"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A revision-pinned metadata layout for one mirror dataset."""

    source: str
    repository: str
    revision: str
    metadata_path: str
    openvid_tree_path: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.repository.strip():
            raise ValueError("source and repository must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("dataset revision must be a 40-character commit SHA")
        if not self.metadata_path.strip():
            raise ValueError("metadata_path must be non-empty")


DATAVERSE_REVISION = "e068be25f4d06a837992a1e9096fd00105c83f2c"
OPENVID_REVISION = "d8a63bd22989c80b5734ec2bb989f4e1b61a5807"
SEEDANCE_REVISION = "515aa5bd59123fb489914ce9cd21419badb08be4"

LOCAL_DATASET_DIRECTORIES: dict[str, str] = {
    SourceName.DATAVERSE.value: "Vchitect_T2V_DataVerse",
    SourceName.OPENVID.value: "OpenVid-1M",
    SourceName.SEEDANCE.value: "seedance-2-prompts-datasets",
}

DEFAULT_SOURCE_SPECS: dict[str, SourceSpec] = {
    SourceName.DATAVERSE.value: SourceSpec(
        SourceName.DATAVERSE.value,
        "Vchitect/Vchitect_T2V_DataVerse",
        DATAVERSE_REVISION,
        "annotation.json",
    ),
    SourceName.OPENVID.value: SourceSpec(
        SourceName.OPENVID.value,
        "nkp37/OpenVid-1M",
        OPENVID_REVISION,
        "data/train/OpenVidHD.csv",
        openvid_tree_path="OpenVidHD",
    ),
    SourceName.SEEDANCE.value: SourceSpec(
        SourceName.SEEDANCE.value,
        "GokuScraper/seedance-2-prompts-datasets",
        SEEDANCE_REVISION,
        "metadata.jsonl",
    ),
}

_OPENVID_PART = re.compile(r"(?:^|/)OpenVidHD_part_(\d+)\.zip$")


def source_root(endpoint: str, spec: SourceSpec) -> str:
    """Return the immutable resolve root for one source."""

    normalized_endpoint = endpoint.rstrip("/")
    if not normalized_endpoint:
        raise ValueError("mirror endpoint must be non-empty")
    return f"{normalized_endpoint}/datasets/{spec.repository}/resolve/{spec.revision}"


def metadata_url(endpoint: str, spec: SourceSpec) -> str:
    """Return the pinned metadata URL for one source."""

    return f"{source_root(endpoint, spec)}/{spec.metadata_path.lstrip('/')}"


def discover_openvid_archives(
    endpoint: str,
    spec: SourceSpec,
    *,
    max_archives: int | None = None,
    fetch_json: Callable[[str], object] | None = None,
) -> tuple[str, ...]:
    """Discover every complete OpenVid zip from the mirror tree API."""

    if not spec.openvid_tree_path:
        raise ValueError("OpenVid source spec must define openvid_tree_path")
    if max_archives is not None and max_archives <= 0:
        raise ValueError("max_archives must be positive when provided")
    api_url = (
        f"{endpoint.rstrip('/')}/api/datasets/{spec.repository}/tree/{spec.revision}/"
        f"{spec.openvid_tree_path}?recursive=false&expand=false"
    )
    payload = _fetch_json(api_url) if fetch_json is None else fetch_json(api_url)
    if not isinstance(payload, list):
        raise ValueError("mirror tree response must be a JSON list")
    paths: list[tuple[int, str]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        raw_path = item.get("path")
        raw_type = item.get("type")
        if not isinstance(raw_path, str) or raw_type != "file":
            continue
        match = _OPENVID_PART.search(raw_path)
        if match is not None:
            paths.append((int(match.group(1)), raw_path))
    if not paths:
        raise ValueError("mirror tree did not expose any complete OpenVid zip archives")
    paths.sort(key=lambda item: (item[0], item[1]))
    if max_archives is not None:
        paths = paths[:max_archives]
    root = source_root(endpoint, spec)
    return tuple(f"{root}/{path}" for _, path in paths)


def build_adapters(
    client: RangeClient,
    *,
    endpoint: str,
    specs: Mapping[str, SourceSpec] = DEFAULT_SOURCE_SPECS,
    chunk_bytes: int = _DEFAULT_METADATA_CHUNK_BYTES,
    openvid_max_index_entries: int = 2_000_000,
    openvid_archive_limit: int | None = None,
) -> dict[str, DatasetAdapter]:
    """Construct all three revision-pinned adapters without reading media."""

    required = {SourceName.DATAVERSE.value, SourceName.OPENVID.value, SourceName.SEEDANCE.value}
    if set(specs) != required:
        raise ValueError(f"source specs must contain exactly {sorted(required)}")
    dataverse = specs[SourceName.DATAVERSE.value]
    openvid = specs[SourceName.OPENVID.value]
    seedance = specs[SourceName.SEEDANCE.value]
    return {
        SourceName.DATAVERSE.value: DataVerseAdapter(
            client,
            metadata_url=metadata_url(endpoint, dataverse),
            resolve_root=source_root(endpoint, dataverse),
            revision=dataverse.revision,
            chunk_bytes=chunk_bytes,
        ),
        SourceName.OPENVID.value: OpenVidAdapter(
            client,
            metadata_url=metadata_url(endpoint, openvid),
            archive_urls=discover_openvid_archives(
                endpoint,
                openvid,
                max_archives=openvid_archive_limit,
            ),
            revision=openvid.revision,
            chunk_bytes=chunk_bytes,
            max_index_entries=openvid_max_index_entries,
        ),
        SourceName.SEEDANCE.value: SeedanceAdapter(
            client,
            metadata_url=metadata_url(endpoint, seedance),
            resolve_root=source_root(endpoint, seedance),
            revision=seedance.revision,
            chunk_bytes=chunk_bytes,
        ),
    }


def normalize_local_dataset_roots(
    sources: Sequence[Path | str],
) -> dict[str, Path]:
    """Validate and canonicalize the three local dataset directories."""

    if len(sources) != len(LOCAL_DATASET_DIRECTORIES):
        raise ValueError("dataset source catalog must contain exactly three local directories")
    source_by_directory = {
        directory: source for source, directory in LOCAL_DATASET_DIRECTORIES.items()
    }
    roots: dict[str, Path] = {}
    for raw_source in sources:
        path = Path(raw_source).expanduser().resolve()
        source = source_by_directory.get(path.name)
        if source is None:
            raise ValueError(
                "local dataset directory must use one of the canonical names "
                f"{sorted(source_by_directory)}: {path}"
            )
        if source in roots:
            raise ValueError(f"local dataset source is duplicated: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"local dataset directory does not exist: {path}")
        roots[source] = path
    missing = set(LOCAL_DATASET_DIRECTORIES) - set(roots)
    if missing:
        raise ValueError(f"local dataset catalog is missing required sources: {sorted(missing)}")
    return roots


def build_local_adapters(
    client: RangeClient,
    *,
    dataset_roots: Mapping[str, Path | str],
    chunk_bytes: int = _DEFAULT_METADATA_CHUNK_BYTES,
    openvid_max_index_entries: int = 2_000_000,
    openvid_archive_limit: int | None = None,
    prefer_manifests: bool = True,
) -> dict[str, DatasetAdapter]:
    """Construct adapters over the exact archives and media present on local disk."""

    required = set(LOCAL_DATASET_DIRECTORIES)
    if set(dataset_roots) != required:
        raise ValueError(f"local dataset roots must contain exactly {sorted(required)}")
    roots = {source: Path(dataset_roots[source]).expanduser().resolve() for source in required}
    for source, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"local {source} dataset directory does not exist: {root}")

    manifest_paths = {source: root / LOCAL_MANIFEST_NAME for source, root in roots.items()}
    available_manifests = {source for source, path in manifest_paths.items() if path.is_file()}
    if prefer_manifests and available_manifests:
        if available_manifests != required:
            missing = sorted(required - available_manifests)
            raise FileNotFoundError(
                f"local TARDIS manifests are incomplete; missing sources: {missing}"
            )
        revisions = {
            SourceName.DATAVERSE.value: DATAVERSE_REVISION,
            SourceName.OPENVID.value: OPENVID_REVISION,
            SourceName.SEEDANCE.value: SEEDANCE_REVISION,
        }
        return {
            source: LocalManifestAdapter(
                client,
                source=source,
                revision=revisions[source],
                manifest_url=manifest_paths[source].as_uri(),
                chunk_bytes=chunk_bytes,
            )
            for source in LOCAL_DATASET_DIRECTORIES
        }

    dataverse_root = roots[SourceName.DATAVERSE.value]
    dataverse_archives = tuple(sorted(dataverse_root.rglob("*.tar")))
    if not dataverse_archives:
        raise FileNotFoundError(f"no DataVerse TAR archives found below {dataverse_root}")
    dataverse_shards = frozenset(
        _dataverse_shard(path, dataverse_root) for path in dataverse_archives
    )

    openvid_root = roots[SourceName.OPENVID.value]
    openvid_archives = tuple(sorted(openvid_root.rglob("*.zip")))
    if openvid_archive_limit is not None:
        if openvid_archive_limit <= 0:
            raise ValueError("openvid_archive_limit must be positive when provided")
        openvid_archives = openvid_archives[:openvid_archive_limit]
    if not openvid_archives:
        raise FileNotFoundError(f"no complete OpenVid ZIP archives found below {openvid_root}")
    openvid_metadata = _first_existing_file(
        openvid_root / "data/train/OpenVid-1M.csv",
        openvid_root / "data/train/OpenVidHD.csv",
    )

    seedance_root = roots[SourceName.SEEDANCE.value]
    seedance_media = frozenset(
        path.relative_to(seedance_root).as_posix()
        for path in seedance_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov"}
    )
    if not seedance_media:
        raise FileNotFoundError(f"no Seedance video files found below {seedance_root}")
    return {
        SourceName.DATAVERSE.value: DataVerseAdapter(
            client,
            metadata_url=_required_file(dataverse_root / "annotation.json").as_uri(),
            resolve_root=dataverse_root.as_uri(),
            revision=DATAVERSE_REVISION,
            chunk_bytes=chunk_bytes,
            available_archive_shards=dataverse_shards,
        ),
        SourceName.OPENVID.value: OpenVidAdapter(
            client,
            metadata_url=openvid_metadata.as_uri(),
            archive_urls=tuple(path.as_uri() for path in openvid_archives),
            revision=OPENVID_REVISION,
            chunk_bytes=chunk_bytes,
            max_index_entries=openvid_max_index_entries,
        ),
        SourceName.SEEDANCE.value: SeedanceAdapter(
            client,
            metadata_url=_required_file(seedance_root / "metadata.jsonl").as_uri(),
            resolve_root=seedance_root.as_uri(),
            revision=SEEDANCE_REVISION,
            chunk_bytes=chunk_bytes,
            available_media_paths=seedance_media,
        ),
    }


def _dataverse_shard(path: Path, root: Path) -> int:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"DataVerse archive is outside its dataset root: {path}") from error
    if not path.stem.isdecimal():
        raise ValueError(f"DataVerse archive name must be a numeric shard: {relative}")
    shard = int(path.stem)
    expected_group = f"{shard // 200:05d}"
    if path.parent.name != expected_group:
        raise ValueError(
            f"DataVerse archive {relative} must be placed below group {expected_group}"
        )
    return shard


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required local dataset file does not exist: {path}")
    return path


def _first_existing_file(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "none of the required local dataset metadata files exist: "
        + ", ".join(str(path) for path in paths)
    )


def _fetch_json(url: str) -> object:
    try:
        response = httpx.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
        return cast(Any, response).json()
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError(f"failed to read mirror tree metadata: {url}") from error
