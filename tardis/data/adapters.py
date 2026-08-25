"""Mirror-pinned adapters for the three prompt-video sources."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping, Sequence, Set
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import parse_qs, quote, unquote, urlsplit, urlunsplit

from tardis.data.archives import ArchiveMember, RemoteTarReader, RemoteZipReader
from tardis.data.contracts import (
    MetadataParseError,
    MetadataValue,
    ObjectTooLargeError,
    VideoRecord,
)
from tardis.data.http_range import RangeClient
from tardis.data.metadata import (
    iter_csv_records,
    iter_json_array_records,
    iter_jsonl_objects,
)


class DatasetAdapter(Protocol):
    """A revision-pinned stream of canonical prompt-video records."""

    source: str
    revision: str

    def iter_records(self) -> Iterator[VideoRecord]: ...


@dataclass(slots=True)
class AdapterStats:
    metadata_records: int = 0
    emitted_records: int = 0
    unmatched_metadata_records: int = 0
    indexed_media_members: int = 0
    excluded_archives: int = 0


class LocalManifestAdapter:
    """Read one prefiltered local prompt-video manifest."""

    def __init__(
        self,
        client: RangeClient,
        *,
        source: str,
        revision: str,
        manifest_url: str,
        chunk_bytes: int,
    ) -> None:
        self.client = client
        self.source = source.strip()
        self.revision = _require_revision(revision)
        self.manifest_url = manifest_url
        self.chunk_bytes = chunk_bytes
        self.stats = AdapterStats()
        if not self.source:
            raise ValueError("manifest source cannot be empty")

    def iter_records(self) -> Iterator[VideoRecord]:
        for line_number, value in iter_jsonl_objects(
            self.client,
            self.manifest_url,
            chunk_bytes=self.chunk_bytes,
        ):
            self.stats.metadata_records += 1
            source = _required_string(value, "source", line_number)
            if source != self.source:
                raise MetadataParseError(
                    f"manifest line {line_number} source {source!r} does not match {self.source!r}"
                )
            metadata_value = value.get("metadata", {})
            if not isinstance(metadata_value, dict):
                raise MetadataParseError(f"manifest line {line_number} metadata must be an object")
            metadata = {str(key): _metadata_value(item) for key, item in metadata_value.items()}
            if metadata.get("revision") != self.revision:
                raise MetadataParseError(
                    f"manifest line {line_number} does not match revision {self.revision}"
                )
            self.stats.emitted_records += 1
            yield VideoRecord(
                id=_required_string(value, "id", line_number),
                caption=_required_string(value, "caption", line_number),
                media_locator=_required_string(value, "media_locator", line_number),
                source=self.source,
                metadata=metadata,
            )


class DataVerseAdapter:
    """Stream DataVerse annotations and derive their deterministic TAR shard."""

    source = "dataverse"

    def __init__(
        self,
        client: RangeClient,
        *,
        metadata_url: str,
        resolve_root: str,
        revision: str,
        chunk_bytes: int,
        available_archive_shards: Set[int] | None = None,
    ) -> None:
        self.client = client
        self.metadata_url = metadata_url
        self.resolve_root = resolve_root.rstrip("/")
        self.revision = _require_revision(revision)
        self.chunk_bytes = chunk_bytes
        self.available_archive_shards = (
            None
            if available_archive_shards is None
            else frozenset(int(shard) for shard in available_archive_shards)
        )
        if self.available_archive_shards is not None and not self.available_archive_shards:
            raise ValueError("available_archive_shards cannot be empty")
        self.stats = AdapterStats()

    def iter_records(self) -> Iterator[VideoRecord]:
        records = iter_json_array_records(
            self.client,
            self.metadata_url,
            source=self.source,
            id_field="video",
            caption_field="text",
            media_field="video",
            chunk_bytes=self.chunk_bytes,
        )
        for record in records:
            self.stats.metadata_records += 1
            video_name = PurePosixPath(record.media_locator).name
            stem = PurePosixPath(video_name).stem
            if not stem.isdecimal():
                raise MetadataParseError(f"invalid DataVerse numeric video ID {video_name!r}")
            shard = int(stem) // 1000
            if (
                self.available_archive_shards is not None
                and shard not in self.available_archive_shards
            ):
                self.stats.unmatched_metadata_records += 1
                continue
            group = shard // 200
            archive_url = f"{self.resolve_root}/{group:05d}/{shard:06d}.tar"
            metadata = dict(record.metadata)
            metadata.update({"revision": self.revision, "archive_shard": shard})
            self.stats.emitted_records += 1
            yield VideoRecord(
                id=video_name,
                caption=record.caption,
                media_locator=f"tar+{archive_url}#member=./{quote(video_name, safe='.')}",
                source=self.source,
                metadata=metadata,
            )


class OpenVidAdapter:
    """Join OpenVid CSV metadata with lazily indexed remote ZIP members."""

    source = "openvid"

    def __init__(
        self,
        client: RangeClient,
        *,
        metadata_url: str,
        archive_urls: Sequence[str],
        revision: str,
        chunk_bytes: int,
        max_index_entries: int,
    ) -> None:
        if max_index_entries <= 0:
            raise ValueError("max_index_entries must be positive")
        self.client = client
        self.metadata_url = metadata_url
        self.archive_urls = tuple(archive_urls)
        self.revision = _require_revision(revision)
        self.chunk_bytes = chunk_bytes
        self.max_index_entries = max_index_entries
        self.stats = AdapterStats()

    def iter_records(self) -> Iterator[VideoRecord]:
        catalog = self._build_catalog()
        records = iter_csv_records(
            self.client,
            self.metadata_url,
            source=self.source,
            id_field="video",
            caption_field="caption",
            media_field="video",
            chunk_bytes=self.chunk_bytes,
        )
        for record in records:
            self.stats.metadata_records += 1
            video_name = PurePosixPath(record.id).name
            location = catalog.get(video_name)
            if location is None:
                self.stats.unmatched_metadata_records += 1
                continue
            archive_url, member_name = location
            metadata = dict(record.metadata)
            metadata["revision"] = self.revision
            self.stats.emitted_records += 1
            yield VideoRecord(
                id=video_name,
                caption=record.caption,
                media_locator=f"zip+{archive_url}#member={quote(member_name, safe='/._-')}",
                source=self.source,
                metadata=metadata,
            )

    def _build_catalog(self) -> dict[str, tuple[str, str]]:
        catalog: dict[str, tuple[str, str]] = {}
        for archive_url in self.archive_urls:
            reader = RemoteZipReader(self.client, archive_url)
            for member in reader.iter_members():
                basename = PurePosixPath(member.name).name
                if not basename.lower().endswith((".mp4", ".webm", ".mov")):
                    continue
                existing = catalog.get(basename)
                if existing is not None:
                    raise MetadataParseError(
                        f"OpenVid filename {basename!r} occurs in multiple archives"
                    )
                if len(catalog) >= self.max_index_entries:
                    raise MetadataParseError(
                        f"OpenVid catalog exceeds max_index_entries={self.max_index_entries}"
                    )
                catalog[basename] = (archive_url, member.name)
                self.stats.indexed_media_members += 1
        return catalog


class SeedanceAdapter:
    """Stream Seedance JSONL and resolve individual media objects."""

    source = "seedance"

    def __init__(
        self,
        client: RangeClient,
        *,
        metadata_url: str,
        resolve_root: str,
        revision: str,
        chunk_bytes: int,
        available_media_paths: Set[str] | None = None,
    ) -> None:
        self.client = client
        self.metadata_url = metadata_url
        self.resolve_root = resolve_root.rstrip("/")
        self.revision = _require_revision(revision)
        self.chunk_bytes = chunk_bytes
        self.available_media_paths = (
            None
            if available_media_paths is None
            else frozenset(_normalized_media_path(path) for path in available_media_paths)
        )
        if self.available_media_paths is not None and not self.available_media_paths:
            raise ValueError("available_media_paths cannot be empty")
        self.stats = AdapterStats()

    def iter_records(self) -> Iterator[VideoRecord]:
        raw_records = iter_jsonl_objects(
            self.client,
            self.metadata_url,
            chunk_bytes=self.chunk_bytes,
        )
        for line_number, value in raw_records:
            self.stats.metadata_records += 1
            record_id = _required_string(value, "id", line_number)
            raw_prompt = _required_string(value, "raw_p", line_number)
            metadata: dict[str, MetadataValue] = {
                str(key): _metadata_value(item)
                for key, item in value.items()
                if key not in {"id", "raw_p", "file_name"}
            }
            prompt = _seedance_english_prompt(metadata) or raw_prompt
            media = _seedance_video_path(metadata)
            if media is None:
                media = _required_string(value, "file_name", line_number)
            media = _normalized_media_path(media)
            if self.available_media_paths is not None and media not in self.available_media_paths:
                self.stats.unmatched_metadata_records += 1
                continue
            spec = metadata.get("spec")
            if isinstance(spec, dict):
                for key in ("duration", "width", "height"):
                    if key in spec:
                        metadata[key] = spec[key]
            metadata["revision"] = self.revision
            self.stats.emitted_records += 1
            yield VideoRecord(
                id=record_id,
                caption=prompt,
                media_locator=f"{self.resolve_root}/{quote(media, safe='/._-')}",
                source=self.source,
                metadata=metadata,
            )


def _seedance_english_prompt(metadata: dict[str, MetadataValue]) -> str | None:
    i18n = metadata.get("i18n")
    if not isinstance(i18n, dict):
        return None
    english = i18n.get("en")
    if not isinstance(english, dict):
        return None
    prompt = english.get("p")
    return prompt.strip() if isinstance(prompt, str) and prompt.strip() else None


def _seedance_video_path(metadata: dict[str, MetadataValue]) -> str | None:
    media = metadata.get("media")
    if not isinstance(media, dict):
        return None
    video = media.get("v")
    return video.strip() if isinstance(video, str) and video.strip() else None


def _normalized_media_path(path: str) -> str:
    normalized = str(PurePosixPath(path.strip().lstrip("/")))
    if not normalized or normalized == ".":
        raise MetadataParseError("media path cannot be empty")
    return normalized


def _require_revision(revision: str) -> str:
    revision = revision.strip()
    if not revision:
        raise ValueError("dataset revision cannot be empty")
    return revision


def _required_string(value: dict[str, object], field: str, line_number: int) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise MetadataParseError(f"metadata line {line_number} has no non-empty {field!r}")
    return item.strip()


def _metadata_value(value: object) -> MetadataValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return str(value)


def read_record_media(
    client: RangeClient,
    record: VideoRecord,
    *,
    max_media_bytes: int,
) -> bytes:
    """Materialize one selected direct or archive-backed video entirely in RAM."""

    if max_media_bytes <= 0:
        raise ValueError("max_media_bytes must be positive")
    locator = record.media_locator
    if locator.startswith("tar+"):
        archive_url, member_name = _parse_archive_locator(locator, "tar+")
        member = _find_cached_tar_member(
            client,
            archive_url,
            member_name,
            max_member_bytes=max_media_bytes,
        )
        tar_reader = RemoteTarReader(client, archive_url, max_member_bytes=max_media_bytes)
        return tar_reader.read_member(member)
    if locator.startswith("zip+"):
        archive_url, member_name = _parse_archive_locator(locator, "zip+")
        member = _find_cached_zip_member(
            client,
            archive_url,
            member_name,
            max_member_bytes=max_media_bytes,
        )
        zip_reader = RemoteZipReader(client, archive_url, max_member_bytes=max_media_bytes)
        return zip_reader.read_member(member)

    remote = client.inspect(locator)
    if remote.size > max_media_bytes:
        raise ObjectTooLargeError(
            f"media object for record {record.id!r} is {remote.size} bytes; "
            f"max_media_bytes={max_media_bytes}"
        )
    if remote.size == 0:
        return b""
    return client.read(locator, 0, remote.size - 1)


@contextmanager
def stream_oversized_local_media(
    client: RangeClient,
    record: VideoRecord,
    *,
    max_media_bytes: int,
) -> Iterator[Path]:
    """Expose oversized local media as a seekable path without materializing it."""

    if max_media_bytes <= 0:
        raise ValueError("max_media_bytes must be positive")
    direct_path = _local_file_uri_path(record.media_locator)
    if direct_path is not None:
        if not direct_path.is_file():
            raise FileNotFoundError(
                f"local media for record {record.id!r} does not exist: {direct_path}"
            )
        yield direct_path
        return
    with stream_oversized_local_tar_member(
        client,
        record,
        max_media_bytes=max_media_bytes,
    ) as staged_path:
        yield staged_path


@contextmanager
def stream_oversized_local_tar_member(
    client: RangeClient,
    record: VideoRecord,
    *,
    max_media_bytes: int,
) -> Iterator[Path]:
    """Stage one oversized local TAR member with bounded buffers for seekable decoding."""

    if max_media_bytes <= 0:
        raise ValueError("max_media_bytes must be positive")
    locator = record.media_locator
    if not locator.startswith("tar+"):
        raise ObjectTooLargeError(
            f"oversized media for record {record.id!r} is not a local TAR member"
        )
    archive_url, member_name = _parse_archive_locator(locator, "tar+")
    archive_path = _local_file_uri_path(archive_url)
    if archive_path is None:
        raise ObjectTooLargeError(
            f"oversized TAR member for record {record.id!r} is not stored locally"
        )
    member = _find_cached_tar_member(
        client,
        archive_url,
        member_name,
        max_member_bytes=max_media_bytes,
    )
    if member.compression != "stored":
        raise ValueError("TAR media members must use stored compression")
    chunk_bytes = min(16 * 1024 * 1024, max_media_bytes, client.max_object_bytes)
    suffix = PurePosixPath(member_name).suffix or ".media"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".tardis-media-",
            suffix=suffix,
            dir=archive_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            copied = 0
            while copied < member.uncompressed_size:
                size = min(chunk_bytes, member.uncompressed_size - copied)
                start = member.data_offset + copied
                payload = client.read(archive_url, start, start + size - 1)
                handle.write(payload)
                copied += len(payload)
            handle.flush()
        if temporary.stat().st_size != member.uncompressed_size:
            raise RuntimeError(
                f"staged TAR member for record {record.id!r} has an unexpected size"
            )
        yield temporary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _local_file_uri_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        raise MetadataParseError(f"invalid local archive URI: {url!r}")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise MetadataParseError(f"local archive URI must be absolute: {url!r}")
    return path


def _find_cached_tar_member(
    client: RangeClient,
    archive_url: str,
    member_name: str,
    *,
    max_member_bytes: int,
) -> ArchiveMember:
    """Resume a per-worker TAR header scan until one requested member is found."""

    raw_state = client.get_archive_index_state(archive_url)
    state: dict[str, object]
    if isinstance(raw_state, Mapping):
        state = dict(raw_state)
    else:
        state = {"members": {}, "next_offset": 0, "complete": False}
    raw_members = state.get("members")
    members: dict[str, ArchiveMember] = {}
    if isinstance(raw_members, Mapping):
        members = {
            str(name): value
            for name, value in raw_members.items()
            if isinstance(name, str) and isinstance(value, ArchiveMember)
        }
    cached = members.get(member_name)
    if cached is not None:
        return cached
    if bool(state.get("complete", False)):
        raise MetadataParseError(
            f"archive member {member_name!r} for record was not found in completed index"
        )

    tar_reader = RemoteTarReader(client, archive_url, max_member_bytes=max_member_bytes)
    raw_next_offset = state.get("next_offset", 0)
    if isinstance(raw_next_offset, bool) or not isinstance(raw_next_offset, int):
        raise MetadataParseError("cached TAR index has a non-integer next offset")
    next_offset = raw_next_offset
    try:
        for member in tar_reader.iter_members(start_offset=next_offset):
            members[member.name] = member
            next_offset = _tar_next_offset(member)
            state["members"] = members
            state["next_offset"] = next_offset
            client.put_archive_index_state(archive_url, state)
            if member.name == member_name:
                return member
    finally:
        # A transport timeout can interrupt a scan after several valid headers.
        # Persist the last safe block boundary so a retry makes forward progress.
        state["members"] = members
        state["next_offset"] = next_offset
        client.put_archive_index_state(archive_url, state)
    state["complete"] = True
    client.put_archive_index_state(archive_url, state)
    raise MetadataParseError(
        f"archive member {member_name!r} for record was not found in TAR archive"
    )


def _tar_next_offset(member: ArchiveMember) -> int:
    padded_size = ((member.uncompressed_size + 511) // 512) * 512
    return member.data_offset + padded_size


def _find_cached_zip_member(
    client: RangeClient,
    archive_url: str,
    member_name: str,
    *,
    max_member_bytes: int,
) -> ArchiveMember:
    """Materialize one ZIP central directory once per DataLoader worker."""

    raw_state = client.get_archive_index_state(archive_url)
    if isinstance(raw_state, Mapping) and raw_state.get("archive_format") == "zip":
        raw_members = raw_state.get("members")
        if isinstance(raw_members, Mapping):
            cached = raw_members.get(member_name)
            if isinstance(cached, ArchiveMember):
                return cached
        if bool(raw_state.get("complete", False)):
            raise MetadataParseError(
                f"archive member {member_name!r} for record was not found in completed index"
            )

    zip_reader = RemoteZipReader(client, archive_url, max_member_bytes=max_member_bytes)
    members = {member.name: member for member in zip_reader.iter_members()}
    client.put_archive_index_state(
        archive_url,
        {"archive_format": "zip", "members": members, "complete": True},
    )
    member = members.get(member_name)
    if member is None:
        raise MetadataParseError(
            f"archive member {member_name!r} for record was not found in ZIP archive"
        )
    return member


def _parse_archive_locator(locator: str, prefix: str) -> tuple[str, str]:
    parsed = urlsplit(locator.removeprefix(prefix))
    parameters = parse_qs(parsed.fragment, strict_parsing=True)
    members = parameters.get("member")
    if members is None or len(members) != 1 or not members[0]:
        raise MetadataParseError(f"archive locator has no unique member parameter: {locator!r}")
    archive_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    return archive_url, members[0]
