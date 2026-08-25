"""Shared contracts for bounded local and remote data access."""

from __future__ import annotations

from dataclasses import dataclass, field

type JsonScalar = str | int | float | bool | None
type MetadataValue = JsonScalar | list[MetadataValue] | dict[str, MetadataValue]


class RemoteDataError(RuntimeError):
    """Base class for bounded remote-data failures."""


class RangeProtocolError(RemoteDataError):
    """Raised when a server violates the HTTP byte-range contract."""


class ObjectTooLargeError(RemoteDataError):
    """Raised before a requested in-memory allocation exceeds its hard limit."""


class TransientRemoteError(RemoteDataError):
    """Raised after bounded retries of a transient remote failure are exhausted."""


class MetadataParseError(RemoteDataError):
    """Raised for malformed or incomplete source metadata."""


@dataclass(frozen=True, slots=True)
class RemoteObject:
    """Metadata required to stream one remote object safely."""

    size: int
    etag: str | None
    accept_ranges: bool


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """Canonical caption-video pair emitted by every source adapter."""

    id: str
    caption: str
    media_locator: str
    source: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RangeTelemetrySnapshot:
    """Immutable transport counters suitable for manifests and logs."""

    request_count: int
    bytes_received: int
    retry_count: int
    elapsed_seconds: float

    @property
    def throughput_bytes_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.bytes_received / self.elapsed_seconds
