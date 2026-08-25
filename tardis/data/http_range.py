"""Strict bounded-memory HTTP byte-range transport."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from urllib.parse import unquote, urlsplit

import httpx

from tardis.data.contracts import (
    ObjectTooLargeError,
    RangeProtocolError,
    RangeTelemetrySnapshot,
    RemoteObject,
    TransientRemoteError,
)

_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_DEFAULT_MAX_LINE_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class _MutableTelemetry:
    request_count: int = 0
    bytes_received: int = 0
    retry_count: int = 0
    elapsed_seconds: float = 0.0


class RangeTelemetry:
    """Thread-safe counters shared by one client."""

    def __init__(self) -> None:
        self._values = _MutableTelemetry()
        self._lock = threading.Lock()

    def record_request(self) -> None:
        with self._lock:
            self._values.request_count += 1

    def record_retry(self) -> None:
        with self._lock:
            self._values.retry_count += 1

    def record_transfer(self, byte_count: int, elapsed_seconds: float) -> None:
        with self._lock:
            self._values.bytes_received += byte_count
            self._values.elapsed_seconds += elapsed_seconds

    def snapshot(self) -> RangeTelemetrySnapshot:
        with self._lock:
            return RangeTelemetrySnapshot(
                request_count=self._values.request_count,
                bytes_received=self._values.bytes_received,
                retry_count=self._values.retry_count,
                elapsed_seconds=self._values.elapsed_seconds,
            )


class RangeClient:
    """Fetch exact object ranges while bounding each materialized allocation."""

    def __init__(
        self,
        *,
        max_object_bytes: int,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.25,
        client: httpx.Client | None = None,
    ) -> None:
        if max_object_bytes <= 0:
            raise ValueError("max_object_bytes must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds cannot be negative")

        self.max_object_bytes = max_object_bytes
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.telemetry = RangeTelemetry()
        # Archive indexes contain headers only, never media payloads. Keeping a
        # small per-worker LRU lets repeated samples from one TAR shard resume
        # the header scan instead of starting at byte zero again.
        self._archive_index_cache: OrderedDict[str, object] = OrderedDict()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_seconds),
        )

    def __enter__(self) -> RangeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get_archive_index_state(self, url: str) -> object | None:
        """Return an in-memory archive scan state and mark it recently used."""

        state = self._archive_index_cache.get(url)
        if state is not None:
            self._archive_index_cache.move_to_end(url)
        return state

    def put_archive_index_state(self, url: str, state: object) -> None:
        """Store bounded archive metadata without persisting any remote bytes."""

        self._archive_index_cache[url] = state
        self._archive_index_cache.move_to_end(url)
        while len(self._archive_index_cache) > 16:
            self._archive_index_cache.popitem(last=False)

    def inspect(self, url: str) -> RemoteObject:
        """Read object metadata without materializing its body."""

        local_path = _local_file_path(url)
        if local_path is not None:
            self.telemetry.record_request()
            try:
                stat = local_path.stat()
            except OSError as error:
                raise RangeProtocolError(f"cannot inspect local file {local_path}") from error
            if not local_path.is_file():
                raise RangeProtocolError(f"local data object is not a file: {local_path}")
            return RemoteObject(
                size=stat.st_size,
                etag=f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
                accept_ranges=True,
            )

        response = self._request_with_retries("HEAD", url)
        try:
            if not 200 <= response.status_code < 300:
                raise RangeProtocolError(
                    f"HEAD {url!r} returned non-success status {response.status_code}"
                )
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                raise RangeProtocolError(f"HEAD {url!r} omitted Content-Length")
            try:
                size = int(content_length)
            except ValueError as error:
                raise RangeProtocolError(
                    f"HEAD {url!r} returned invalid Content-Length {content_length!r}"
                ) from error
            if size < 0:
                raise RangeProtocolError(f"HEAD {url!r} returned a negative object size")
            return RemoteObject(
                size=size,
                etag=response.headers.get("ETag"),
                accept_ranges="bytes" in response.headers.get("Accept-Ranges", "").lower(),
            )
        finally:
            response.close()

    def read(self, url: str, start: int, end: int) -> bytes:
        """Read the exact inclusive range ``[start, end]``."""

        if start < 0:
            raise ValueError("range start cannot be negative")
        if end < start:
            raise ValueError("range end must be greater than or equal to start")
        requested = end - start + 1
        self._check_allocation(requested)
        local_path = _local_file_path(url)
        if local_path is not None:
            return self._read_local(local_path, start=start, end=end)
        return self._read_range(url, f"bytes={start}-{end}", start=start, end=end)

    def read_suffix(self, url: str, length: int) -> bytes:
        """Read up to ``length`` bytes from the end of an object."""

        if length <= 0:
            raise ValueError("suffix length must be positive")
        self._check_allocation(length)
        local_path = _local_file_path(url)
        if local_path is not None:
            try:
                size = local_path.stat().st_size
            except OSError as error:
                raise RangeProtocolError(f"cannot inspect local file {local_path}") from error
            if size == 0:
                self.telemetry.record_request()
                return b""
            start = max(0, size - length)
            return self._read_local(local_path, start=start, end=size - 1)
        return self._read_range(url, f"bytes=-{length}", suffix_length=length)

    def iter_lines(
        self,
        url: str,
        *,
        chunk_bytes: int,
        max_line_bytes: int = _DEFAULT_MAX_LINE_BYTES,
    ) -> Iterator[bytes]:
        """Yield physical lines while requesting at most one bounded chunk at a time."""

        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be positive")
        self._check_allocation(chunk_bytes)
        remote = self.inspect(url)
        if not remote.accept_ranges and remote.size > 0:
            raise RangeProtocolError(f"server for {url!r} does not advertise byte ranges")

        pending = bytearray()
        for start in range(0, remote.size, chunk_bytes):
            end = min(start + chunk_bytes, remote.size) - 1
            parts = self.read(url, start, end).split(b"\n")
            for part_index, part in enumerate(parts):
                line_bytes = len(pending) + len(part)
                if line_bytes > max_line_bytes:
                    raise ObjectTooLargeError(
                        f"line buffer of {line_bytes} bytes exceeds max_line_bytes={max_line_bytes}"
                    )
                pending.extend(part)
                if part_index < len(parts) - 1:
                    yield bytes(pending).removesuffix(b"\r")
                    pending.clear()
        if pending:
            yield bytes(pending).removesuffix(b"\r")

    def iter_chunks(self, url: str, *, chunk_bytes: int) -> Iterator[bytes]:
        """Yield an entire remote object as independently bounded byte chunks."""

        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self._check_allocation(chunk_bytes)
        remote = self.inspect(url)
        if not remote.accept_ranges and remote.size > 0:
            raise RangeProtocolError(f"server for {url!r} does not advertise byte ranges")
        for start in range(0, remote.size, chunk_bytes):
            end = min(start + chunk_bytes, remote.size) - 1
            yield self.read(url, start, end)

    def _check_allocation(self, byte_count: int) -> None:
        if byte_count > self.max_object_bytes:
            raise ObjectTooLargeError(
                f"requested allocation of {byte_count} bytes exceeds "
                f"max_object_bytes={self.max_object_bytes}"
            )

    def _read_local(self, path: Path, *, start: int, end: int) -> bytes:
        requested = end - start + 1
        transfer_started = time.monotonic()
        payload = b""
        self.telemetry.record_request()
        try:
            try:
                size = path.stat().st_size
            except OSError as error:
                raise RangeProtocolError(f"cannot inspect local file {path}") from error
            if start >= size or end >= size:
                raise RangeProtocolError(f"local range {start}-{end} exceeds {path} size {size}")
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    payload = stream.read(requested)
            except OSError as error:
                raise RangeProtocolError(f"cannot read local file {path}") from error
            if len(payload) != requested:
                raise RangeProtocolError(
                    f"local file {path} returned {len(payload)} bytes; expected {requested}"
                )
            return payload
        finally:
            elapsed = time.monotonic() - transfer_started
            self.telemetry.record_transfer(len(payload), elapsed)

    def _read_range(
        self,
        url: str,
        range_header: str,
        *,
        start: int | None = None,
        end: int | None = None,
        suffix_length: int | None = None,
    ) -> bytes:
        response = self._request_with_retries("GET", url, headers={"Range": range_header})
        transfer_started = time.monotonic()
        byte_count = 0
        try:
            if response.status_code != 206:
                raise RangeProtocolError(
                    f"range request for {url!r} requires status 206; got {response.status_code}"
                )
            actual_start, actual_end, total = self._parse_content_range(response, url)
            if suffix_length is None:
                if actual_start != start or actual_end != end:
                    raise RangeProtocolError(
                        f"Content-Range {actual_start}-{actual_end} does not match "
                        f"requested {start}-{end}"
                    )
            else:
                expected_start = max(0, total - suffix_length)
                expected_end = total - 1
                if actual_start != expected_start or actual_end != expected_end:
                    raise RangeProtocolError(
                        f"Content-Range {actual_start}-{actual_end} does not match "
                        f"requested suffix length {suffix_length}"
                    )

            expected_bytes = actual_end - actual_start + 1
            self._check_allocation(expected_bytes)
            content_length = response.headers.get("Content-Length")
            if content_length is not None and content_length != str(expected_bytes):
                raise RangeProtocolError(
                    f"Content-Length {content_length!r} does not match Content-Range size "
                    f"{expected_bytes}"
                )

            result = bytearray()
            for chunk in response.iter_bytes():
                byte_count += len(chunk)
                if byte_count > expected_bytes or byte_count > self.max_object_bytes:
                    raise RangeProtocolError("response body exceeded its declared bounded range")
                result.extend(chunk)
            if byte_count != expected_bytes:
                raise RangeProtocolError(
                    f"response body contains {byte_count} bytes; expected {expected_bytes}"
                )
            return bytes(result)
        finally:
            self.telemetry.record_transfer(byte_count, time.monotonic() - transfer_started)
            response.close()

    @staticmethod
    def _parse_content_range(response: httpx.Response, url: str) -> tuple[int, int, int]:
        header = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE.fullmatch(header)
        if match is None:
            raise RangeProtocolError(
                f"range response for {url!r} has invalid Content-Range {header!r}"
            )
        start, end, total = (int(value) for value in match.groups())
        if start > end or end >= total:
            raise RangeProtocolError(f"range response has inconsistent Content-Range {header!r}")
        return start, end, total

    def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            self.telemetry.record_request()
            try:
                request = self._client.build_request(method, url, headers=headers)
                response = self._client.send(request, stream=True)
            except httpx.TransportError as error:
                last_error = error
            else:
                if response.status_code not in _TRANSIENT_STATUSES:
                    return response
                last_error = TransientRemoteError(
                    f"{method} {url!r} returned transient status {response.status_code}"
                )
                response.close()

            if attempt == self.max_retries:
                break
            self.telemetry.record_retry()
            time.sleep(self.backoff_base_seconds * (2**attempt))

        assert last_error is not None
        raise TransientRemoteError(
            f"{method} {url!r} failed after {self.max_retries + 1} attempts"
        ) from last_error


def _local_file_path(locator: str) -> Path | None:
    parsed = urlsplit(locator)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        raise RangeProtocolError(f"invalid local file URI: {locator!r}")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise RangeProtocolError(f"local file URI must be absolute: {locator!r}")
    return path
