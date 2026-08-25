from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from tardis.data.contracts import ObjectTooLargeError, RangeProtocolError
from tardis.data.http_range import RangeClient


@dataclass
class ServerState:
    payload: bytes
    requests: list[tuple[str, str | None]] = field(default_factory=list)
    transient_failures: int = 0
    force_full_response: bool = False
    corrupt_content_range: bool = False


@contextmanager
def range_server(payload: bytes) -> Iterator[tuple[str, ServerState]]:
    state = ServerState(payload=payload)

    class Handler(BaseHTTPRequestHandler):
        protocol_version: ClassVar[str] = "HTTP/1.1"

        def do_HEAD(self) -> None:  # noqa: N802
            state.requests.append(("HEAD", self.headers.get("Range")))
            self.send_response(200)
            self.send_header("Content-Length", str(len(state.payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", '"fixture-v1"')
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers.get("Range")
            state.requests.append(("GET", range_header))
            if state.transient_failures > 0:
                state.transient_failures -= 1
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if state.force_full_response or range_header is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(state.payload)))
                self.end_headers()
                self.wfile.write(state.payload)
                return

            unit, requested = range_header.split("=", maxsplit=1)
            assert unit == "bytes"
            if requested.startswith("-"):
                length = int(requested[1:])
                start = max(0, len(state.payload) - length)
                end = len(state.payload) - 1
            else:
                start_text, end_text = requested.split("-", maxsplit=1)
                start = int(start_text)
                end = min(int(end_text), len(state.payload) - 1)

            body = state.payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            content_start = start + 1 if state.corrupt_content_range else start
            self.send_header(
                "Content-Range",
                f"bytes {content_start}-{end}/{len(state.payload)}",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/fixture.bin", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reads_exact_inclusive_and_suffix_ranges() -> None:
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    with range_server(payload) as (url, state), RangeClient(max_object_bytes=1024) as client:
        assert client.read(url, 7, 13) == payload[7:14]
        assert client.read_suffix(url, 5) == payload[-5:]

        telemetry = client.telemetry.snapshot()

    assert state.requests == [("GET", "bytes=7-13"), ("GET", "bytes=-5")]
    assert telemetry.request_count == 2
    assert telemetry.bytes_received == 12
    assert telemetry.retry_count == 0


def test_inspect_returns_remote_object_contract() -> None:
    payload = b"metadata"
    with range_server(payload) as (url, _), RangeClient(max_object_bytes=1024) as client:
        remote = client.inspect(url)

    assert remote.size == len(payload)
    assert remote.etag == '"fixture-v1"'
    assert remote.accept_ranges is True


def test_rejects_non_partial_or_mismatched_partial_responses() -> None:
    payload = b"0123456789"
    with range_server(payload) as (url, state), RangeClient(max_object_bytes=1024) as client:
        state.force_full_response = True
        with pytest.raises(RangeProtocolError, match="206"):
            client.read(url, 0, 3)

        state.force_full_response = False
        state.corrupt_content_range = True
        with pytest.raises(RangeProtocolError, match="Content-Range"):
            client.read(url, 0, 3)

    assert len(state.requests) == 2


def test_enforces_allocation_limit_before_network_request() -> None:
    with range_server(b"x" * 64) as (url, state), RangeClient(max_object_bytes=8) as client:
        with pytest.raises(ObjectTooLargeError, match="9"):
            client.read(url, 0, 8)
        with pytest.raises(ObjectTooLargeError, match="9"):
            client.read_suffix(url, 9)

    assert state.requests == []


def test_retries_transient_statuses_with_bounded_attempts() -> None:
    with range_server(b"retry-success") as (url, state):
        state.transient_failures = 2
        with RangeClient(max_object_bytes=1024, max_retries=2, backoff_base_seconds=0) as client:
            assert client.read(url, 0, 4) == b"retry"
            telemetry = client.telemetry.snapshot()

    assert len(state.requests) == 3
    assert telemetry.request_count == 3
    assert telemetry.retry_count == 2


def test_does_not_retry_non_transient_protocol_failure() -> None:
    with (
        range_server(b"no-retry") as (url, state),
        RangeClient(
            max_object_bytes=1024,
            max_retries=3,
            backoff_base_seconds=0,
        ) as client,
    ):
        state.force_full_response = True
        with pytest.raises(RangeProtocolError):
            client.read(url, 0, 1)

    assert len(state.requests) == 1


def test_line_iterator_stops_requesting_after_close() -> None:
    payload = b"first\nsecond\nthird\nfourth\n"
    with range_server(payload) as (url, state), RangeClient(max_object_bytes=8) as client:
        lines = client.iter_lines(url, chunk_bytes=8)
        assert next(lines) == b"first"
        lines.close()
        completed_requests = len(state.requests)

    assert completed_requests == 2  # HEAD plus one bounded GET.
    assert len(state.requests) == completed_requests


def test_line_iterator_rejects_unbounded_line_buffer() -> None:
    payload = b"x" * 17
    with (
        range_server(payload) as (url, _),
        RangeClient(max_object_bytes=8) as client,
        pytest.raises(ObjectTooLargeError, match="line buffer"),
    ):
        list(client.iter_lines(url, chunk_bytes=8, max_line_bytes=8))


def test_local_file_uri_supports_bounded_ranges_and_line_iteration(tmp_path: Path) -> None:
    payload = b"first\nsecond\nthird\n"
    path = tmp_path / "metadata.txt"
    path.write_bytes(payload)

    with RangeClient(max_object_bytes=8) as client:
        local = client.inspect(path.as_uri())
        middle = client.read(path.as_uri(), 6, 11)
        suffix = client.read_suffix(path.as_uri(), 6)
        lines = list(client.iter_lines(path.as_uri(), chunk_bytes=8))

    assert local.size == len(payload)
    assert local.accept_ranges is True
    assert middle == b"second"
    assert suffix == b"third\n"
    assert lines == [b"first", b"second", b"third"]


def test_local_file_uri_rejects_out_of_bounds_ranges(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abcd")

    with (
        RangeClient(max_object_bytes=8) as client,
        pytest.raises(
            RangeProtocolError,
            match="exceeds",
        ),
    ):
        client.read(path.as_uri(), 0, 4)
