from __future__ import annotations

import io
import struct
import tarfile
import threading
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from tardis.data.archives import (
    ArchiveFormatError,
    RemoteTarReader,
    RemoteZipReader,
    UnsupportedArchiveError,
)
from tardis.data.contracts import ObjectTooLargeError
from tardis.data.http_range import RangeClient


@dataclass
class ArchiveServerState:
    payload: bytes
    ranges: list[str] = field(default_factory=list)


@contextmanager
def archive_server(payload: bytes) -> Iterator[tuple[str, ArchiveServerState]]:
    state = ArchiveServerState(payload)

    class Handler(BaseHTTPRequestHandler):
        protocol_version: ClassVar[str] = "HTTP/1.1"

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(state.payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers["Range"]
            state.ranges.append(range_header)
            requested = range_header.removeprefix("bytes=")
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
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(state.payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/archive", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def make_tar(*, archive_format: int = tarfile.USTAR_FORMAT, long_name: bool = False) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=archive_format) as archive:
        for name, payload in (("first.txt", b"first"), ("nested/second.bin", b"second-data")):
            resolved_name = ("a" * 180 + ".txt") if long_name and name == "first.txt" else name
            info = tarfile.TarInfo(resolved_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def make_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("stored.txt", b"stored-payload", compress_type=zipfile.ZIP_STORED)
        archive.writestr("deflated.txt", b"deflated-" * 20, compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def make_zip64() -> bytes:
    name = b"zip64.txt"
    payload = b"zip64-payload"
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    zip64_local_extra = struct.pack("<HHQQ", 0x0001, 16, len(payload), len(payload))
    local = struct.pack(
        "<4s5H3I2H",
        b"PK\x03\x04",
        45,
        0,
        0,
        0,
        0,
        crc,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(name),
        len(zip64_local_extra),
    )
    local_record = local + name + zip64_local_extra + payload

    zip64_central_extra = struct.pack("<HHQQQ", 0x0001, 24, len(payload), len(payload), 0)
    central = (
        struct.pack(
            "<4s6H3I5H2I",
            b"PK\x01\x02",
            45,
            45,
            0,
            0,
            0,
            0,
            crc,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name),
            len(zip64_central_extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + name
        + zip64_central_extra
    )
    central_offset = len(local_record)
    central_size = len(central)

    zip64_eocd_offset = central_offset + central_size
    zip64_eocd = struct.pack(
        "<4sQ2H2I4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        1,
        1,
        central_size,
        central_offset,
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, zip64_eocd_offset, 1)
    eocd = struct.pack(
        "<4s4H2IH",
        b"PK\x05\x06",
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    return local_record + central + zip64_eocd + locator + eocd


def patch_first_zip_flags(payload: bytes, flag: int) -> bytes:
    result = bytearray(payload)
    central_offset = result.index(b"PK\x01\x02")
    local_flags = struct.unpack_from("<H", result, 6)[0]
    central_flags = struct.unpack_from("<H", result, central_offset + 8)[0]
    struct.pack_into("<H", result, 6, local_flags | flag)
    struct.pack_into("<H", result, central_offset + 8, central_flags | flag)
    return bytes(result)


def test_tar_indexes_headers_and_reads_only_selected_payload_range() -> None:
    payload = make_tar()
    with archive_server(payload) as (url, state), RangeClient(max_object_bytes=4096) as client:
        reader = RemoteTarReader(client, url, max_member_bytes=1024)
        members = list(reader.iter_members())
        ranges_after_index = list(state.ranges)
        assert reader.read_member(members[1]) == b"second-data"

    assert [(item.name, item.uncompressed_size) for item in members] == [
        ("first.txt", 5),
        ("nested/second.bin", 11),
    ]
    assert state.ranges[-1] == (f"bytes={members[1].data_offset}-{members[1].data_offset + 10}")
    assert state.ranges[:-1] == ranges_after_index


@pytest.mark.parametrize("archive_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_tar_rejects_extended_name_formats(archive_format: int) -> None:
    payload = make_tar(archive_format=archive_format, long_name=True)
    with (
        archive_server(payload) as (url, _),
        RangeClient(max_object_bytes=4096) as client,
        pytest.raises(UnsupportedArchiveError, match="PAX|GNU"),
    ):
        list(RemoteTarReader(client, url).iter_members())


def test_zip32_reads_stored_and_deflated_members_with_crc() -> None:
    payload = make_zip()
    with (
        archive_server(payload) as (url, state),
        RangeClient(max_object_bytes=128 * 1024) as client,
    ):
        reader = RemoteZipReader(client, url, max_member_bytes=4096)
        members = list(reader.iter_members())
        assert state.ranges == [
            f"bytes=-{len(payload)}",
            f"bytes={payload.index(b'PK\x01\x02')}-{payload.rindex(b'PK\x05\x06') - 1}",
        ]
        stored = members[0]
        deflated = members[1]
        before_stored = len(state.ranges)
        assert reader.read_member(stored) == b"stored-payload"
        stored_ranges = state.ranges[before_stored:]
        assert reader.read_member(deflated) == b"deflated-" * 20

    assert [member.name for member in members] == ["stored.txt", "deflated.txt"]
    assert stored_ranges[0] == "bytes=0-29"
    stored_data_offset = 30 + len("stored.txt")
    assert stored_ranges[-1] == (
        f"bytes={stored_data_offset}-{stored_data_offset + stored.compressed_size - 1}"
    )


def test_zip64_eocd_and_member_offsets_are_resolved() -> None:
    payload = make_zip64()
    with archive_server(payload) as (url, _), RangeClient(max_object_bytes=128 * 1024) as client:
        reader = RemoteZipReader(client, url)
        members = list(reader.iter_members())
        result = reader.read_member(members[0])

    assert len(members) == 1
    assert members[0].name == "zip64.txt"
    assert result == b"zip64-payload"


def test_zip_verifies_crc32() -> None:
    payload = bytearray(make_zip())
    marker = payload.index(b"stored-payload")
    payload[marker] ^= 0xFF
    with (
        archive_server(bytes(payload)) as (url, _),
        RangeClient(max_object_bytes=128 * 1024) as client,
    ):
        reader = RemoteZipReader(client, url)
        member = next(item for item in reader.iter_members() if item.name == "stored.txt")
        with pytest.raises(ArchiveFormatError, match="CRC32"):
            reader.read_member(member)


def test_zip_rejects_oversized_member_before_payload_request() -> None:
    payload = make_zip()
    with (
        archive_server(payload) as (url, state),
        RangeClient(max_object_bytes=128 * 1024) as client,
    ):
        reader = RemoteZipReader(client, url, max_member_bytes=8)
        member = next(item for item in reader.iter_members() if item.name == "stored.txt")
        requests_before_read = len(state.ranges)
        with pytest.raises(ObjectTooLargeError, match="uncompressed"):
            reader.read_member(member)

    assert len(state.ranges) == requests_before_read


@pytest.mark.parametrize(
    ("flag", "message"),
    [(0x1, "encrypted"), (0x8, "data descriptor")],
)
def test_zip_rejects_unsafe_member_flags(flag: int, message: str) -> None:
    payload = patch_first_zip_flags(make_zip(), flag)
    with (
        archive_server(payload) as (url, _),
        RangeClient(max_object_bytes=128 * 1024) as client,
        pytest.raises(UnsupportedArchiveError, match=message),
    ):
        list(RemoteZipReader(client, url).iter_members())


def test_zip_rejects_multi_disk_eocd() -> None:
    payload = bytearray(make_zip())
    eocd_offset = payload.rindex(b"PK\x05\x06")
    struct.pack_into("<H", payload, eocd_offset + 4, 1)
    with (
        archive_server(bytes(payload)) as (url, _),
        RangeClient(max_object_bytes=128 * 1024) as client,
        pytest.raises(UnsupportedArchiveError, match="multi-disk"),
    ):
        list(RemoteZipReader(client, url).iter_members())
