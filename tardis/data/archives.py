"""Bounded remote TAR and ZIP64 readers built on exact HTTP ranges."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from tardis.data.contracts import ObjectTooLargeError, RemoteDataError
from tardis.data.http_range import RangeClient

_TAR_BLOCK_BYTES = 512
_ZIP_EOCD_BYTES = 22
_ZIP_EOCD_SEARCH_BYTES = 65_535 + _ZIP_EOCD_BYTES
_ZIP64_LOCATOR_BYTES = 20
_ZIP64_EOCD_MIN_BYTES = 56


class ArchiveFormatError(RemoteDataError):
    """Raised when archive metadata or member integrity is invalid."""


class UnsupportedArchiveError(RemoteDataError):
    """Raised when safe bounded access cannot support an archive feature."""


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """A remotely addressable regular-file member."""

    name: str
    data_offset: int
    compressed_size: int
    uncompressed_size: int
    compression: Literal["stored", "deflate"]
    crc32: int | None = None
    local_header_offset: int | None = None
    central_directory_offset: int | None = None


class RemoteTarReader:
    """Scan standard USTAR headers without reading skipped member payloads."""

    def __init__(
        self,
        client: RangeClient,
        url: str,
        *,
        max_member_bytes: int | None = None,
        header_chunk_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.client = client
        self.url = url
        self.max_member_bytes = max_member_bytes or client.max_object_bytes
        if self.max_member_bytes <= 0:
            raise ValueError("max_member_bytes must be positive")
        if header_chunk_bytes <= 0:
            raise ValueError("header_chunk_bytes must be positive")
        self.header_chunk_bytes = min(header_chunk_bytes, client.max_object_bytes)

    def iter_members(self, *, start_offset: int = 0) -> Iterator[ArchiveMember]:
        remote = self.client.inspect(self.url)
        if start_offset < 0 or start_offset % _TAR_BLOCK_BYTES:
            raise ValueError("TAR scan start_offset must be a non-negative block boundary")
        offset = start_offset
        buffer_start = -1
        buffer = b""
        while offset + _TAR_BLOCK_BYTES <= remote.size:
            buffer_end = buffer_start + len(buffer)
            if not (buffer_start <= offset and offset + _TAR_BLOCK_BYTES <= buffer_end):
                chunk_end = min(remote.size, offset + self.header_chunk_bytes) - 1
                buffer_start = offset
                buffer = self.client.read(self.url, offset, chunk_end)
            header_start = offset - buffer_start
            header = buffer[header_start : header_start + _TAR_BLOCK_BYTES]
            if header == bytes(_TAR_BLOCK_BYTES):
                return
            _validate_tar_checksum(header, offset)
            type_flag = header[156:157]
            if type_flag in {b"x", b"g"}:
                raise UnsupportedArchiveError(
                    "PAX extended TAR headers are not supported for bounded remote indexing"
                )
            if type_flag in {b"L", b"K"}:
                raise UnsupportedArchiveError(
                    "GNU long-name TAR headers are not supported for bounded remote indexing"
                )

            size = _parse_tar_octal(header[124:136], "member size", offset)
            name = _parse_tar_name(header, offset)
            data_offset = offset + _TAR_BLOCK_BYTES
            padded_size = ((size + _TAR_BLOCK_BYTES - 1) // _TAR_BLOCK_BYTES) * _TAR_BLOCK_BYTES
            next_offset = data_offset + padded_size
            if next_offset > remote.size:
                raise ArchiveFormatError(
                    f"TAR member {name!r} at offset {offset} exceeds archive size"
                )

            if type_flag in {b"", b"\0", b"0"}:
                yield ArchiveMember(
                    name=name,
                    data_offset=data_offset,
                    compressed_size=size,
                    uncompressed_size=size,
                    compression="stored",
                )
            elif type_flag != b"5":
                raise UnsupportedArchiveError(
                    f"unsupported TAR type flag {type_flag!r} for member {name!r}"
                )
            offset = next_offset

        if offset != remote.size:
            raise ArchiveFormatError("TAR archive ends with a partial 512-byte header")

    def read_member(self, member: ArchiveMember) -> bytes:
        if member.compression != "stored" or member.crc32 is not None:
            raise ValueError("member does not belong to a TAR reader")
        self._check_member_size(member.uncompressed_size)
        if member.uncompressed_size == 0:
            return b""
        return self.client.read(
            self.url,
            member.data_offset,
            member.data_offset + member.uncompressed_size - 1,
        )

    def _check_member_size(self, size: int) -> None:
        if size > self.max_member_bytes:
            raise ObjectTooLargeError(
                f"TAR member size {size} exceeds max_member_bytes={self.max_member_bytes}"
            )


class RemoteZipReader:
    """Resolve ZIP32/ZIP64 central directories and fetch selected members."""

    def __init__(
        self,
        client: RangeClient,
        url: str,
        *,
        max_member_bytes: int | None = None,
    ) -> None:
        self.client = client
        self.url = url
        self.max_member_bytes = max_member_bytes or client.max_object_bytes
        if self.max_member_bytes <= 0:
            raise ValueError("max_member_bytes must be positive")

    def iter_members(self) -> Iterator[ArchiveMember]:
        remote_size, central_offset, central_size, entry_count = self._directory_location()
        if central_size == 0:
            if entry_count != 0:
                raise ArchiveFormatError("ZIP has entries but an empty central directory")
            return
        central_end = central_offset + central_size
        if central_offset < 0 or central_end > remote_size:
            raise ArchiveFormatError("ZIP central directory lies outside the remote object")
        directory = self.client.read(self.url, central_offset, central_end - 1)
        cursor = 0
        emitted = 0
        for _ in range(entry_count):
            member, consumed = self._parse_central_member(directory, cursor, central_offset)
            cursor += consumed
            if member.name.endswith("/"):
                continue
            yield member
            emitted += 1
        if cursor != len(directory):
            trailing = directory[cursor:]
            if trailing.startswith(b"PK\x05\x05"):
                raise UnsupportedArchiveError("ZIP digital signatures are not supported")
            raise ArchiveFormatError(
                f"ZIP central directory has {len(trailing)} unparsed trailing bytes"
            )
        if emitted > entry_count:
            raise ArchiveFormatError("ZIP emitted more members than declared")

    def read_member(self, member: ArchiveMember) -> bytes:
        if member.local_header_offset is None or member.crc32 is None:
            raise ValueError("member does not belong to a ZIP reader")
        self._check_member_size("uncompressed", member.uncompressed_size)
        self._check_member_size("compressed", member.compressed_size)
        resolved = self._resolve_data_offset(member)
        if resolved.compressed_size == 0:
            compressed = b""
        else:
            compressed = self.client.read(
                self.url,
                resolved.data_offset,
                resolved.data_offset + resolved.compressed_size - 1,
            )

        if member.compression == "stored":
            result = compressed
        else:
            result = self._inflate_member(member, compressed)
        if len(result) != member.uncompressed_size:
            raise ArchiveFormatError(
                f"ZIP member {member.name!r} expands to {len(result)} bytes; "
                f"expected {member.uncompressed_size}"
            )
        actual_crc = zlib.crc32(result) & 0xFFFFFFFF
        if actual_crc != member.crc32:
            raise ArchiveFormatError(
                f"ZIP member {member.name!r} CRC32 mismatch: "
                f"expected {member.crc32:08x}, got {actual_crc:08x}"
            )
        return result

    def _directory_location(self) -> tuple[int, int, int, int]:
        remote = self.client.inspect(self.url)
        if remote.size < _ZIP_EOCD_BYTES:
            raise ArchiveFormatError("remote object is too small to contain a ZIP EOCD")
        suffix_size = min(remote.size, _ZIP_EOCD_SEARCH_BYTES)
        tail = self.client.read_suffix(self.url, suffix_size)
        eocd_index = tail.rfind(b"PK\x05\x06")
        if eocd_index < 0 or eocd_index + _ZIP_EOCD_BYTES > len(tail):
            raise ArchiveFormatError("ZIP end-of-central-directory record was not found")
        eocd = struct.unpack_from("<4s4H2IH", tail, eocd_index)
        (
            _,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_size,
        ) = eocd
        if eocd_index + _ZIP_EOCD_BYTES + comment_size != len(tail):
            raise ArchiveFormatError("ZIP EOCD comment length does not terminate the object")
        eocd_offset = remote.size - len(tail) + eocd_index

        uses_zip64 = (
            disk_entries == 0xFFFF
            or total_entries == 0xFFFF
            or central_size == 0xFFFFFFFF
            or central_offset == 0xFFFFFFFF
        )
        if uses_zip64:
            return self._zip64_directory_location(remote.size, eocd_offset)
        if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
            raise UnsupportedArchiveError("multi-disk ZIP archives are not supported")
        return remote.size, central_offset, central_size, total_entries

    def _zip64_directory_location(
        self,
        remote_size: int,
        eocd_offset: int,
    ) -> tuple[int, int, int, int]:
        locator_offset = eocd_offset - _ZIP64_LOCATOR_BYTES
        if locator_offset < 0:
            raise ArchiveFormatError("ZIP64 EOCD locator is missing")
        locator = self.client.read(
            self.url,
            locator_offset,
            locator_offset + _ZIP64_LOCATOR_BYTES - 1,
        )
        signature, zip64_disk, zip64_offset, total_disks = struct.unpack("<4sIQI", locator)
        if signature != b"PK\x06\x07":
            raise ArchiveFormatError("ZIP64 EOCD locator has an invalid signature")
        if zip64_disk != 0 or total_disks != 1:
            raise UnsupportedArchiveError("multi-disk ZIP64 archives are not supported")
        if zip64_offset + _ZIP64_EOCD_MIN_BYTES > remote_size:
            raise ArchiveFormatError("ZIP64 EOCD lies outside the remote object")
        record = self.client.read(
            self.url,
            zip64_offset,
            zip64_offset + _ZIP64_EOCD_MIN_BYTES - 1,
        )
        values = struct.unpack("<4sQ2H2I4Q", record)
        (
            signature,
            record_size,
            _,
            _,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
        ) = values
        if signature != b"PK\x06\x06" or record_size < 44:
            raise ArchiveFormatError("ZIP64 EOCD record is invalid")
        if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
            raise UnsupportedArchiveError("multi-disk ZIP64 archives are not supported")
        return remote_size, central_offset, central_size, total_entries

    def _parse_central_member(
        self,
        directory: bytes,
        cursor: int,
        central_offset: int,
    ) -> tuple[ArchiveMember, int]:
        fixed_size = 46
        if cursor + fixed_size > len(directory):
            raise ArchiveFormatError("truncated ZIP central-directory entry")
        values = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
        (
            signature,
            _,
            _,
            flags,
            method,
            _,
            _,
            crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            comment_size,
            disk_start,
            _,
            _,
            local_offset,
        ) = values
        if signature != b"PK\x01\x02":
            raise ArchiveFormatError("invalid ZIP central-directory signature")
        variable_end = cursor + fixed_size + name_size + extra_size + comment_size
        if variable_end > len(directory):
            raise ArchiveFormatError("truncated ZIP central-directory variable fields")
        name_bytes = directory[cursor + fixed_size : cursor + fixed_size + name_size]
        extra_start = cursor + fixed_size + name_size
        extra = directory[extra_start : extra_start + extra_size]
        uncompressed_size, compressed_size, local_offset, disk_start = _resolve_zip64_fields(
            extra,
            uncompressed_size=uncompressed_size,
            compressed_size=compressed_size,
            local_offset=local_offset,
            disk_start=disk_start,
        )
        if flags & 0x1:
            raise UnsupportedArchiveError("encrypted ZIP members are not supported")
        if flags & 0x8:
            raise UnsupportedArchiveError("ZIP members using a data descriptor are not supported")
        if disk_start != 0:
            raise UnsupportedArchiveError("multi-disk ZIP members are not supported")
        if method not in {0, 8}:
            raise UnsupportedArchiveError(f"ZIP compression method {method} is not supported")
        encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            name = name_bytes.decode(encoding)
        except UnicodeDecodeError as error:
            raise ArchiveFormatError("ZIP member name uses invalid encoding") from error
        if local_offset >= central_offset:
            raise ArchiveFormatError(f"ZIP member {name!r} local header overlaps central directory")
        member = ArchiveMember(
            name=name,
            data_offset=-1,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            compression="stored" if method == 0 else "deflate",
            crc32=crc32,
            local_header_offset=local_offset,
            central_directory_offset=central_offset,
        )
        return member, variable_end - cursor

    def _resolve_data_offset(self, member: ArchiveMember) -> ArchiveMember:
        if member.local_header_offset is None or member.central_directory_offset is None:
            raise ValueError("ZIP member lacks remote offset metadata")
        local = self.client.read(
            self.url,
            member.local_header_offset,
            member.local_header_offset + 29,
        )
        values = struct.unpack("<4s5H3I2H", local)
        signature, _, flags, method, _, _, _, _, _, name_size, extra_size = values
        if signature != b"PK\x03\x04":
            raise ArchiveFormatError(f"ZIP member {member.name!r} has an invalid local header")
        if flags & 0x1:
            raise UnsupportedArchiveError("encrypted ZIP members are not supported")
        expected_method = 0 if member.compression == "stored" else 8
        if method != expected_method:
            raise ArchiveFormatError(
                f"ZIP member {member.name!r} compression differs between headers"
            )
        data_offset = member.local_header_offset + 30 + name_size + extra_size
        if data_offset + member.compressed_size > member.central_directory_offset:
            raise ArchiveFormatError(
                f"ZIP member {member.name!r} payload overlaps central directory"
            )
        return ArchiveMember(
            name=member.name,
            data_offset=data_offset,
            compressed_size=member.compressed_size,
            uncompressed_size=member.uncompressed_size,
            compression=member.compression,
            crc32=member.crc32,
            local_header_offset=member.local_header_offset,
            central_directory_offset=member.central_directory_offset,
        )

    def _inflate_member(self, member: ArchiveMember, compressed: bytes) -> bytes:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        output_limit = member.uncompressed_size + 1
        try:
            result = decompressor.decompress(compressed, output_limit)
        except zlib.error as error:
            raise ArchiveFormatError(
                f"ZIP member {member.name!r} has invalid DEFLATE data"
            ) from error
        if len(result) > member.uncompressed_size or decompressor.unconsumed_tail:
            raise ArchiveFormatError(f"ZIP member {member.name!r} expands beyond its declared size")
        try:
            result += decompressor.flush(output_limit - len(result))
        except zlib.error as error:
            raise ArchiveFormatError(
                f"ZIP member {member.name!r} has invalid DEFLATE data"
            ) from error
        if not decompressor.eof or decompressor.unused_data:
            raise ArchiveFormatError(f"ZIP member {member.name!r} has incomplete DEFLATE data")
        return result

    def _check_member_size(self, kind: str, size: int) -> None:
        if size > self.max_member_bytes:
            raise ObjectTooLargeError(
                f"ZIP {kind} member size {size} exceeds max_member_bytes={self.max_member_bytes}"
            )


def _parse_tar_octal(field: bytes, label: str, offset: int) -> int:
    stripped = field.strip(b" \0")
    if not stripped:
        return 0
    if field[0] & 0x80:
        raise UnsupportedArchiveError("base-256 TAR numeric fields are not supported")
    try:
        return int(stripped, 8)
    except ValueError as error:
        raise ArchiveFormatError(f"invalid TAR {label} at offset {offset}") from error


def _parse_tar_name(header: bytes, offset: int) -> str:
    name = header[0:100].split(b"\0", maxsplit=1)[0]
    prefix = header[345:500].split(b"\0", maxsplit=1)[0]
    raw = prefix + (b"/" if prefix and name else b"") + name
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveFormatError(f"invalid UTF-8 TAR name at offset {offset}") from error
    if not decoded:
        raise ArchiveFormatError(f"empty TAR member name at offset {offset}")
    return decoded


def _validate_tar_checksum(header: bytes, offset: int) -> None:
    stored = _parse_tar_octal(header[148:156], "checksum", offset)
    computed = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
    if stored != computed:
        raise ArchiveFormatError(
            f"TAR checksum mismatch at offset {offset}: expected {stored}, got {computed}"
        )


def _resolve_zip64_fields(
    extra: bytes,
    *,
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int,
    disk_start: int,
) -> tuple[int, int, int, int]:
    requires_zip64 = (
        uncompressed_size == 0xFFFFFFFF
        or compressed_size == 0xFFFFFFFF
        or local_offset == 0xFFFFFFFF
        or disk_start == 0xFFFF
    )
    if not requires_zip64:
        return uncompressed_size, compressed_size, local_offset, disk_start

    payload: bytes | None = None
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, field_size = struct.unpack_from("<HH", extra, cursor)
        field_end = cursor + 4 + field_size
        if field_end > len(extra):
            raise ArchiveFormatError("truncated ZIP extra field")
        if header_id == 0x0001:
            payload = extra[cursor + 4 : field_end]
            break
        cursor = field_end
    if payload is None:
        raise ArchiveFormatError("ZIP64 sentinel fields have no ZIP64 extra data")

    cursor = 0

    def read_value(width: int, label: str) -> int:
        nonlocal cursor
        if cursor + width > len(payload):
            raise ArchiveFormatError(f"ZIP64 extra data omits {label}")
        format_code = "<Q" if width == 8 else "<I"
        value = int(struct.unpack_from(format_code, payload, cursor)[0])
        cursor += width
        return value

    if uncompressed_size == 0xFFFFFFFF:
        uncompressed_size = read_value(8, "uncompressed size")
    if compressed_size == 0xFFFFFFFF:
        compressed_size = read_value(8, "compressed size")
    if local_offset == 0xFFFFFFFF:
        local_offset = read_value(8, "local-header offset")
    if disk_start == 0xFFFF:
        disk_start = read_value(4, "disk number")
    return uncompressed_size, compressed_size, local_offset, disk_start
