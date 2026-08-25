"""Incremental metadata parsers over bounded HTTP ranges."""

from __future__ import annotations

import codecs
import csv
import json
from collections.abc import Iterator, Mapping
from typing import Any

from tardis.data.contracts import MetadataParseError, MetadataValue, VideoRecord
from tardis.data.http_range import RangeClient


def iter_jsonl_records(
    client: RangeClient,
    url: str,
    *,
    source: str,
    id_field: str,
    caption_field: str,
    media_field: str,
    chunk_bytes: int,
) -> Iterator[VideoRecord]:
    """Parse one JSON object per non-empty physical line."""

    for line_number, value in iter_jsonl_objects(client, url, chunk_bytes=chunk_bytes):
        yield _record_from_mapping(
            value,
            source=source,
            id_field=id_field,
            caption_field=caption_field,
            media_field=media_field,
            location=f"line {line_number}",
        )


def iter_jsonl_objects(
    client: RangeClient,
    url: str,
    *,
    chunk_bytes: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield raw JSON objects with physical line numbers."""

    lines = client.iter_lines(url, chunk_bytes=chunk_bytes)
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MetadataParseError(f"invalid JSONL record at line {line_number}") from error
        if not isinstance(value, dict):
            raise MetadataParseError(f"JSONL line {line_number} is not an object")
        yield line_number, value


def iter_csv_records(
    client: RangeClient,
    url: str,
    *,
    source: str,
    id_field: str,
    caption_field: str,
    media_field: str,
    chunk_bytes: int,
) -> Iterator[VideoRecord]:
    """Parse RFC-compatible CSV rows from a byte-line stream."""

    decoded_lines = (
        raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8") + "\n"
        for line_number, raw_line in enumerate(
            client.iter_lines(url, chunk_bytes=chunk_bytes),
            start=1,
        )
    )
    try:
        reader = csv.DictReader(decoded_lines)
        if reader.fieldnames is None:
            return
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise MetadataParseError(f"CSV row {row_number} has excess columns")
            yield _record_from_mapping(
                row,
                source=source,
                id_field=id_field,
                caption_field=caption_field,
                media_field=media_field,
                location=f"row {row_number}",
            )
    except UnicodeDecodeError as error:
        raise MetadataParseError("CSV metadata is not valid UTF-8") from error
    except csv.Error as error:
        raise MetadataParseError("invalid CSV metadata") from error


def iter_json_array_records(
    client: RangeClient,
    url: str,
    *,
    source: str,
    id_field: str,
    caption_field: str,
    media_field: str,
    chunk_bytes: int,
    max_record_bytes: int = 8 * 1024 * 1024,
) -> Iterator[VideoRecord]:
    """Incrementally parse objects from one top-level JSON array."""

    if max_record_bytes <= 0:
        raise ValueError("max_record_bytes must be positive")
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    cursor = 0
    started = False
    finished = False
    item_index = 0
    needs_separator = False

    chunks = client.iter_chunks(url, chunk_bytes=chunk_bytes)
    for byte_chunk in chunks:
        try:
            buffer += utf8.decode(byte_chunk)
        except UnicodeDecodeError as error:
            raise MetadataParseError("JSON array metadata is not valid UTF-8") from error
        while True:
            cursor = _skip_json_whitespace(buffer, cursor)
            if not started:
                if cursor >= len(buffer):
                    break
                if buffer[cursor] != "[":
                    raise MetadataParseError("JSON metadata root must be an array")
                started = True
                cursor += 1
                continue
            cursor = _skip_json_whitespace(buffer, cursor)
            if cursor >= len(buffer):
                break
            if buffer[cursor] == "]":
                finished = True
                cursor += 1
                break
            if needs_separator:
                if buffer[cursor] != ",":
                    raise MetadataParseError(
                        f"JSON array item {item_index + 1} is missing a separator"
                    )
                cursor = _skip_json_whitespace(buffer, cursor + 1)
                needs_separator = False
                if cursor >= len(buffer):
                    break
                if buffer[cursor] == "]":
                    raise MetadataParseError("JSON array cannot end with a trailing comma")
            try:
                value, end = decoder.raw_decode(buffer, cursor)
            except json.JSONDecodeError:
                break
            if not isinstance(value, dict):
                raise MetadataParseError(f"JSON array item {item_index + 1} is not an object")
            item_index += 1
            yield _record_from_mapping(
                value,
                source=source,
                id_field=id_field,
                caption_field=caption_field,
                media_field=media_field,
                location=f"item {item_index}",
            )
            cursor = end
            needs_separator = True

        if finished:
            trailing = buffer[cursor:]
            if trailing.strip():
                raise MetadataParseError("JSON array has trailing non-whitespace content")
            for remaining in chunks:
                try:
                    trailing += utf8.decode(remaining)
                except UnicodeDecodeError as error:
                    raise MetadataParseError("JSON array metadata is not valid UTF-8") from error
                if trailing.strip():
                    raise MetadataParseError("JSON array has trailing non-whitespace content")
            break

        if cursor > 0:
            buffer = buffer[cursor:]
            cursor = 0
        if len(buffer.encode("utf-8")) > max_record_bytes:
            raise MetadataParseError(
                f"JSON array record exceeds max_record_bytes={max_record_bytes}"
            )

    try:
        buffer += utf8.decode(b"", final=True)
    except UnicodeDecodeError as error:
        raise MetadataParseError("JSON array metadata ends with invalid UTF-8") from error
    if not finished:
        cursor = _skip_json_whitespace(buffer, cursor)
        if cursor < len(buffer) and buffer[cursor] == "]":
            cursor += 1
            finished = True
        if not finished:
            raise MetadataParseError("JSON array metadata is truncated")
    if buffer[cursor:].strip():
        raise MetadataParseError("JSON array has trailing non-whitespace content")


def _record_from_mapping(
    value: Mapping[str, Any],
    *,
    source: str,
    id_field: str,
    caption_field: str,
    media_field: str,
    location: str,
) -> VideoRecord:
    required = (id_field, caption_field, media_field)
    missing = [field for field in required if field not in value or value[field] is None]
    if missing:
        raise MetadataParseError(f"metadata {location} is missing fields: {', '.join(missing)}")

    record_id = str(value[id_field]).strip()
    caption = str(value[caption_field]).strip()
    media_locator = str(value[media_field]).strip()
    if not record_id or not caption or not media_locator:
        raise MetadataParseError(f"metadata {location} contains an empty required field")

    excluded = frozenset(required)
    metadata: dict[str, MetadataValue] = {
        str(key): _metadata_value(item)
        for key, item in value.items()
        if key not in excluded and key is not None
    }
    return VideoRecord(
        id=record_id,
        caption=caption,
        media_locator=media_locator,
        source=source,
        metadata=metadata,
    )


def _metadata_value(value: Any) -> MetadataValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return str(value)


def _skip_json_whitespace(value: str, cursor: int) -> int:
    while cursor < len(value) and value[cursor] in " \t\r\n":
        cursor += 1
    return cursor
