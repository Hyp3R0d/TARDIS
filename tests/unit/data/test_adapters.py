from __future__ import annotations

import io
import json
import threading
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from tardis.data.adapters import (
    DataVerseAdapter,
    LocalManifestAdapter,
    OpenVidAdapter,
    SeedanceAdapter,
    read_record_media,
)
from tardis.data.contracts import MetadataParseError, VideoRecord
from tardis.data.http_range import RangeClient
from tardis.data.metadata import iter_json_array_records


@contextmanager
def object_server(objects: dict[str, bytes]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version: ClassVar[str] = "HTTP/1.1"

        def _payload(self) -> bytes:
            return objects[self.path.removeprefix("/")]

        def do_HEAD(self) -> None:  # noqa: N802
            payload = self._payload()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            payload = self._payload()
            requested = self.headers["Range"].removeprefix("bytes=")
            if requested.startswith("-"):
                start = max(0, len(payload) - int(requested[1:]))
                end = len(payload) - 1
            else:
                start_text, end_text = requested.split("-", maxsplit=1)
                start = int(start_text)
                end = min(int(end_text), len(payload) - 1)
            body = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def make_zip(name: str, payload: bytes = b"video") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, payload)
    return output.getvalue()


def test_incremental_json_array_parser_handles_nested_values() -> None:
    values = [
        {"video": "0000000000.mp4", "text": "alpha", "nested": {"x": [1, 2]}},
        {"video": "0000100282.mp4", "text": "beta"},
    ]
    payload = json.dumps(values, indent=2).encode()
    with (
        object_server({"metadata.json": payload}) as root,
        RangeClient(max_object_bytes=17) as client,
    ):
        records = list(
            iter_json_array_records(
                client,
                f"{root}/metadata.json",
                source="dataverse",
                id_field="video",
                caption_field="text",
                media_field="video",
                chunk_bytes=17,
            )
        )

    assert [record.id for record in records] == ["0000000000.mp4", "0000100282.mp4"]
    assert records[0].metadata == {"nested": {"x": [1, 2]}}


def test_dataverse_adapter_resolves_video_to_shard_and_pins_revision() -> None:
    payload = json.dumps(
        [
            {"video": "0000000000.mp4", "text": "alpha"},
            {"video": "0000100282.mp4", "text": "beta"},
        ]
    ).encode()
    with (
        object_server({"annotation.json": payload}) as root,
        RangeClient(max_object_bytes=64) as client,
    ):
        adapter = DataVerseAdapter(
            client,
            metadata_url=f"{root}/annotation.json",
            resolve_root=f"{root}/repo",
            revision="revision-a",
            chunk_bytes=64,
        )
        records = list(adapter.iter_records())

    assert records[0].media_locator == (f"tar+{root}/repo/00000/000000.tar#member=./0000000000.mp4")
    assert records[1].media_locator == (f"tar+{root}/repo/00000/000100.tar#member=./0000100282.mp4")
    assert records[1].metadata["revision"] == "revision-a"


def test_dataverse_adapter_emits_only_locally_available_archive_shards() -> None:
    payload = json.dumps(
        [
            {"video": "0000000000.mp4", "text": "available"},
            {"video": "0000100282.mp4", "text": "missing"},
        ]
    ).encode()
    with (
        object_server({"annotation.json": payload}) as root,
        RangeClient(max_object_bytes=64) as client,
    ):
        adapter = DataVerseAdapter(
            client,
            metadata_url=f"{root}/annotation.json",
            resolve_root=f"{root}/repo",
            revision="revision-a",
            chunk_bytes=64,
            available_archive_shards={0},
        )
        records = list(adapter.iter_records())

    assert [record.id for record in records] == ["0000000000.mp4"]
    assert adapter.stats.unmatched_metadata_records == 1


def test_openvid_adapter_joins_csv_to_remote_zip_catalog() -> None:
    csv_payload = (
        b"video,caption,aesthetic score\r\n"
        b'clip-a.mp4,"alpha, scene",7.0\r\n'
        b"clip-b.mp4,beta,8.0\r\n"
        b"missing.mp4,missing,0.0\r\n"
    )
    archive_a = make_zip("OpenVid_part1/clip-a.mp4", b"a")
    archive_b = make_zip("OpenVid_part2/clip-b.mp4", b"b")
    objects = {"metadata.csv": csv_payload, "part0.zip": archive_a, "part1.zip": archive_b}
    with object_server(objects) as root, RangeClient(max_object_bytes=4096) as client:
        adapter = OpenVidAdapter(
            client,
            metadata_url=f"{root}/metadata.csv",
            archive_urls=(f"{root}/part0.zip", f"{root}/part1.zip"),
            revision="revision-b",
            chunk_bytes=29,
            max_index_entries=8,
        )
        records = list(adapter.iter_records())

    assert [record.id for record in records] == ["clip-a.mp4", "clip-b.mp4"]
    assert records[0].media_locator == (f"zip+{root}/part0.zip#member=OpenVid_part1/clip-a.mp4")
    assert records[1].metadata["aesthetic score"] == "8.0"
    assert adapter.stats.unmatched_metadata_records == 1


def test_seedance_adapter_prefers_english_prompt_and_direct_video() -> None:
    rows = [
        {
            "id": "SD2_00001",
            "raw_p": "中文提示",
            "i18n": {"en": {"p": "English prompt"}},
            "media": {"v": "seedance-2/videos/SD2_00001.mp4"},
            "spec": {"duration": 15.0},
        },
        {
            "id": "SD2_00002",
            "raw_p": "fallback prompt",
            "file_name": "seedance-2/videos/SD2_00002.mp4",
        },
    ]
    payload = b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows)
    with (
        object_server({"metadata.jsonl": payload}) as root,
        RangeClient(max_object_bytes=73) as client,
    ):
        adapter = SeedanceAdapter(
            client,
            metadata_url=f"{root}/metadata.jsonl",
            resolve_root=f"{root}/repo",
            revision="revision-c",
            chunk_bytes=73,
        )
        records = list(adapter.iter_records())

    assert [record.caption for record in records] == ["English prompt", "fallback prompt"]
    assert records[0].media_locator == f"{root}/repo/seedance-2/videos/SD2_00001.mp4"
    assert records[0].metadata["duration"] == 15.0
    assert records[1].metadata["revision"] == "revision-c"


def test_seedance_adapter_emits_only_locally_available_media() -> None:
    rows = [
        {
            "id": "SD2_00001",
            "raw_p": "available",
            "file_name": "seedance-2/videos/SD2_00001.mp4",
        },
        {
            "id": "SD2_00002",
            "raw_p": "missing",
            "file_name": "seedance-2/videos/SD2_00002.mp4",
        },
    ]
    payload = b"\n".join(json.dumps(row).encode() for row in rows)
    with (
        object_server({"metadata.jsonl": payload}) as root,
        RangeClient(max_object_bytes=73) as client,
    ):
        adapter = SeedanceAdapter(
            client,
            metadata_url=f"{root}/metadata.jsonl",
            resolve_root=f"{root}/repo",
            revision="revision-c",
            chunk_bytes=73,
            available_media_paths={"seedance-2/videos/SD2_00001.mp4"},
        )
        records = list(adapter.iter_records())

    assert [record.id for record in records] == ["SD2_00001"]
    assert adapter.stats.unmatched_metadata_records == 1


def test_local_manifest_adapter_restores_prefiltered_record(tmp_path: Path) -> None:
    manifest = tmp_path / "tardis_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "clip.mp4",
                "caption": "local prompt",
                "media_locator": (tmp_path / "clip.mp4").as_uri(),
                "source": "seedance",
                "metadata": {"revision": "revision-c", "duration": 2.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with RangeClient(max_object_bytes=1024) as client:
        adapter = LocalManifestAdapter(
            client,
            source="seedance",
            revision="revision-c",
            manifest_url=manifest.as_uri(),
            chunk_bytes=64,
        )
        records = list(adapter.iter_records())

    assert records == [
        VideoRecord(
            id="clip.mp4",
            caption="local prompt",
            media_locator=(tmp_path / "clip.mp4").as_uri(),
            source="seedance",
            metadata={"revision": "revision-c", "duration": 2.0},
        )
    ]


def test_unified_media_reader_fetches_direct_tar_and_zip_locators() -> None:
    tar_output = io.BytesIO()
    import tarfile

    with tarfile.open(fileobj=tar_output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("./clip.mp4")
        info.size = 9
        archive.addfile(info, io.BytesIO(b"tar-video"))
    zip_payload = make_zip("folder/clip.mp4", b"zip-video")
    objects = {
        "direct.mp4": b"direct-video",
        "archive.tar": tar_output.getvalue(),
        "archive.zip": zip_payload,
    }
    with object_server(objects) as root, RangeClient(max_object_bytes=4096) as client:
        records = [
            VideoRecord("direct", "caption", f"{root}/direct.mp4", "fixture"),
            VideoRecord("tar", "caption", f"tar+{root}/archive.tar#member=./clip.mp4", "fixture"),
            VideoRecord(
                "zip",
                "caption",
                f"zip+{root}/archive.zip#member=folder/clip.mp4",
                "fixture",
            ),
        ]
        payloads = [read_record_media(client, record, max_media_bytes=1024) for record in records]

    assert payloads == [b"direct-video", b"tar-video", b"zip-video"]


def test_zip_member_index_is_reused_within_one_worker(tmp_path: Path) -> None:
    archive_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("clips/first.mp4", b"first")
        archive.writestr("clips/second.mp4", b"second")
    records = (
        VideoRecord(
            "first",
            "caption",
            f"zip+{archive_path.as_uri()}#member=clips/first.mp4",
            "fixture",
        ),
        VideoRecord(
            "second",
            "caption",
            f"zip+{archive_path.as_uri()}#member=clips/second.mp4",
            "fixture",
        ),
    )

    with RangeClient(max_object_bytes=4096) as client:
        assert read_record_media(client, records[0], max_media_bytes=1024) == b"first"
        requests_after_first = client.telemetry.snapshot().request_count
        assert read_record_media(client, records[1], max_media_bytes=1024) == b"second"
        second_request_count = client.telemetry.snapshot().request_count - requests_after_first

    assert second_request_count == 2


def test_cached_tar_index_rejects_invalid_resume_offset() -> None:
    import tarfile

    tar_output = io.BytesIO()
    with tarfile.open(fileobj=tar_output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("clip.mp4")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"data"))

    with (
        object_server({"archive.tar": tar_output.getvalue()}) as root,
        RangeClient(max_object_bytes=4096) as client,
    ):
        client.put_archive_index_state(
            f"{root}/archive.tar",
            {"members": {}, "next_offset": "not-an-offset", "complete": False},
        )
        record = VideoRecord(
            "tar",
            "caption",
            f"tar+{root}/archive.tar#member=clip.mp4",
            "fixture",
        )
        with pytest.raises(MetadataParseError, match="non-integer next offset"):
            read_record_media(client, record, max_media_bytes=1024)
