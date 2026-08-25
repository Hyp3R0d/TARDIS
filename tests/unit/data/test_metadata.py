from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from tardis.data.http_range import RangeClient
from tardis.data.metadata import iter_csv_records, iter_jsonl_records


@contextmanager
def metadata_server(payload: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version: ClassVar[str] = "HTTP/1.1"

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers["Range"]
            start_text, end_text = range_header.removeprefix("bytes=").split("-", maxsplit=1)
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
        yield f"http://{host}:{port}/metadata"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_jsonl_stream_crosses_chunk_boundaries_without_files(tmp_path: object) -> None:
    del tmp_path
    payload = (
        b'{"id":"v1","caption":"alpha","path":"a.mp4"}\r\n'
        b"\n"
        b'{"id":"v2","caption":"beta","path":"b.mp4","score":0.9}'
    )
    with metadata_server(payload) as url, RangeClient(max_object_bytes=11) as client:
        records = list(
            iter_jsonl_records(
                client,
                url,
                source="seedance",
                id_field="id",
                caption_field="caption",
                media_field="path",
                chunk_bytes=11,
            )
        )

    assert [(record.id, record.caption, record.media_locator) for record in records] == [
        ("v1", "alpha", "a.mp4"),
        ("v2", "beta", "b.mp4"),
    ]
    assert records[1].source == "seedance"
    assert records[1].metadata == {"score": 0.9}


def test_csv_stream_handles_quotes_unicode_and_chunk_boundaries(tmp_path: object) -> None:
    del tmp_path
    payload = (
        "video_id,caption,video_path,width\r\n"
        'v1,"a caption, with comma",part0/v1.mp4,512\r\n'
        'v2,"雪中的机器人",part1/v2.mp4,720\r\n'
    ).encode()
    with metadata_server(payload) as url, RangeClient(max_object_bytes=13) as client:
        records = list(
            iter_csv_records(
                client,
                url,
                source="openvid",
                id_field="video_id",
                caption_field="caption",
                media_field="video_path",
                chunk_bytes=13,
            )
        )

    assert [record.id for record in records] == ["v1", "v2"]
    assert records[0].caption == "a caption, with comma"
    assert records[1].caption == "雪中的机器人"
    assert records[1].metadata == {"width": "720"}
