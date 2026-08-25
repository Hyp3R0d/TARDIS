from __future__ import annotations

import importlib
import shutil
import subprocess
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest
import torch

from tardis.data.contracts import VideoRecord
from tardis.data.dataset import ClipDecodeOptions
from tardis.data.splits import StablePartition

SOURCES = ("dataverse", "openvid", "seedance")


def _assembly() -> ModuleType:
    return importlib.import_module("tardis.data.assembly")


class _Adapter:
    revision = "2" * 40

    def __init__(self, source: str, records: Sequence[VideoRecord]) -> None:
        self.source = source
        self._records = tuple(records)

    def iter_records(self) -> Iterator[VideoRecord]:
        yield from self._records


def _make_mp4() -> bytes:
    frames = torch.arange(6 * 16 * 18 * 3, dtype=torch.int64)
    frames = frames.remainder(256).to(torch.uint8).reshape(6, 16, 18, 3)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "18x16",
        "-r",
        "6",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "pipe:1",
    ]
    return subprocess.run(
        command,
        input=frames.numpy().tobytes(),
        capture_output=True,
        check=True,
    ).stdout


@contextmanager
def _range_server(payload: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version: ClassVar[str] = "HTTP/1.1"

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            requested = self.headers["Range"].removeprefix("bytes=")
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
        yield f"http://{host}:{port}/clip.mp4"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.integration
def test_production_loader_fetches_and_decodes_in_ram_without_disk_residue(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for the data assembly integration test")
    module = _assembly()
    payload = _make_mp4()
    before = tuple(tmp_path.rglob("*"))

    with _range_server(payload) as media_url:
        records = {
            source: tuple(
                VideoRecord(
                    id=f"{source}-{index}",
                    caption=f"prompt {source} {index}",
                    media_locator=media_url,
                    source=source,
                    metadata={"revision": "2" * 40},
                )
                for index in range(3)
            )
            for source in SOURCES
        }

        def adapters(_client: object) -> dict[str, _Adapter]:
            return {source: _Adapter(source, records[source]) for source in SOURCES}

        clip_options = ClipDecodeOptions(
            num_frames=4,
            height=12,
            width=12,
            mode="train",
            max_media_bytes=2 * 1024 * 1024,
            max_decoded_bytes=2 * 1024 * 1024,
            timeout_seconds=30,
            random_flip=False,
        )
        benchmark_options = ClipDecodeOptions(
            num_frames=4,
            height=12,
            width=12,
            mode="benchmark",
            max_media_bytes=2 * 1024 * 1024,
            max_decoded_bytes=2 * 1024 * 1024,
            timeout_seconds=30,
            random_flip=False,
        )
        bundle = module.build_remote_dataloaders(
            partition=StablePartition(seed=7, validation_size=1, test_size=1),
            train_clip_options=clip_options,
            evaluation_clip_options=benchmark_options,
            loader_options=module.RemoteDataLoaderOptions(
                steps_per_epoch=1,
                global_batch_size=3,
                evaluation_batch_size=1,
                num_workers=0,
                pin_memory=False,
            ),
            client_factory=module.RangeClientFactory(
                max_object_bytes=2 * 1024 * 1024,
                timeout_seconds=30,
                max_retries=1,
            ),
            adapter_builder=adapters,
        )

        train_batch = next(iter(bundle.train))
        test_batches = [next(iter(bundle.test[source])) for source in SOURCES]

    assert train_batch.video.shape == (3, 4, 3, 12, 12)
    assert set(train_batch.sources) == set(SOURCES)
    assert all(batch.video.shape == (1, 4, 3, 12, 12) for batch in test_batches)
    assert tuple(tmp_path.rglob("*")) == before == ()
