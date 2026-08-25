from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tardis.utils.video_io import (
    VideoEncodingError,
    VideoProbeError,
    probe_video,
    write_mp4,
)


def _video() -> torch.Tensor:
    height, width = 16, 18
    video = torch.empty(4, 3, height, width, dtype=torch.float32)
    video[0].fill_(-1.0)
    video[1].fill_(-0.25)
    video[2].fill_(0.25)
    video[3].fill_(1.0)
    return video


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_write_mp4_streams_normalized_video_and_probe_reports_playable(
    tmp_path: Path, batched: bool
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for the MP4 integration test")
    video = _video().unsqueeze(0) if batched else _video()
    destination = tmp_path / ("batched.mp4" if batched else "unbatched.mp4")

    write_mp4(video, destination, fps=7.0)
    info = probe_video(destination)

    assert destination.is_file() and destination.stat().st_size > 0
    assert info.playable
    assert info.codec_name == "h264"
    assert info.pixel_format == "yuv420p"
    assert (info.width, info.height) == (18, 16)
    assert info.frame_count == 4
    assert info.fps == pytest.approx(7.0, rel=0.01)
    assert {path.name for path in tmp_path.iterdir()} == {destination.name}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("video", "message"),
    [
        (torch.zeros(2, 3, 4, 4, dtype=torch.uint8), "floating point"),
        (torch.zeros(2, 4, 4), "shape"),
        (torch.zeros(2, 4, 3, 4, 4), "batch dimension.*1"),
        (torch.zeros(2, 1, 4, 4), "three channels"),
        (torch.full((2, 3, 4, 4), math.nan), "finite"),
        (torch.full((2, 3, 4, 4), 1.01), r"\[-1, 1\]"),
    ],
)
def test_write_mp4_rejects_invalid_video(video: torch.Tensor, message: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=message):
        write_mp4(video, tmp_path / "invalid.mp4", fps=8)


@pytest.mark.unit
@pytest.mark.parametrize("fps", [0.0, -1.0, math.nan, math.inf])
def test_write_mp4_rejects_invalid_fps(fps: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps.*finite and positive"):
        write_mp4(_video(), tmp_path / "invalid.mp4", fps=fps)


@pytest.mark.unit
def test_write_mp4_surfaces_ffmpeg_failure_and_removes_partial_output(tmp_path: Path) -> None:
    false_binary = shutil.which("false")
    if false_binary is None:
        pytest.skip("false executable is required for this failure-path test")
    destination = tmp_path / "failed.mp4"

    with pytest.raises(VideoEncodingError, match="ffmpeg failed.*exit code"):
        write_mp4(_video(), destination, fps=8, ffmpeg_bin=false_binary)

    assert not destination.exists()


@pytest.mark.unit
def test_write_mp4_reports_missing_ffmpeg_binary(tmp_path: Path) -> None:
    with pytest.raises(VideoEncodingError, match="ffmpeg executable.*not found"):
        write_mp4(
            _video(),
            tmp_path / "missing.mp4",
            fps=8,
            ffmpeg_bin="definitely-not-a-real-ffmpeg-binary",
        )


@pytest.mark.unit
def test_probe_rejects_non_video_with_clear_error(tmp_path: Path) -> None:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for the probe failure-path test")
    invalid = tmp_path / "not-video.mp4"
    invalid.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoProbeError, match="ffprobe failed|no playable video stream"):
        probe_video(invalid)


@pytest.mark.unit
def test_write_mp4_declares_output_muxer_independent_of_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tardis.utils.video_io as video_io_module

    destination = tmp_path / "showcase.artifact"
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        destination.write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(video_io_module.subprocess, "run", fake_run)

    write_mp4(_video(), destination, fps=8)

    assert commands[0][-3:] == ["-f", "mp4", str(destination)]


@pytest.mark.unit
def test_write_mp4_pads_odd_dimensions_for_yuv420p(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for the odd-size integration test")
    destination = tmp_path / "odd.mp4"
    video = torch.zeros(3, 3, 15, 17)

    write_mp4(video, destination, fps=8)
    info = probe_video(destination)

    assert (info.width, info.height) == (18, 16)
    assert {path.name for path in tmp_path.iterdir()} == {destination.name}


@pytest.mark.unit
def test_probe_rejects_truncated_payload_after_successful_header_probe(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the truncation integration test")
    generator = torch.Generator().manual_seed(17)
    video = torch.rand(24, 3, 64, 66, generator=generator).mul_(2).sub_(1)
    valid = tmp_path / "valid.mp4"
    truncated = tmp_path / "truncated.mp4"
    write_mp4(video, valid, fps=8, ffmpeg_bin=ffmpeg)
    payload = valid.read_bytes()
    truncated.write_bytes(payload[: len(payload) * 3 // 4])

    header_probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", str(truncated)],
        capture_output=True,
        check=False,
    )
    assert header_probe.returncode == 0

    with pytest.raises(VideoProbeError, match="decode"):
        probe_video(truncated, ffprobe_bin=ffprobe)
