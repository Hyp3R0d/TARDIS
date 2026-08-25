"""MP4 encoding and probing without intermediate frame or media files."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch


class VideoEncodingError(RuntimeError):
    """Raised when ffmpeg cannot encode a requested video."""


class VideoProbeError(RuntimeError):
    """Raised when ffprobe cannot identify a playable video stream."""


@dataclass(frozen=True, slots=True)
class VideoProbeInfo:
    """Lightweight playable properties reported by ffprobe."""

    playable: bool
    codec_name: str
    pixel_format: str
    width: int
    height: int
    frame_count: int
    fps: float
    duration_seconds: float


def write_mp4(
    video: torch.Tensor,
    destination: Path,
    *,
    fps: float,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Encode a normalized video through an ffmpeg stdin pipe as H.264 yuv420p."""

    frames = _validate_video(video)
    fps_value = _validate_fps(fps)
    frame_count, _, height, width = frames.shape
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    uint8_frames = (
        frames.detach()
        .to(device="cpu", dtype=torch.float32)
        .add(1.0)
        .mul(127.5)
        .round()
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps_value:.12g}",
        "-i",
        "pipe:0",
        "-frames:v",
        str(frame_count),
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(destination),
    ]

    try:
        process = subprocess.run(
            command,
            input=uint8_frames.numpy().tobytes(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as error:
        raise VideoEncodingError(f"ffmpeg executable {ffmpeg_bin!r} was not found") from error

    if process.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise VideoEncodingError(f"ffmpeg failed with exit code {process.returncode}{suffix}")
    if not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise VideoEncodingError("ffmpeg completed without producing a non-empty MP4")


def probe_video(
    path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> VideoProbeInfo:
    """Return structured properties for the first playable video stream."""

    path = Path(path)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as error:
        raise VideoProbeError(f"ffprobe executable {ffprobe_bin!r} was not found") from error

    if process.returncode != 0:
        detail = process.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise VideoProbeError(f"ffprobe failed with exit code {process.returncode}{suffix}")
    try:
        payload = json.loads(process.stdout)
        stream = _first_stream(payload)
        codec_name = str(stream["codec_name"])
        pixel_format = str(stream["pix_fmt"])
        width = int(stream["width"])
        height = int(stream["height"])
        fps = _parse_fps(stream["avg_frame_rate"])
        duration = _parse_duration(stream, payload)
        frame_count = _parse_frame_count(stream, duration=duration, fps=fps)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VideoProbeError("ffprobe returned no playable video stream") from error

    playable = (
        bool(codec_name)
        and bool(pixel_format)
        and width > 0
        and height > 0
        and frame_count > 0
        and math.isfinite(fps)
        and fps > 0
    )
    if not playable:
        raise VideoProbeError("ffprobe returned no playable video stream")
    _verify_decodable_payload(path, ffmpeg_bin=ffmpeg_bin)
    return VideoProbeInfo(
        playable=True,
        codec_name=codec_name,
        pixel_format=pixel_format,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        duration_seconds=duration,
    )


def _validate_video(video: torch.Tensor) -> torch.Tensor:
    if not isinstance(video, torch.Tensor):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("video batch dimension must be 1")
        video = video[0]
    elif video.ndim != 4:
        raise ValueError("video must have shape [T,3,H,W] or [1,T,3,H,W]")
    if video.shape[1] != 3:
        raise ValueError("video must have three channels")
    if video.shape[0] <= 0 or video.shape[2] <= 0 or video.shape[3] <= 0:
        raise ValueError("video must have positive frame and spatial dimensions")
    if not video.is_floating_point():
        raise ValueError("video must use floating point values")
    if not bool(torch.isfinite(video).all().item()):
        raise ValueError("video must contain only finite values")
    minimum = float(video.detach().min().item())
    maximum = float(video.detach().max().item())
    if minimum < -1.0 or maximum > 1.0:
        raise ValueError("video values must be in [-1, 1]")
    return video


def _validate_fps(fps: float) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError) as error:
        raise ValueError("fps must be finite and positive") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be finite and positive")
    return value


def _verify_decodable_payload(path: Path, *, ffmpeg_bin: str) -> None:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as error:
        raise VideoProbeError(f"ffmpeg executable {ffmpeg_bin!r} was not found") from error
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise VideoProbeError(
            f"ffmpeg full decode failed with exit code {process.returncode}{suffix}"
        )


def _first_stream(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("ffprobe payload must be an object")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError("ffprobe payload has no video stream")
    return streams[0]


def _parse_fps(value: object) -> float:
    fps = float(Fraction(str(value)))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid frame rate")
    return fps


def _parse_duration(stream: dict[str, Any], payload: Any) -> float:
    raw_duration = stream.get("duration")
    if raw_duration in (None, "N/A") and isinstance(payload, dict):
        format_data = payload.get("format")
        if isinstance(format_data, dict):
            raw_duration = format_data.get("duration")
    duration = float(str(raw_duration))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    return duration


def _parse_frame_count(stream: dict[str, Any], *, duration: float, fps: float) -> int:
    raw_count = stream.get("nb_frames")
    frame_count = (
        int(str(raw_count)) if raw_count not in (None, "N/A") else int(round(duration * fps))
    )
    if frame_count <= 0:
        raise ValueError("invalid frame count")
    return frame_count
