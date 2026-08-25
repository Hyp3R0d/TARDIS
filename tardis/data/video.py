"""Diskless video decoding, temporal sampling, and spatial transforms."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as functional

VideoMode = Literal["train", "benchmark"]
TemporalMode = Literal["window", "uniform"]


class VideoDecodeError(RuntimeError):
    """Raised when an in-memory video cannot be decoded within its bounds."""


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    """Complete decoded RGB frames and source frame rate."""

    frames: torch.Tensor
    fps: float

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError("decoded frames must have shape [T,H,W,3]")
        if self.frames.dtype != torch.uint8:
            raise ValueError("decoded frames must use uint8 RGB")
        if self.frames.shape[0] <= 0:
            raise ValueError("decoded video cannot be empty")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("decoded FPS must be positive and finite")


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    width: int
    height: int
    fps: float
    frame_count: int


def decode_video_bytes(
    payload: bytes,
    *,
    max_decoded_bytes: int,
    timeout_seconds: float = 60.0,
) -> DecodedVideo:
    """Decode one compressed video through FFmpeg pipes without a filesystem path."""

    if not payload:
        raise VideoDecodeError("video payload is empty")
    if max_decoded_bytes <= 0:
        raise ValueError("max_decoded_bytes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    probe = _probe_video(payload, timeout_seconds)
    if probe.frame_count <= 0:
        raise VideoDecodeError("FFprobe could not establish a bounded frame count")
    frame_bytes = probe.width * probe.height * 3
    if frame_bytes <= 0 or frame_bytes > max_decoded_bytes:
        raise VideoDecodeError(
            f"one decoded frame requires {frame_bytes} bytes; max_decoded_bytes={max_decoded_bytes}"
        )
    if probe.frame_count > 0 and probe.frame_count * frame_bytes > max_decoded_bytes:
        raise VideoDecodeError(
            f"declared decoded video requires {probe.frame_count * frame_bytes} bytes; "
            f"max_decoded_bytes={max_decoded_bytes}"
        )

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-frames:v",
        str(probe.frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    stdout = _run_ffmpeg_decode(command, payload, timeout_seconds)
    if len(stdout) > max_decoded_bytes:
        raise VideoDecodeError(
            f"decoded output is {len(stdout)} bytes; max_decoded_bytes={max_decoded_bytes}"
        )
    if len(stdout) % frame_bytes != 0:
        raise VideoDecodeError(
            f"decoded byte count {len(stdout)} is not divisible by frame size {frame_bytes}"
        )
    frame_count = len(stdout) // frame_bytes
    if frame_count <= 0:
        raise VideoDecodeError("FFmpeg produced no video frames")
    if frame_count != probe.frame_count:
        raise VideoDecodeError(
            f"FFmpeg produced {frame_count} frames; FFprobe counted {probe.frame_count}"
        )
    frames = torch.frombuffer(bytearray(stdout), dtype=torch.uint8).reshape(
        frame_count,
        probe.height,
        probe.width,
        3,
    )
    return DecodedVideo(frames=frames, fps=probe.fps)


def decode_sampled_video_bytes(
    payload: bytes,
    *,
    num_frames: int,
    mode: TemporalMode,
    seed: int,
    max_decoded_bytes: int,
    timeout_seconds: float = 60.0,
) -> DecodedVideo:
    """Decode only the unique source frames required by one protocol clip."""

    if not payload:
        raise VideoDecodeError("video payload is empty")
    return _decode_sampled_video(
        payload,
        num_frames=num_frames,
        mode=mode,
        seed=seed,
        max_decoded_bytes=max_decoded_bytes,
        timeout_seconds=timeout_seconds,
    )


def decode_sampled_video_path(
    path: Path,
    *,
    num_frames: int,
    mode: TemporalMode,
    seed: int,
    max_decoded_bytes: int,
    timeout_seconds: float = 60.0,
) -> DecodedVideo:
    """Decode selected frames from a seekable media file without loading it into RAM."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise VideoDecodeError(f"video path is missing or empty: {path}")
    return _decode_sampled_video(
        path,
        num_frames=num_frames,
        mode=mode,
        seed=seed,
        max_decoded_bytes=max_decoded_bytes,
        timeout_seconds=timeout_seconds,
    )


def _decode_sampled_video(
    source: bytes | Path,
    *,
    num_frames: int,
    mode: TemporalMode,
    seed: int,
    max_decoded_bytes: int,
    timeout_seconds: float,
) -> DecodedVideo:
    if max_decoded_bytes <= 0:
        raise ValueError("max_decoded_bytes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    probe = _probe_video(source, timeout_seconds)
    if probe.frame_count <= 0:
        raise VideoDecodeError("FFprobe could not establish a bounded frame count")
    indices = temporal_sample_indices(
        probe.frame_count,
        num_frames,
        mode=mode,
        seed=seed,
    )
    unique_indices = tuple(dict.fromkeys(indices))
    frame_bytes = probe.width * probe.height * 3
    output_bytes = frame_bytes * len(unique_indices)
    if output_bytes > max_decoded_bytes:
        raise VideoDecodeError(
            f"sampled decode requires {output_bytes} bytes; max_decoded_bytes={max_decoded_bytes}"
        )
    unique_frames = _decode_selected_frames(
        source,
        unique_indices,
        height=probe.height,
        width=probe.width,
    )
    position = {source_index: index for index, source_index in enumerate(unique_indices)}
    remap = torch.tensor([position[index] for index in indices], dtype=torch.int64)
    frames = unique_frames.index_select(0, remap)
    return DecodedVideo(frames=frames, fps=probe.fps)


def _decode_selected_frames(
    source: bytes | Path,
    indices: tuple[int, ...],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    try:
        import av
    except ImportError as error:
        raise VideoDecodeError("PyAV is required for seekable in-memory video decoding") from error

    targets = frozenset(indices)
    selected: dict[int, torch.Tensor] = {}
    try:
        with av.open(_av_source(source), mode="r") as container:
            if not container.streams.video:
                raise VideoDecodeError("video payload has no video stream")
            stream = container.streams.video[0]
            path_source = isinstance(source, Path)
            first_target = min(targets)
            if path_source and first_target > 0:
                _seek_stream_to_frame(container, stream, first_target, float(stream.average_rate))
            for decoded_index, frame in enumerate(container.decode(stream)):
                if path_source:
                    frame_time = frame.time
                    if frame_time is None:
                        frame_index = int(frame.pts or 0)
                    else:
                        frame_index = int(round(float(frame_time) * float(stream.average_rate)))
                else:
                    frame_index = decoded_index
                if frame_index not in targets:
                    continue
                array = frame.to_ndarray(format="rgb24")
                if array.shape != (height, width, 3):
                    raise VideoDecodeError(
                        f"decoded frame has shape {array.shape}; expected {(height, width, 3)}"
                    )
                selected[frame_index] = torch.from_numpy(array.copy())
                if len(selected) == len(targets):
                    break
    except av.FFmpegError as error:
        raise VideoDecodeError(f"PyAV decode failed: {error}") from error

    missing = targets - set(selected)
    if missing:
        raise VideoDecodeError(f"PyAV did not decode requested frames: {sorted(missing)}")
    return torch.stack([selected[index] for index in indices])


def _seek_stream_to_frame(container: object, stream: object, frame_index: int, fps: float) -> None:
    if frame_index <= 0:
        return
    if not math.isfinite(fps) or fps <= 0:
        raise VideoDecodeError("cannot seek video with an invalid frame rate")
    time_base = getattr(stream, "time_base", None)
    if time_base is None:
        raise VideoDecodeError("video stream has no time base for seeking")
    timestamp = int((frame_index / fps) / float(time_base))
    try:
        container.seek(timestamp, stream=stream, backward=True)
    except Exception as error:
        raise VideoDecodeError(f"failed to seek video to frame {frame_index}") from error


def temporal_sample_indices(
    frame_count: int,
    num_frames: int,
    *,
    mode: TemporalMode,
    seed: int,
) -> tuple[int, ...]:
    """Return exactly ``num_frames`` source indices."""

    if frame_count <= 0 or num_frames <= 0:
        raise ValueError("frame_count and num_frames must be positive")
    if mode == "uniform" or frame_count < num_frames:
        indices = torch.linspace(0, frame_count - 1, steps=num_frames).round().to(torch.int64)
        return tuple(int(index) for index in indices)
    if mode != "window":
        raise ValueError(f"unknown temporal sampling mode {mode!r}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_start = frame_count - num_frames
    start = int(torch.randint(max_start + 1, (), generator=generator).item())
    return tuple(range(start, start + num_frames))


def prepare_clip(
    decoded: DecodedVideo,
    *,
    num_frames: int,
    height: int,
    width: int,
    mode: VideoMode,
    seed: int,
    random_flip: bool,
) -> torch.Tensor:
    """Sample and transform a video into normalized ``[T,3,H,W]``."""

    if height <= 0 or width <= 0:
        raise ValueError("output height and width must be positive")
    temporal_mode: TemporalMode = "window"
    indices = temporal_sample_indices(
        decoded.frames.shape[0],
        num_frames,
        mode=temporal_mode,
        seed=seed,
    )
    clip = decoded.frames[list(indices)].permute(0, 3, 1, 2).to(torch.float32)
    source_height, source_width = clip.shape[-2:]
    scale = max(height / source_height, width / source_width)
    resized_height = max(height, math.ceil(source_height * scale))
    resized_width = max(width, math.ceil(source_width * scale))
    clip = functional.interpolate(
        clip,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    if mode == "train":
        generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x5EED5EED)
        max_top = resized_height - height
        max_left = resized_width - width
        top = int(torch.randint(max_top + 1, (), generator=generator).item())
        left = int(torch.randint(max_left + 1, (), generator=generator).item())
        should_flip = random_flip and bool(torch.randint(2, (), generator=generator).item())
    elif mode == "benchmark":
        top = (resized_height - height) // 2
        left = (resized_width - width) // 2
        should_flip = False
    else:
        raise ValueError(f"unknown video transform mode {mode!r}")

    clip = clip[:, :, top : top + height, left : left + width]
    if should_flip:
        clip = clip.flip(-1)
    normalized: torch.Tensor = clip.div(127.5).sub(1.0).clamp_(-1.0, 1.0).contiguous()
    return normalized


def _probe_video(payload: bytes | Path, timeout_seconds: float) -> _VideoProbe:
    if timeout_seconds <= 0:
        raise ValueError("probe timeout must be positive")
    try:
        import av
    except ImportError as error:
        raise VideoDecodeError("PyAV is required for seekable in-memory video probing") from error
    try:
        with av.open(_av_source(payload), mode="r") as container:
            if not container.streams.video:
                raise VideoDecodeError("video payload has no video stream")
            stream = container.streams.video[0]
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            fps = float(rate) if rate is not None else 0.0
            frame_count = int(stream.frames)
            if frame_count <= 0:
                frame_count = sum(1 for _frame in container.decode(stream))
    except av.FFmpegError as error:
        raise VideoDecodeError(f"PyAV probe failed: {error}") from error
    if width <= 0 or height <= 0:
        raise VideoDecodeError("PyAV returned invalid video dimensions")
    if not math.isfinite(fps) or fps <= 0:
        raise VideoDecodeError("PyAV returned invalid video frame rate")
    return _VideoProbe(width=width, height=height, fps=fps, frame_count=frame_count)


def _av_source(source: bytes | Path) -> BytesIO | str:
    return BytesIO(source) if isinstance(source, bytes) else str(source)


def _run_ffmpeg_decode(command: list[str], payload: bytes, timeout_seconds: float) -> bytes:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(payload, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise VideoDecodeError("FFmpeg decode timed out") from error
    except OSError as error:
        raise VideoDecodeError("FFmpeg executable is unavailable") from error
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise VideoDecodeError(f"FFmpeg decode failed: {message}")
    return stdout
