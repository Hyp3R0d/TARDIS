from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from tardis.data.video import (
    DecodedVideo,
    VideoDecodeError,
    decode_sampled_video_bytes,
    decode_sampled_video_path,
    decode_video_bytes,
    prepare_clip,
    temporal_sample_indices,
)


def make_mp4(frame_count: int = 20, *, height: int = 24, width: int = 32) -> bytes:
    frames = torch.zeros(frame_count, height, width, 3, dtype=torch.uint8)
    for index in range(frame_count):
        frames[index, :, :, 0] = index * 4
        frames[index, :, :, 1] = torch.arange(width, dtype=torch.uint8)[None, :]
        frames[index, :, :, 2] = torch.arange(height, dtype=torch.uint8)[:, None]
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "10",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv444p",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=frames.numpy().tobytes(),
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_ffmpeg_decodes_mp4_bytes_to_complete_rgb_tensor() -> None:
    payload = make_mp4()

    decoded = decode_video_bytes(payload, max_decoded_bytes=2 * 1024 * 1024)

    assert decoded.frames.shape == (20, 24, 32, 3)
    assert decoded.frames.dtype == torch.uint8
    assert decoded.fps == pytest.approx(10.0)
    means = decoded.frames[:, :, :, 0].float().mean(dim=(1, 2))
    assert torch.all(means[1:] > means[:-1])


def test_decode_rejects_output_above_bound_before_returning_tensor() -> None:
    payload = make_mp4()

    with pytest.raises(VideoDecodeError, match="max_decoded_bytes"):
        decode_video_bytes(payload, max_decoded_bytes=100)


def test_decode_rejects_probe_without_bounded_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from tardis.data import video

    monkeypatch.setattr(
        video,
        "_probe_video",
        lambda payload, timeout: video._VideoProbe(width=32, height=24, fps=10.0, frame_count=0),
    )

    with pytest.raises(VideoDecodeError, match="frame count"):
        decode_video_bytes(b"container", max_decoded_bytes=2 * 1024 * 1024)


def test_uniform_temporal_sampling_includes_both_endpoints_without_truncation() -> None:
    assert temporal_sample_indices(48, 16, mode="uniform", seed=7) == (
        0,
        3,
        6,
        9,
        13,
        16,
        19,
        22,
        25,
        28,
        31,
        34,
        38,
        41,
        44,
        47,
    )


def test_short_video_sampling_repeats_frames_to_exact_protocol_length() -> None:
    indices = temporal_sample_indices(5, 16, mode="uniform", seed=7)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 4
    assert all(left <= right for left, right in zip(indices, indices[1:], strict=False))


def test_train_window_and_spatial_augmentation_are_seed_deterministic() -> None:
    frames = torch.arange(24 * 28 * 36 * 3, dtype=torch.int64)
    frames = frames.remainder(256).to(torch.uint8).reshape(24, 28, 36, 3)
    decoded = DecodedVideo(frames=frames, fps=12.0)

    first = prepare_clip(
        decoded,
        num_frames=16,
        height=16,
        width=16,
        mode="train",
        seed=3407,
        random_flip=True,
    )
    repeated = prepare_clip(
        decoded,
        num_frames=16,
        height=16,
        width=16,
        mode="train",
        seed=3407,
        random_flip=True,
    )
    different = prepare_clip(
        decoded,
        num_frames=16,
        height=16,
        width=16,
        mode="train",
        seed=3408,
        random_flip=True,
    )

    assert first.shape == (16, 3, 16, 16)
    assert first.dtype == torch.float32
    assert first.min() >= -1.0 and first.max() <= 1.0
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_benchmark_transform_uses_a_seeded_window_and_center_crop() -> None:
    frames = torch.zeros(20, 24, 32, 3, dtype=torch.uint8)
    frames[..., 0] = torch.arange(20, dtype=torch.uint8)[:, None, None] * 10
    frames[..., 1] = torch.arange(32, dtype=torch.uint8)[None, None, :]
    frames[..., 2] = torch.arange(24, dtype=torch.uint8)[None, :, None]
    decoded = DecodedVideo(frames=frames, fps=10.0)

    first = prepare_clip(
        decoded,
        num_frames=16,
        height=16,
        width=16,
        mode="benchmark",
        seed=1,
        random_flip=False,
    )
    repeated = prepare_clip(
        decoded,
        num_frames=16,
        height=16,
        width=16,
        mode="benchmark",
        seed=1,
        random_flip=False,
    )
    different_seed = next(
        seed
        for seed in range(2, 100)
        if temporal_sample_indices(20, 16, mode="window", seed=seed)
        != temporal_sample_indices(20, 16, mode="window", seed=1)
    )
    different = prepare_clip(
        decoded,
        num_frames=16,
        height=16,
        width=16,
        mode="benchmark",
        seed=different_seed,
        random_flip=False,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
    assert first.min() >= -1.0 and first.max() <= 1.0


def test_sampled_decoder_outputs_only_requested_protocol_frames() -> None:
    payload = make_mp4(frame_count=48)

    decoded = decode_sampled_video_bytes(
        payload,
        num_frames=16,
        mode="uniform",
        seed=7,
        max_decoded_bytes=16 * 24 * 32 * 3,
    )

    assert decoded.frames.shape == (16, 24, 32, 3)
    means = decoded.frames[:, :, :, 0].float().mean(dim=(1, 2))
    assert torch.all(means[1:] > means[:-1])


def test_sampled_decoder_uses_a_seekable_in_memory_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tardis.data import video

    payload = make_mp4(frame_count=12)
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sampled probe must not use a non-seekable FFprobe pipe")
        ),
    )
    monkeypatch.setattr(
        video,
        "_run_ffmpeg_decode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sampled decode must not use a non-seekable FFmpeg pipe")
        ),
    )

    decoded = decode_sampled_video_bytes(
        payload,
        num_frames=3,
        mode="uniform",
        seed=7,
        max_decoded_bytes=3 * 24 * 32 * 3,
    )

    assert decoded.frames.shape == (3, 24, 32, 3)


def test_sampled_decoder_reconstructs_repeated_indices_for_short_video() -> None:
    payload = make_mp4(frame_count=5)

    decoded = decode_sampled_video_bytes(
        payload,
        num_frames=16,
        mode="uniform",
        seed=7,
        max_decoded_bytes=5 * 24 * 32 * 3,
    )

    assert decoded.frames.shape == (16, 24, 32, 3)
    assert torch.equal(decoded.frames[0], decoded.frames[1])
    assert torch.equal(decoded.frames[-1], decoded.frames[-2])


def test_sampled_path_decoder_seeks_to_late_window_without_changing_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tardis.data import video

    payload = make_mp4(frame_count=48)
    path = tmp_path / "long.mp4"
    path.write_bytes(payload)
    seed = next(
        candidate
        for candidate in range(1000)
        if temporal_sample_indices(48, 8, mode="window", seed=candidate)[0] >= 24
    )
    first_index = temporal_sample_indices(48, 8, mode="window", seed=seed)[0]
    calls: list[int] = []
    original = video._seek_stream_to_frame

    def tracked_seek(container: object, stream: object, frame_index: int, fps: float) -> None:
        calls.append(frame_index)
        original(container, stream, frame_index, fps)

    monkeypatch.setattr(video, "_seek_stream_to_frame", tracked_seek)

    actual = decode_sampled_video_path(
        path,
        num_frames=8,
        mode="window",
        seed=seed,
        max_decoded_bytes=8 * 24 * 32 * 3,
    )
    expected = decode_sampled_video_bytes(
        payload,
        num_frames=8,
        mode="window",
        seed=seed,
        max_decoded_bytes=8 * 24 * 32 * 3,
    )

    assert calls == [first_index]
    assert torch.equal(actual.frames, expected.frames)
