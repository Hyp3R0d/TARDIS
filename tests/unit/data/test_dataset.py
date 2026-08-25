from __future__ import annotations

import tarfile

import torch

from tardis.data.contracts import VideoRecord
from tardis.data.dataset import (
    ClipDecodeOptions,
    ClipExample,
    RemoteClipIterableDataset,
    RemoteClipLoader,
    RemoteSourceClipIterableDataset,
    build_split_records,
    collate_clip_examples,
)
from tardis.data.sampler import BalancedVirtualEpoch, SampleRequest
from tardis.data.splits import StablePartition


def _record(source: str, index: int) -> VideoRecord:
    return VideoRecord(
        id=f"{source}-{index}",
        caption=f"prompt {source} {index}",
        media_locator=f"memory://{source}/{index}",
        source=source,
        metadata={"revision": "0" * 40},
    )


def test_split_records_are_exact_and_source_local() -> None:
    records = {
        source: tuple(_record(source, index) for index in range(6))
        for source in ("dataverse", "openvid", "seedance")
    }

    splits = build_split_records(
        records,
        StablePartition(seed=3407, validation_size=2, test_size=2),
    )

    assert all(
        len(splits[split][source]) == 2 for split in ("validation", "test") for source in records
    )
    assert all(len(splits["train"][source]) == 2 for source in records)
    train_ids = {record.id for record in splits["train"]["dataverse"]}
    validation_ids = {record.id for record in splits["validation"]["dataverse"]}
    assert train_ids.isdisjoint(validation_ids)


class _Client:
    def close(self) -> None:
        pass


def test_remote_clip_dataset_retries_with_deterministic_replacement() -> None:
    records = {"dataverse": (_record("dataverse", 0), _record("dataverse", 1))}
    schedule = BalancedVirtualEpoch(
        {"dataverse": 2},
        steps_per_epoch=1,
        global_batch_size=1,
        seed=3407,
    )
    attempts: list[int] = []

    def loader(_client: object, record: VideoRecord, request: object) -> torch.Tensor:
        del record
        source_index = request.source_index
        attempts.append(source_index)
        if len(attempts) == 1:
            raise RuntimeError("transient decode failure")
        return torch.zeros(4, 3, 8, 8)

    dataset = RemoteClipIterableDataset(
        records,
        schedule=schedule,
        epoch=0,
        rank=0,
        world_size=1,
        client_factory=_Client,
        clip_loader=loader,
        max_retries=2,
    )

    examples = list(iter(dataset))

    assert len(examples) == 1
    assert examples[0].video.shape == (4, 3, 8, 8)
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]


def test_training_dataset_seeks_to_resume_batch_without_decoding_prior_samples() -> None:
    records = {"dataverse": tuple(_record("dataverse", index) for index in range(8))}
    from tardis.data.sampler import FullTrainingEpoch

    schedule = FullTrainingEpoch(
        {"dataverse": 8},
        global_batch_size=2,
        gradient_accumulation_steps=1,
        seed=3407,
    )
    loaded_positions: list[int] = []

    def loader(_client: object, _record: VideoRecord, request: object) -> torch.Tensor:
        loaded_positions.append(int(request.global_position))
        return torch.zeros(4, 3, 8, 8)

    dataset = RemoteClipIterableDataset(
        records,
        schedule=schedule,
        epoch=0,
        rank=0,
        world_size=1,
        client_factory=_Client,
        clip_loader=loader,
    )
    dataset.set_start_batch(2)

    examples = list(dataset)

    assert len(examples) == 4
    assert loaded_positions == [4, 5, 6, 7]


def test_clip_collate_keeps_prompt_and_source_metadata() -> None:
    examples = [
        ClipExample(_record("dataverse", 0), torch.zeros(4, 3, 8, 8), 11),
        ClipExample(_record("openvid", 0), torch.ones(4, 3, 8, 8), 12),
    ]

    batch = collate_clip_examples(examples)

    assert batch.prompts == ["prompt dataverse 0", "prompt openvid 0"]
    assert batch.video.shape == (2, 4, 3, 8, 8)
    assert batch.sources == ("dataverse", "openvid")


def test_dataset_epoch_and_rank_geometry_is_reproducible() -> None:
    records = {"dataverse": tuple(_record("dataverse", i) for i in range(8))}
    schedule = BalancedVirtualEpoch(
        {"dataverse": 8},
        steps_per_epoch=3,
        global_batch_size=4,
        seed=17,
    )

    def loader(_client: object, _record: VideoRecord, request: object) -> torch.Tensor:
        value = float(request.global_position) / 100
        return torch.full((2, 3, 4, 4), value)

    first = RemoteClipIterableDataset(
        records,
        schedule=schedule,
        epoch=2,
        rank=0,
        world_size=2,
        client_factory=_Client,
        clip_loader=loader,
    )
    second = RemoteClipIterableDataset(
        records,
        schedule=schedule,
        epoch=2,
        rank=0,
        world_size=2,
        client_factory=_Client,
        clip_loader=loader,
    )

    assert [item.sample_seed for item in first] == [item.sample_seed for item in second]


def test_benchmark_dataset_excludes_completed_records_before_media_load() -> None:
    records = tuple(_record("dataverse", index) for index in range(3))
    loaded: list[str] = []

    def loader(_client: object, record: VideoRecord, _request: object) -> torch.Tensor:
        loaded.append(record.id)
        return torch.zeros(2, 3, 4, 4)

    dataset = RemoteSourceClipIterableDataset(
        records,
        source="dataverse",
        split="test",
        seed=3407,
        rank=0,
        world_size=1,
        client_factory=_Client,
        clip_loader=loader,
    )
    dataset.exclude_record_ids({"dataverse-0", "dataverse-2"})

    examples = list(dataset)

    assert [example.record.id for example in examples] == ["dataverse-1"]
    assert loaded == ["dataverse-1"]


def test_benchmark_exclusion_preserves_original_rank_ownership() -> None:
    records = tuple(_record("dataverse", index) for index in range(6))

    def loader(_client: object, _record: VideoRecord, _request: object) -> torch.Tensor:
        return torch.zeros(2, 3, 4, 4)

    dataset = RemoteSourceClipIterableDataset(
        records,
        source="dataverse",
        split="test",
        seed=3407,
        rank=1,
        world_size=2,
        client_factory=_Client,
        clip_loader=loader,
    )
    dataset.exclude_record_ids({"dataverse-1"})

    assert [example.record.id for example in dataset] == ["dataverse-3", "dataverse-5"]
    assert len(dataset) == 2


def test_benchmark_loader_decodes_a_seeded_contiguous_window(monkeypatch) -> None:
    import tardis.data.dataset as dataset_module
    from tardis.data.video import DecodedVideo

    observed_modes: list[str] = []
    monkeypatch.setattr(dataset_module, "read_record_media", lambda *_args, **_kwargs: b"video")

    def decode(_payload: bytes, **kwargs: object) -> DecodedVideo:
        observed_modes.append(str(kwargs["mode"]))
        return DecodedVideo(torch.zeros(4, 2, 2, 3, dtype=torch.uint8), fps=30.0)

    monkeypatch.setattr(dataset_module, "decode_sampled_video_bytes", decode)
    loader = RemoteClipLoader(
        ClipDecodeOptions(
            num_frames=4,
            height=2,
            width=2,
            mode="benchmark",
            random_flip=False,
        )
    )

    clip = loader(
        _Client(),
        _record("dataverse", 0),
        SampleRequest("dataverse", 0, 3407, 0, 0),
    )

    assert clip.shape == (4, 3, 2, 2)
    assert observed_modes == ["window"]


def test_benchmark_loader_streams_oversized_local_tar_member_without_residue(
    tmp_path,
) -> None:
    from tardis.data.http_range import RangeClient
    from tardis.utils.video_io import write_mp4

    video_path = tmp_path / "source.mp4"
    write_mp4(torch.zeros(1, 8, 3, 32, 32), video_path, fps=8.0)
    assert video_path.stat().st_size > 512
    archive_path = tmp_path / "archive.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(video_path, arcname="./clip.mp4")
    record = VideoRecord(
        id="clip.mp4",
        caption="a local benchmark clip",
        media_locator=f"tar+{archive_path.as_uri()}#member=./clip.mp4",
        source="dataverse",
    )
    loader = RemoteClipLoader(
        ClipDecodeOptions(
            num_frames=4,
            height=16,
            width=16,
            mode="benchmark",
            max_media_bytes=512,
            random_flip=False,
        )
    )

    with RangeClient(max_object_bytes=4096) as client:
        clip = loader(
            client,
            record,
            SampleRequest("dataverse", 0, 3407, 0, 0),
        )

    assert clip.shape == (4, 3, 16, 16)
    assert not list(tmp_path.glob(".tardis-media-*"))


def test_benchmark_loader_decodes_oversized_direct_local_video_by_path(
    tmp_path,
) -> None:
    from tardis.data.http_range import RangeClient
    from tardis.utils.video_io import write_mp4

    video_path = tmp_path / "direct.mp4"
    write_mp4(torch.zeros(1, 8, 3, 32, 32), video_path, fps=8.0)
    assert video_path.stat().st_size > 512
    record = VideoRecord(
        id="direct.mp4",
        caption="an oversized direct local benchmark clip",
        media_locator=video_path.as_uri(),
        source="seedance",
    )
    loader = RemoteClipLoader(
        ClipDecodeOptions(
            num_frames=4,
            height=16,
            width=16,
            mode="benchmark",
            max_media_bytes=512,
            random_flip=False,
        )
    )

    with RangeClient(max_object_bytes=4096) as client:
        clip = loader(
            client,
            record,
            SampleRequest("seedance", 0, 3407, 0, 0),
        )

    assert clip.shape == (4, 3, 16, 16)
