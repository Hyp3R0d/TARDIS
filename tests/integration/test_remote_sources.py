from __future__ import annotations

from itertools import islice

import pytest

from tardis.data.adapters import (
    DataVerseAdapter,
    OpenVidAdapter,
    SeedanceAdapter,
    read_record_media,
)
from tardis.data.http_range import RangeClient
from tardis.data.video import decode_sampled_video_bytes, prepare_clip

DATAVERSE_REVISION = "e068be25f4d06a837992a1e9096fd00105c83f2c"
OPENVID_REVISION = "d8a63bd22989c80b5734ec2bb989f4e1b61a5807"
SEEDANCE_REVISION = "515aa5bd59123fb489914ce9cd21419badb08be4"
MIRROR = "https://hf-mirror.com/datasets"


@pytest.mark.integration
@pytest.mark.remote
@pytest.mark.parametrize("source", ["dataverse", "openvid", "seedance"])
def test_one_real_prompt_video_pair_streams_without_disk_residue(
    source: str,
    tmp_path: object,
) -> None:
    from pathlib import Path

    watched = Path(str(tmp_path))
    before = tuple(watched.rglob("*"))
    with RangeClient(
        max_object_bytes=64 * 1024 * 1024,
        timeout_seconds=120,
        max_retries=3,
    ) as client:
        if source == "dataverse":
            root = f"{MIRROR}/Vchitect/Vchitect_T2V_DataVerse/resolve/{DATAVERSE_REVISION}"
            adapter = DataVerseAdapter(
                client,
                metadata_url=f"{root}/annotation.json",
                resolve_root=root,
                revision=DATAVERSE_REVISION,
                chunk_bytes=1024 * 1024,
            )
            record = next(
                record
                for record in islice(adapter.iter_records(), 301)
                if record.id == "0000000300.mp4"
            )
        elif source == "openvid":
            root = f"{MIRROR}/nkp37/OpenVid-1M/resolve/{OPENVID_REVISION}"
            adapter = OpenVidAdapter(
                client,
                metadata_url=f"{root}/data/train/OpenVidHD.csv",
                archive_urls=(f"{root}/OpenVidHD/OpenVidHD_part_8.zip",),
                revision=OPENVID_REVISION,
                chunk_bytes=1024 * 1024,
                max_index_entries=20_000,
            )
            record = next(adapter.iter_records())
            assert record.id == "---_iRTHryQ_13_0to241.mp4"
        else:
            root = f"{MIRROR}/GokuScraper/seedance-2-prompts-datasets/resolve/{SEEDANCE_REVISION}"
            adapter = SeedanceAdapter(
                client,
                metadata_url=f"{root}/metadata.jsonl",
                resolve_root=root,
                revision=SEEDANCE_REVISION,
                chunk_bytes=1024 * 1024,
            )
            record = next(adapter.iter_records())
            assert record.id == "SD2_00001"

        media = read_record_media(client, record, max_media_bytes=64 * 1024 * 1024)
        telemetry = client.telemetry.snapshot()

    after = tuple(watched.rglob("*"))
    assert record.caption.strip()
    assert record.metadata["revision"] in {
        DATAVERSE_REVISION,
        OPENVID_REVISION,
        SEEDANCE_REVISION,
    }
    assert _looks_like_video(media)
    assert telemetry.bytes_received > len(media)
    assert after == before == ()


@pytest.mark.integration
@pytest.mark.remote
def test_remote_media_decodes_to_protocol_clip_without_disk_residue(tmp_path: object) -> None:
    from pathlib import Path

    watched = Path(str(tmp_path))
    root = f"{MIRROR}/GokuScraper/seedance-2-prompts-datasets/resolve/{SEEDANCE_REVISION}"
    with RangeClient(
        max_object_bytes=64 * 1024 * 1024,
        timeout_seconds=120,
        max_retries=3,
    ) as client:
        adapter = SeedanceAdapter(
            client,
            metadata_url=f"{root}/metadata.jsonl",
            resolve_root=root,
            revision=SEEDANCE_REVISION,
            chunk_bytes=1024 * 1024,
        )
        record = next(adapter.iter_records())
        media = read_record_media(client, record, max_media_bytes=64 * 1024 * 1024)
        decoded = decode_sampled_video_bytes(
            media,
            num_frames=16,
            mode="uniform",
            seed=3407,
            max_decoded_bytes=256 * 1024 * 1024,
            timeout_seconds=120,
        )
        clip = prepare_clip(
            decoded,
            num_frames=16,
            height=64,
            width=64,
            mode="benchmark",
            seed=3407,
            random_flip=False,
        )

    assert clip.shape == (16, 3, 64, 64)
    assert clip.isfinite().all()
    assert tuple(watched.rglob("*")) == ()


def _looks_like_video(payload: bytes) -> bool:
    return b"ftyp" in payload[:32] or payload.startswith(b"\x1aE\xdf\xa3")
