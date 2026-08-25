from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

from tardis.data.catalog import (
    DATAVERSE_REVISION,
    LOCAL_MANIFEST_NAME,
    OPENVID_REVISION,
    SEEDANCE_REVISION,
    build_local_adapters,
)
from tardis.data.http_range import RangeClient
from tardis.data.prepare_local import prepare_local_manifests


def _write_tar(path: Path, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _write_zip(path: Path, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, payload)


def test_prepare_local_manifests_filters_downloaded_media_and_enables_fast_path(
    tmp_path: Path,
) -> None:
    dataverse = tmp_path / "Vchitect_T2V_DataVerse"
    openvid = tmp_path / "OpenVid-1M"
    seedance = tmp_path / "seedance-2-prompts-datasets"
    dataverse.mkdir()
    openvid.mkdir()
    seedance.mkdir()

    (dataverse / "annotation.json").write_text(
        json.dumps(
            [
                {"video": "0000003001.mp4", "text": "local DataVerse clip"},
                {"video": "0000100001.mp4", "text": "not downloaded"},
            ]
        ),
        encoding="utf-8",
    )
    _write_tar(dataverse / "00000/000003.tar", "./0000003001.mp4", b"dataverse")

    (openvid / "data/train").mkdir(parents=True)
    (openvid / "data/train/OpenVid-1M.csv").write_text(
        "video,caption\nclip.mp4,local OpenVid clip\nmissing.mp4,not downloaded\n",
        encoding="utf-8",
    )
    _write_zip(openvid / "OpenVid_part100.zip", "OpenVid_part100/clip.mp4", b"openvid")

    (seedance / "seedance-2/videos").mkdir(parents=True)
    (seedance / "seedance-2/videos/SD2_00001.mp4").write_bytes(b"seedance")
    (seedance / "metadata.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "id": "SD2_00001",
                        "raw_p": "local Seedance clip",
                        "file_name": "seedance-2/videos/SD2_00001.mp4",
                    }
                ),
                json.dumps(
                    {
                        "id": "SD2_00002",
                        "raw_p": "missing Seedance clip",
                        "file_name": "seedance-2/videos/SD2_00002.mp4",
                    }
                ),
                "",
            )
        ),
        encoding="utf-8",
    )

    roots = {"dataverse": dataverse, "openvid": openvid, "seedance": seedance}
    counts = prepare_local_manifests(roots, chunk_bytes=64)

    assert counts == {"dataverse": 1, "openvid": 1, "seedance": 1}
    assert all((root / LOCAL_MANIFEST_NAME).is_file() for root in roots.values())

    with RangeClient(max_object_bytes=4096) as client:
        adapters = build_local_adapters(client, dataset_roots=roots, chunk_bytes=64)
        records = {source: list(adapter.iter_records()) for source, adapter in adapters.items()}

    assert records["dataverse"][0].metadata["revision"] == DATAVERSE_REVISION
    assert records["openvid"][0].metadata["revision"] == OPENVID_REVISION
    assert records["seedance"][0].metadata["revision"] == SEEDANCE_REVISION
    assert records["dataverse"][0].media_locator.startswith("tar+file://")
    assert records["openvid"][0].media_locator.startswith("zip+file://")
    assert records["seedance"][0].media_locator.startswith("file://")
