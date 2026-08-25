from __future__ import annotations

from pathlib import Path

from tardis.data.catalog import (
    DEFAULT_SOURCE_SPECS,
    SourceName,
    SourceSpec,
    build_local_adapters,
    discover_openvid_archives,
    normalize_local_dataset_roots,
    source_root,
)
from tardis.data.http_range import RangeClient


def test_default_specs_are_mirror_pinned_and_cover_three_sources() -> None:
    assert set(DEFAULT_SOURCE_SPECS) == {"dataverse", "openvid", "seedance"}
    for source, spec in DEFAULT_SOURCE_SPECS.items():
        assert spec.source == source
        assert len(spec.revision) == 40
        assert source_root("https://hf-mirror.com", spec).startswith(
            "https://hf-mirror.com/datasets/"
        )


def test_openvid_discovery_filters_media_archives_and_sorts_naturally() -> None:
    payload = [
        {"type": "file", "path": "OpenVidHD/OpenVidHD_part_10.zip"},
        {"type": "file", "path": "OpenVidHD/OpenVidHD_part_2.zip"},
        {"type": "file", "path": "OpenVidHD/OpenVidHD.json"},
        {"type": "directory", "path": "OpenVidHD/other"},
    ]

    archives = discover_openvid_archives(
        "https://hf-mirror.com",
        DEFAULT_SOURCE_SPECS["openvid"],
        fetch_json=lambda _url: payload,
    )

    assert archives == (
        "https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/"
        "d8a63bd22989c80b5734ec2bb989f4e1b61a5807/OpenVidHD/OpenVidHD_part_2.zip",
        "https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/"
        "d8a63bd22989c80b5734ec2bb989f4e1b61a5807/OpenVidHD/OpenVidHD_part_10.zip",
    )


def test_openvid_discovery_can_bound_archives_for_remote_production_runs() -> None:
    payload = [
        {"type": "file", "path": "OpenVidHD/OpenVidHD_part_10.zip"},
        {"type": "file", "path": "OpenVidHD/OpenVidHD_part_2.zip"},
    ]

    archives = discover_openvid_archives(
        "https://hf-mirror.com",
        DEFAULT_SOURCE_SPECS["openvid"],
        max_archives=1,
        fetch_json=lambda _url: payload,
    )

    assert len(archives) == 1
    assert archives[0].endswith("/OpenVidHD/OpenVidHD_part_2.zip")


def test_source_spec_exposes_typed_canonical_names() -> None:
    spec = SourceSpec("other", "owner/repo", "0" * 40, "metadata.jsonl")

    assert isinstance(spec.source, str)
    assert SourceName.DATAVERSE.value == "dataverse"


def test_local_catalog_maps_canonical_directories_and_present_archives(tmp_path: Path) -> None:
    dataverse = tmp_path / "Vchitect_T2V_DataVerse"
    openvid = tmp_path / "OpenVid-1M"
    seedance = tmp_path / "seedance-2-prompts-datasets"
    (dataverse / "00000").mkdir(parents=True)
    (openvid / "data/train").mkdir(parents=True)
    seedance.mkdir()
    (dataverse / "annotation.json").write_text("[]", encoding="utf-8")
    (dataverse / "00000/000003.tar").touch()
    (openvid / "data/train/OpenVid-1M.csv").write_text(
        "video,caption\n",
        encoding="utf-8",
    )
    (openvid / "OpenVid_part100.zip").touch()
    (seedance / "metadata.jsonl").touch()
    (seedance / "seedance-2/videos").mkdir(parents=True)
    (seedance / "seedance-2/videos/SD2_00001.mp4").touch()

    roots = normalize_local_dataset_roots((seedance, dataverse, openvid))
    with RangeClient(max_object_bytes=1024) as client:
        adapters = build_local_adapters(client, dataset_roots=roots)

    assert tuple(sorted(adapters)) == ("dataverse", "openvid", "seedance")
    assert adapters["dataverse"].available_archive_shards == frozenset({3})
    assert adapters["openvid"].archive_urls == ((openvid / "OpenVid_part100.zip").as_uri(),)
    assert adapters["seedance"].available_media_paths == frozenset(
        {"seedance-2/videos/SD2_00001.mp4"}
    )
    assert adapters["seedance"].resolve_root == seedance.as_uri()
