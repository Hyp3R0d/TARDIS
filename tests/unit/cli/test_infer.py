from __future__ import annotations

import pytest


def test_infer_parser_accepts_diskless_remote_catalog_limits() -> None:
    from tardis.cli.infer import parse_args

    args = parse_args(
        [
            "--catalog-record-limit",
            "5",
            "--openvid-archive-limit",
            "1",
            "--dataverse-record-ids",
            "0000000949.mp4,0000000534.mp4",
        ]
    )

    assert args.catalog_record_limit == 5
    assert args.openvid_archive_limit == 1
    assert args.dataverse_record_ids == ("0000000949.mp4", "0000000534.mp4")


def test_infer_defaults_to_five_random_showcases_and_allows_override() -> None:
    from tardis.cli.infer import parse_args

    assert parse_args([]).showcase_count == 5
    assert parse_args(["--showcase-count", "7"]).showcase_count == 7

    with pytest.raises(ValueError, match="showcase_count"):
        parse_args(["--showcase-count", "0"])


def test_infer_selects_exactly_one_dataset() -> None:
    from tardis.cli.infer import parse_args

    assert parse_args([]).dataset == "dataverse"
    assert parse_args(["--dataset", "seedance"]).dataset == "seedance"

    with pytest.raises(SystemExit):
        parse_args(["--dataset", "unknown"])


def test_infer_exposes_checkpoint_mechanism_arguments() -> None:
    from tardis.cli.infer import parse_args

    args = parse_args(
        [
            "--motion-max-flow-pixels",
            "6.0",
            "--transport-max-correction-pixels",
            "0.5",
            "--transport-history-fallback-weight",
            "1.0",
            "--router-threshold",
            "0.2",
            "--router-halo-radius",
            "2",
            "--state-anchor-decay",
            "0.9",
            "--scene-cut-threshold",
            "0.95",
            "--oracle-temperature",
            "0.2",
            "--training-noise-scale",
            "0.0",
            "--lite-max-magnitude",
            "0.25",
            "--keyframe-lite-alignment",
        ]
    )

    assert args.motion_max_flow_pixels == pytest.approx(6.0)
    assert args.transport_max_correction_pixels == pytest.approx(0.5)
    assert args.transport_history_fallback_weight == pytest.approx(1.0)
    assert args.router_threshold == pytest.approx(0.2)
    assert args.router_halo_radius == 2
    assert args.state_anchor_decay == pytest.approx(0.9)
    assert args.scene_cut_threshold == pytest.approx(0.95)
    assert args.oracle_temperature == pytest.approx(0.2)
    assert args.training_noise_scale == pytest.approx(0.0)
    assert args.lite_max_magnitude == pytest.approx(0.25)
    assert args.keyframe_lite_alignment is True


def test_showcase_selection_is_seeded_random_and_unique_within_one_source() -> None:
    from tardis.cli.infer import _select_showcase_records

    records = [
        {
            "source": source,
            "record_id": f"{source}-{index}",
            "status": "completed",
        }
        for source in ("openvid",)
        for index in range(8)
    ]

    first = _select_showcase_records(records, count=5, seed=3407)
    second = _select_showcase_records(list(reversed(records)), count=5, seed=3407)

    assert [(item["source"], item["record_id"]) for item in first] == [
        (item["source"], item["record_id"]) for item in second
    ]
    assert len(first) == len({(item["source"], item["record_id"]) for item in first}) == 5
    assert {item["source"] for item in first} == {"openvid"}
