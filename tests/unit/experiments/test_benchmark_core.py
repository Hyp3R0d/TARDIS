from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tardis.experiments.benchmark import (
    append_jsonl,
    load_jsonl,
    normalize_video,
    parse_args,
    primary_metrics,
    summarize_latencies,
)
from tardis.experiments.selection import choose_record_ids


class MeanAbsoluteFeature:
    provenance_id = "test/mean-absolute"

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3))


def test_normalize_video_accepts_batched_channel_last_zero_one() -> None:
    raw = torch.rand(1, 4, 8, 8, 3)

    result = normalize_video(
        raw,
        num_frames=4,
        height=8,
        width=8,
        input_range="zero_one",
    )

    assert result.shape == (4, 3, 8, 8)
    assert result.min() >= -1
    assert result.max() <= 1
    assert torch.allclose(result, raw[0].permute(0, 3, 1, 2).mul(2).sub(1))


def test_normalize_video_rejects_wrong_protocol_shape() -> None:
    with pytest.raises(ValueError, match="protocol shape"):
        normalize_video(
            torch.zeros(3, 3, 8, 8),
            num_frames=4,
            height=8,
            width=8,
            input_range="minus_one_one",
        )


def test_primary_metrics_match_official_tc_and_framewise_lpips() -> None:
    reference = torch.zeros(3, 3, 2, 2)
    generated = reference.clone()
    generated[1:] = 0.5

    metrics = primary_metrics(generated, reference, MeanAbsoluteFeature())

    assert metrics["tc"] == pytest.approx(0.25)
    assert metrics["lpips"] == pytest.approx(1 / 3)


def test_jsonl_round_trip_and_duplicate_rejection(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"record_id": "a", "tc": 0.1})
    append_jsonl(path, {"record_id": "b", "tc": 0.2})

    assert [item["record_id"] for item in load_jsonl(path)] == ["a", "b"]

    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"record_id": "a", "tc": 0.1},
                {"record_id": "a", "tc": 0.2},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(path)


def test_latency_summary_reports_video_and_frame_statistics() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0], num_frames=16)

    assert summary["video_count"] == 4
    assert summary["mean_video_seconds"] == pytest.approx(2.5)
    assert summary["p50_video_seconds"] == pytest.approx(2.5)
    assert summary["p95_video_seconds"] == pytest.approx(3.85)
    assert summary["mean_frame_milliseconds"] == pytest.approx(156.25)


def test_benchmark_accepts_opt_in_temporal_diagnostics() -> None:
    args = parse_args(
        [
            "--method",
            "sd_turbo_independent",
            "--dataset",
            "dataverse",
            "--output",
            "/tmp/benchmark",
            "--protocol",
            "source50_diagnostics",
            "--temporal-diagnostics",
        ]
    )

    assert args.temporal_diagnostics is True


def test_benchmark_accepts_validation_split_for_selection_runs() -> None:
    args = parse_args(
        [
            "--method",
            "tardis",
            "--dataset",
            "seedance",
            "--output",
            "/tmp/benchmark",
            "--protocol",
            "source_pilot",
            "--data-split",
            "validation",
            "--limit",
            "10",
        ]
    )

    assert args.data_split == "validation"
    assert args.protocol == "source_pilot"


def test_record_selection_is_stable_and_independent_of_catalog_order() -> None:
    records = [SimpleNamespace(id=value) for value in ("record-c", "record-a", "record-b")]

    first = choose_record_ids(records, dataset="seedance", split="validation", seed=3407, count=2)
    second = choose_record_ids(
        list(reversed(records)), dataset="seedance", split="validation", seed=3407, count=2
    )

    assert first == second
    assert len(first) == 2
    assert set(first).issubset({"record-a", "record-b", "record-c"})
