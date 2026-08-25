from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tardis.cli.common import parse_args
from tardis.utils.distributed import DistributedContext
from tardis.utils.manifest import create_run_paths, write_json_manifest
from tardis.utils.random import effective_seed, seed_everything
from tardis.utils.resources import ResourceMonitor, ResourceSnapshot


@pytest.mark.unit
def test_common_defaults_are_production_values() -> None:
    args = parse_args([])

    assert args.seed == 3407
    assert args.height == args.width == 512
    assert args.num_frames == 16
    assert args.fps == 30
    assert args.epochs == 20
    assert args.steps_per_epoch == 64
    assert args.validation_interval == 1
    assert args.validation_batch_size == 8
    assert args.showcase_count == 5
    assert args.mirror_endpoint == "https://hf-mirror.com"


@pytest.mark.unit
def test_core_defaults_can_be_overridden() -> None:
    args = parse_args(
        [
            "--seed",
            "17",
            "--height",
            "256",
            "--num-frames",
            "8",
            "--epochs",
            "3",
            "--active-ratio",
            "0.5",
        ]
    )

    assert (args.seed, args.height, args.num_frames, args.epochs) == (17, 256, 8, 3)
    assert args.active_ratio == pytest.approx(0.5)


@pytest.mark.unit
def test_rank_aware_seeds_are_repeatable_and_distinct() -> None:
    assert effective_seed(3407, rank=0) == effective_seed(3407, rank=0)
    assert effective_seed(3407, rank=0) != effective_seed(3407, rank=1)

    first = seed_everything(3407, rank=2)
    second = seed_everything(3407, rank=2)
    assert first == second


@pytest.mark.unit
def test_distributed_context_reads_torchrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    context = DistributedContext.from_environment(device_type="cuda")

    assert context.rank == 3
    assert context.local_rank == 1
    assert context.world_size == 8
    assert str(context.device) == "cuda:1"
    assert not context.is_main


@pytest.mark.unit
def test_run_paths_are_timestamped_and_collision_safe(tmp_path: Path) -> None:
    now = datetime(2026, 8, 2, 1, 2, 3, 456789, tzinfo=UTC)
    first = create_run_paths(tmp_path / "outputs", tmp_path / "checkpoints", "train", now)
    second = create_run_paths(tmp_path / "outputs", tmp_path / "checkpoints", "train", now)

    assert first.run_id.startswith("20260802_010203_456789")
    assert first.run_id != second.run_id
    assert first.output_dir.is_dir() and first.checkpoint_dir.is_dir()
    assert second.output_dir.is_dir() and second.checkpoint_dir.is_dir()


@pytest.mark.unit
def test_json_manifest_supports_paths_datetimes_and_dataclasses(tmp_path: Path) -> None:
    destination = tmp_path / "run.json"
    snapshot = ResourceSnapshot(
        timestamp=1.5,
        gpu_allocated_mb=2.0,
        gpu_reserved_mb=3.0,
        gpu_total_mb=4.0,
        gpu_utilization_percent=5.0,
        process_rss_mb=6.0,
    )

    write_json_manifest(
        destination,
        {"path": tmp_path, "when": datetime(2026, 8, 2, tzinfo=UTC), "snapshot": snapshot},
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["path"] == str(tmp_path)
    assert payload["when"] == "2026-08-02T00:00:00+00:00"
    assert payload["snapshot"]["gpu_total_mb"] == 4.0


@pytest.mark.unit
def test_resource_monitor_uses_injected_sampler() -> None:
    samples = iter(
        [
            ResourceSnapshot(0, 1, 2, 10, 40, 5),
            ResourceSnapshot(1, 2, 3, 10, 80, 6),
        ]
    )
    monitor = ResourceMonitor(sample_fn=lambda: next(samples), interval_seconds=0.001)

    monitor.sample_once()
    monitor.sample_once()
    summary = monitor.summary()

    assert summary.sample_count == 2
    assert summary.peak_allocated_mb == 2
    assert summary.mean_gpu_utilization_percent == 60
