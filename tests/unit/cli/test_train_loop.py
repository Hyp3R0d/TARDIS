from __future__ import annotations

import copy
import json
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tardis.cli.common import parse_args
from tardis.data.assembly import RemoteCatalog, RemoteDataLoaders
from tardis.data.contracts import VideoRecord
from tardis.data.dataset import ClipBatch
from tardis.training.engine import ModelEMA, TrainStepResult
from tardis.training.validation import (
    METRIC_BASELINES,
    VALIDATION_SOURCES,
    ValidationCheckpointSelector,
    ValidationMetric,
)
from tardis.utils.checkpoint import atomic_torch_save, load_checkpoint
from tardis.utils.distributed import DistributedContext
from tardis.utils.manifest import RunPaths
from tardis.utils.resources import ResourceSummary


def _args(tmp_path: Path, *, epochs: int = 3, steps: int = 2, interval: int = 2) -> Any:
    return parse_args(
        [
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--epochs",
            str(epochs),
            "--steps-per-epoch",
            str(steps),
            "--gradient-accumulation-steps",
            "1",
            "--validation-interval",
            str(interval),
            "--num-workers",
            "0",
            "--output-root",
            str(tmp_path / "outputs"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
        ]
    )


def _batch(epoch: int, index: int) -> ClipBatch:
    return ClipBatch(
        prompts=[f"prompt-{epoch}-{index}"],
        video=torch.full((1, 2, 3, 2, 2), float(epoch * 10 + index)),
        sources=("dataverse",),
        record_ids=(f"{epoch}-{index}",),
        sample_seeds=(epoch * 100 + index,),
    )


class _TrainStream:
    def __init__(self, owner: _LoopLoaders) -> None:
        self.owner = owner

    def __iter__(self) -> Iterator[ClipBatch]:
        self.owner.train_iterations += 1
        yield from self.owner.batches[self.owner.current_epoch][self.owner.start_batch :]


class _NeverIterated:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self) -> Iterator[object]:
        self.iterations += 1
        yield from ()


class _LoopLoaders:
    def __init__(self, *, epochs: int, steps: int) -> None:
        self.batches = {
            epoch: [_batch(epoch, index) for index in range(steps)] for epoch in range(epochs)
        }
        self.current_epoch = 0
        self.set_epochs: list[int] = []
        self.start_batches: list[int] = []
        self.start_batch = 0
        self.train_iterations = 0
        self.train = _TrainStream(self)
        self.validation = {"dataverse": _NeverIterated()}
        self.test = {"dataverse": _NeverIterated()}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.set_epochs.append(epoch)

    def set_start_batch(self, batch_index: int) -> None:
        self.start_batch = batch_index
        self.start_batches.append(batch_index)


class _TemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Parameter(torch.tensor(1.0))


class _FakeEngine:
    def __init__(
        self,
        model: nn.Module,
        *,
        stop: object | None = None,
        stop_after: int | None = None,
        accumulation_steps: int = 1,
    ) -> None:
        self.unwrapped_model = model
        self.execution_model = model
        self.ema = ModelEMA(model, decay=0.9)
        self.selector = ValidationCheckpointSelector()
        self.micro_step = 0
        self.optimizer_step = 0
        self.accumulation_index = 0
        self.nonfinite_ledger: list[tuple[int, tuple[str, ...]]] = []
        self.processed: list[str] = []
        self.value = 0
        self.stop: Any = stop
        self.stop_after = stop_after
        self.accumulation_steps = accumulation_steps
        self.loaded_rank_state: Mapping[str, object] | None = None

    def train_microbatch(
        self,
        _batch: object,
        *,
        batch_ids: tuple[str, ...] = (),
    ) -> TrainStepResult:
        self.micro_step += 1
        self.accumulation_index += 1
        optimizer_updated = self.accumulation_index == self.accumulation_steps
        if optimizer_updated:
            self.optimizer_step += 1
            self.accumulation_index = 0
        self.processed.extend(batch_ids)
        for record_id in batch_ids:
            self.value = self.value * 37 + sum(record_id.encode("ascii"))
        if self.stop_after == self.micro_step and self.stop is not None:
            self.stop.request("SIGTERM")
        return TrainStepResult(
            total_loss=float(self.value),
            losses={"diffusion": float(self.value)},
            optimizer_updated=optimizer_updated,
            skipped_nonfinite=False,
            gradient_norm=1.0,
            learning_rate=1.0e-4,
            micro_step=self.micro_step,
            optimizer_step=self.optimizer_step,
            stage="metric_alignment",
        )

    def state_dict(self, *, epoch: int, next_batch_index: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "fake_train",
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "value": self.value,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> tuple[int, int]:
        self.micro_step = int(state["micro_step"])
        self.optimizer_step = int(state["optimizer_step"])
        self.value = int(state["value"])
        return int(state["epoch"]), int(state["next_batch_index"])

    def stochastic_state_dict(self) -> dict[str, object]:
        return {"micro_step": self.micro_step, "rank_marker": 0}

    def load_stochastic_state_dict(self, state: Mapping[str, object]) -> None:
        self.loaded_rank_state = state


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []

    def record(self, event: Mapping[str, object]) -> None:
        self.events.append(dict(event))

    def update(self, patch: Mapping[str, object]) -> None:
        self.updates.append(dict(patch))


class _CheckpointWriter:
    def __init__(
        self,
        on_write: Callable[[Mapping[str, object], Path], None] | None = None,
    ) -> None:
        self.writes: list[tuple[Path, dict[str, object]]] = []
        self.on_write = on_write

    def __call__(self, payload: Mapping[str, object], path: Path) -> None:
        atomic_torch_save(payload, path)
        self.writes.append((Path(path), copy.deepcopy(dict(payload))))
        if self.on_write is not None:
            self.on_write(payload, path)

    def payloads(self, filename: str) -> list[dict[str, object]]:
        names = {filename, f"{filename}.candidate"}
        return [payload for path, payload in self.writes if path.name in names]


def _metrics(*, tc_scale: float = 1.0) -> dict[str, dict[str, float]]:
    result = {
        "dataverse_validation": {
            metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric
        }
    }
    for values in result.values():
        values["tc"] *= tc_scale
    return result


def test_validation_summary_reports_six_metrics_for_selected_dataset() -> None:
    from tardis.cli.train import _format_validation_summary
    from tardis.training.validation import score_validation_event

    score = score_validation_event(_metrics())

    summary = _format_validation_summary(
        epoch=3,
        epochs=100,
        score=score,
        improved=True,
    )

    assert summary.startswith("Epoch 3/100 DataVerse validation:")
    for metric in ("TC", "LPIPS", "FVD", "FID", "CLIPScore", "SSIM"):
        assert any(line.split()[0] == metric for line in summary.splitlines()[2:])
        assert summary.count(metric) >= 1
    assert "metric          DataVerse" in summary
    assert summary.count("DataVerse") == 2
    assert "OpenVid" not in summary
    assert "Seedance" not in summary
    rows = summary.splitlines()[2:]
    assert len(rows) == 6
    assert [row.split()[0] for row in rows] == [
        "TC",
        "LPIPS",
        "FVD",
        "FID",
        "CLIPScore",
        "SSIM",
    ]
    assert all(len(row.split()) == 2 for row in rows)
    assert f"weighted_score={score.composite:.6f}" in summary
    assert "target_pass=yes" in summary
    assert "best.pt=updated" in summary


def _paths(tmp_path: Path) -> RunPaths:
    return RunPaths("20260803_010203_000004", tmp_path / "output", tmp_path / "checkpoint")


def test_train_run_recorder_successful_resume_clears_stale_failure_fields(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import TrainRunRecorder

    paths = _paths(tmp_path)
    paths.output_dir.mkdir(parents=True)
    manifest_path = paths.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": paths.run_id,
                "status": "failed",
                "finished_at": "2026-08-10T00:00:00+00:00",
                "resource_summary": {"sample_count": 1},
                "stop_reason": "SIGINT",
                "error": {"type": "OutOfMemoryError", "message": "out of memory"},
            }
        ),
        encoding="utf-8",
    )

    recorder = TrainRunRecorder(paths, {"status": "initializing"})
    try:
        initializing = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert initializing["status"] == "initializing"
        for stale_field in ("finished_at", "resource_summary", "stop_reason", "error"):
            assert stale_field not in initializing

        recorder.finalize(
            status="completed",
            resources=ResourceSummary(1, 2.0, 3.0, 4.0, 5.0),
        )
    finally:
        recorder.close()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert "stop_reason" not in manifest
    assert "error" not in manifest


def _context(*, rank: int = 0) -> DistributedContext:
    return DistributedContext(rank=rank, local_rank=0, world_size=1, device=torch.device("cpu"))


def _validator(
    calls: list[float], metrics: list[dict[str, dict[str, float]]] | None = None
) -> Callable[..., dict[str, dict[str, float]]]:
    pending = list(metrics or [_metrics()])

    def validate(model: nn.Module, loaders: Any, *_args: object, **_kwargs: object) -> Any:
        calls.append(float(next(model.parameters()).item()))
        assert all(item.iterations == 0 for item in loaders.test.values())
        return pending.pop(0) if len(pending) > 1 else pending[0]

    return validate


def test_module_entrypoint_exposes_help_without_loading_weight_dependencies() -> None:
    script = """
import runpy
import sys
sys.argv = ["tardis.cli.train", "--help"]
try:
    runpy.run_module("tardis.cli.train", run_name="__main__")
except SystemExit as error:
    print(f"EXIT={error.code}")
print(any(name in sys.modules for name in ("diffusers", "transformers", "open_clip", "lpips")))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage:" in result.stdout
    assert "EXIT=0" in result.stdout
    assert result.stdout.rstrip().endswith("False")


def test_epoch_loop_runs_exact_microbatches_sets_epochs_and_saves_latest_each_epoch(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path)
    model = _TemporalModel()
    engine = _FakeEngine(model)
    loaders = _LoopLoaders(epochs=3, steps=2)
    writer = _CheckpointWriter()
    recorder = _Recorder()
    validation_weights: list[float] = []

    result = run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=engine,
        loaders=loaders,
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={"test": "signature"},
        start_position=(0, 0),
        stop=CooperativeStop(),
        recorder=recorder,
        validation_runner=_validator(validation_weights),
        checkpoint_writer=writer,
    )

    assert result.status == "completed"
    assert result.position == (3, 0)
    assert result.microbatches_completed == 6
    assert result.validation_epochs == (2, 3)
    assert loaders.set_epochs == [0, 1, 2]
    assert engine.processed == ["0-0", "0-1", "1-0", "1-1", "2-0", "2-1"]
    assert [(item["epoch"], item["next_batch_index"]) for item in writer.payloads("latest.pt")] == [
        (1, 0),
        (2, 0),
        (3, 0),
    ]
    assert len([event for event in recorder.events if event["type"] == "microbatch"]) == 6
    assert all(item.iterations == 0 for item in loaders.test.values())


def test_periodic_checkpoint_saves_running_latest_only_at_accumulation_boundaries(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=6, interval=1)
    args.checkpoint_interval_steps = 2
    args.gradient_accumulation_steps = 2
    model = _TemporalModel()
    engine = _FakeEngine(model, accumulation_steps=2)
    writer = _CheckpointWriter()

    run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=engine,
        loaders=_LoopLoaders(epochs=1, steps=6),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=CooperativeStop(),
        validation_runner=_validator([]),
        checkpoint_writer=writer,
    )

    latest = writer.payloads("latest.pt")
    assert [
        (payload["epoch"], payload["next_batch_index"], payload["run_status"])
        for payload in latest
    ] == [
        (0, 2, "running"),
        (0, 4, "running"),
        (1, 0, "completed"),
    ]
    assert writer.payloads("best.pt")[-1]["epoch"] == 1


def test_validation_uses_ema_then_restores_student_and_runs_on_cadence_plus_final(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=3, steps=2, interval=2)
    model = _TemporalModel()
    engine = _FakeEngine(model)
    engine.ema.shadow["temporal"].fill_(7.0)
    observed: list[float] = []

    result = run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=engine,
        loaders=_LoopLoaders(epochs=3, steps=2),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=CooperativeStop(),
        validation_runner=_validator(observed),
        checkpoint_writer=_CheckpointWriter(),
    )

    assert result.validation_epochs == (2, 3)
    assert observed == [7.0, 7.0]
    assert model.temporal.item() == pytest.approx(1.0)


def test_stop_on_final_microbatch_checkpoints_without_starting_validation(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=1, interval=1)
    model = _TemporalModel()
    stop = CooperativeStop()
    writer = _CheckpointWriter()
    validation_calls: list[object] = []

    def forbidden_validation(*_args: object, **_kwargs: object) -> Any:
        validation_calls.append(object())
        raise AssertionError("validation must not start after a cooperative stop")

    result = run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=_FakeEngine(model, stop=stop, stop_after=1),
        loaders=_LoopLoaders(epochs=1, steps=1),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=stop,
        validation_runner=forbidden_validation,
        checkpoint_writer=writer,
    )

    assert result.status == "interrupted"
    assert result.position == (1, 0)
    assert result.validation_epochs == ()
    assert validation_calls == []
    assert [payload["run_status"] for payload in writer.payloads("latest.pt")] == ["interrupted"]


def test_best_checkpoint_changes_only_on_strict_validation_score_improvement(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=6, steps=1, interval=2)
    model = _TemporalModel()
    engine = _FakeEngine(model)
    writer = _CheckpointWriter()
    validation_metrics = [_metrics(), _metrics(tc_scale=0.5), _metrics(tc_scale=0.5)]

    run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=engine,
        loaders=_LoopLoaders(epochs=6, steps=1),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=CooperativeStop(),
        validation_runner=_validator([], validation_metrics),
        checkpoint_writer=writer,
    )

    assert [payload["epoch"] for payload in writer.payloads("latest.pt")] == [1, 2, 3, 4, 5, 6]
    assert [payload["epoch"] for payload in writer.payloads("best.pt")] == [2, 4]
    assert engine.selector.best_epoch == 4


def test_stop_during_validation_discards_metrics_and_checkpoints_restored_student(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=1, interval=1)
    model = _TemporalModel()
    engine = _FakeEngine(model)
    engine.ema.shadow["temporal"].fill_(7.0)
    stop = CooperativeStop()
    writer = _CheckpointWriter()
    recorder = _Recorder()
    validation_weights: list[float] = []

    def interrupted_validation(
        validation_model: nn.Module,
        loaders: Any,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, dict[str, float]]:
        validation_weights.append(float(next(validation_model.parameters()).item()))
        assert all(item.iterations == 0 for item in loaders.test.values())
        stop.request("SIGTERM")
        return _metrics(tc_scale=0.1)

    result = run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=engine,
        loaders=_LoopLoaders(epochs=1, steps=1),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=stop,
        recorder=recorder,
        validation_runner=interrupted_validation,
        checkpoint_writer=writer,
    )

    assert result.status == "interrupted"
    assert result.position == (1, 0)
    assert result.validation_epochs == ()
    assert validation_weights == [7.0]
    assert model.temporal.item() == pytest.approx(1.0)
    assert engine.selector.best_score is None
    assert engine.selector.best_epoch is None
    assert writer.payloads("best.pt") == []
    assert [payload["run_status"] for payload in writer.payloads("latest.pt")] == ["interrupted"]
    assert writer.payloads("latest.pt")[0]["validation_metrics"] is None
    assert not any(event["type"] == "validation" for event in recorder.events)


def test_interrupted_resume_skips_consumed_batches_and_matches_uninterrupted_execution(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=3, steps=2, interval=3)
    uninterrupted_model = _TemporalModel()
    uninterrupted = _FakeEngine(uninterrupted_model)
    run_train_epoch_loop(
        args,
        context=_context(),
        model=uninterrupted_model,
        engine=uninterrupted,
        loaders=_LoopLoaders(epochs=3, steps=2),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=CooperativeStop(),
        validation_runner=_validator([]),
        checkpoint_writer=_CheckpointWriter(),
    )

    stop = CooperativeStop()
    partial_model = _TemporalModel()
    partial = _FakeEngine(partial_model, stop=stop, stop_after=3)
    first_writer = _CheckpointWriter()
    first = run_train_epoch_loop(
        args,
        context=_context(),
        model=partial_model,
        engine=partial,
        loaders=_LoopLoaders(epochs=3, steps=2),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=stop,
        validation_runner=_validator([]),
        checkpoint_writer=first_writer,
    )
    interrupted_payload = first_writer.payloads("latest.pt")[-1]

    resumed_model = _TemporalModel()
    resumed = _FakeEngine(resumed_model)
    position = resumed.load_state_dict(interrupted_payload)
    resumed.load_stochastic_state_dict(interrupted_payload["rank_random_states"]["0"])
    resumed_loaders = _LoopLoaders(epochs=3, steps=2)
    second = run_train_epoch_loop(
        args,
        context=_context(),
        model=resumed_model,
        engine=resumed,
        loaders=resumed_loaders,
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=position,
        stop=CooperativeStop(),
        validation_runner=_validator([]),
        checkpoint_writer=_CheckpointWriter(),
    )

    assert first.status == "interrupted"
    assert first.position == (1, 1)
    assert interrupted_payload["run_status"] == "interrupted"
    assert second.status == "completed"
    assert resumed_loaders.set_epochs == [1, 2]
    assert resumed_loaders.start_batches == [1, 0]
    assert resumed.processed == ["1-1", "2-0", "2-1"]
    assert resumed.value == uninterrupted.value
    assert resumed.micro_step == uninterrupted.micro_step == 6


@pytest.mark.parametrize("position", [(1, 2), (3, 1), (4, 0)])
def test_epoch_loop_rejects_out_of_range_resume_positions(
    tmp_path: Path,
    position: tuple[int, int],
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=3, steps=2)
    model = _TemporalModel()
    engine = _FakeEngine(model)
    engine.micro_step = position[0] * 2 + position[1]

    with pytest.raises(ValueError, match="resume position"):
        run_train_epoch_loop(
            args,
            context=_context(),
            model=model,
            engine=engine,
            loaders=_LoopLoaders(epochs=3, steps=2),
            metric_suite=object(),
            paths=_paths(tmp_path),
            training_signature={},
            start_position=position,
            stop=CooperativeStop(),
            validation_runner=_validator([]),
            checkpoint_writer=_CheckpointWriter(),
        )


def test_epoch_loop_rejects_progress_that_disagrees_with_engine_state(tmp_path: Path) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=3, steps=2)
    model = _TemporalModel()
    engine = _FakeEngine(model)

    with pytest.raises(ValueError, match="micro_step"):
        run_train_epoch_loop(
            args,
            context=_context(),
            model=model,
            engine=engine,
            loaders=_LoopLoaders(epochs=3, steps=2),
            metric_suite=object(),
            paths=_paths(tmp_path),
            training_signature={},
            start_position=(1, 1),
            stop=CooperativeStop(),
            validation_runner=_validator([]),
            checkpoint_writer=_CheckpointWriter(),
        )


def test_nonzero_rank_never_writes_checkpoints_or_run_artifacts(tmp_path: Path) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=1, interval=1)
    model = _TemporalModel()
    writer = _CheckpointWriter()
    recorder = _Recorder()

    run_train_epoch_loop(
        args,
        context=_context(rank=1),
        model=model,
        engine=_FakeEngine(model),
        loaders=_LoopLoaders(epochs=1, steps=1),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=CooperativeStop(),
        recorder=recorder,
        validation_runner=_validator([]),
        checkpoint_writer=writer,
    )

    assert writer.writes == []
    assert recorder.events == []
    assert recorder.updates == []
    assert not _paths(tmp_path).checkpoint_dir.exists()


def test_coordinate_run_paths_allocates_only_on_rank_zero(tmp_path: Path) -> None:
    from tardis.cli.train import coordinate_run_paths

    args = _args(tmp_path)
    expected = _paths(tmp_path)
    allocations: list[tuple[object, ...]] = []
    broadcasts: list[RunPaths | None] = []

    def allocate(*allocator_args: object, **_kwargs: object) -> RunPaths:
        allocations.append(allocator_args)
        return expected

    def broadcast(paths: RunPaths | None, _context: object) -> RunPaths:
        broadcasts.append(paths)
        return expected

    main = coordinate_run_paths(
        args,
        _context(),
        allocator=allocate,
        broadcaster=broadcast,
    )
    worker = coordinate_run_paths(
        args,
        _context(rank=1),
        allocator=allocate,
        broadcaster=broadcast,
    )

    assert main == worker == expected
    assert len(allocations) == 1
    assert allocations[0] == (
        args.output_root,
        args.checkpoint_root / "dataverse",
        "train/dataverse",
    )
    assert broadcasts == [expected, None]


def test_dataset_checkpoint_path_is_scoped_to_selected_dataset(tmp_path: Path) -> None:
    from tardis.cli.train import validate_dataset_checkpoint_path

    checkpoint_root = tmp_path / "checkpoints"
    dataverse = checkpoint_root / "dataverse" / "20260810_010203_000000" / "best.pt"
    openvid = checkpoint_root / "openvid" / "20260810_010204_000000" / "best.pt"
    dataverse.parent.mkdir(parents=True)
    openvid.parent.mkdir(parents=True)
    dataverse.touch()
    openvid.touch()

    assert (
        validate_dataset_checkpoint_path(
            dataverse,
            checkpoint_root=checkpoint_root,
            dataset="dataverse",
            purpose="warm-start",
        )
        == dataverse.resolve()
    )
    with pytest.raises(ValueError, match="must belong to dataset 'dataverse'"):
        validate_dataset_checkpoint_path(
            openvid,
            checkpoint_root=checkpoint_root,
            dataset="dataverse",
            purpose="warm-start",
        )


def test_dataset_checkpoint_path_allows_explicit_cross_dataset_warm_start(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import validate_dataset_checkpoint_path

    checkpoint_root = tmp_path / "checkpoints"
    dataverse = checkpoint_root / "dataverse" / "20260810_010203_000000" / "best.pt"
    dataverse.parent.mkdir(parents=True)
    dataverse.touch()

    assert (
        validate_dataset_checkpoint_path(
            dataverse,
            checkpoint_root=checkpoint_root,
            dataset="seedance",
            purpose="warm-start",
            allow_cross_dataset=True,
        )
        == dataverse.resolve()
    )


def test_signal_scope_installs_and_restores_sigint_and_sigterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tardis.cli.train import CooperativeStop, cooperative_signal_handlers

    stop = CooperativeStop()
    previous = {signal.SIGINT: object(), signal.SIGTERM: object()}
    installed: list[tuple[signal.Signals, object]] = []
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(
        signal, "signal", lambda signum, handler: installed.append((signum, handler))
    )

    with cooperative_signal_handlers(stop):
        handlers = dict(installed)
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert stop.requested
        assert stop.reason == "SIGTERM"

    assert installed[-2:] == [
        (signal.SIGINT, previous[signal.SIGINT]),
        (signal.SIGTERM, previous[signal.SIGTERM]),
    ]


def test_final_synchronized_stop_decision_never_writes_completed_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tardis.cli.train as train_module
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=1, interval=1)
    model = _TemporalModel()
    stop = CooperativeStop()
    synchronization_calls = 0
    synchronize = train_module._distributed_stop_requested

    def request_at_final_decision(
        current_stop: CooperativeStop,
        context: DistributedContext,
    ) -> bool:
        nonlocal synchronization_calls
        synchronization_calls += 1
        if synchronization_calls == 5:
            stop.request("SIGTERM")
        return synchronize(current_stop, context)

    monkeypatch.setattr(train_module, "_distributed_stop_requested", request_at_final_decision)
    writer = _CheckpointWriter()
    result = run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=_FakeEngine(model),
        loaders=_LoopLoaders(epochs=1, steps=1),
        metric_suite=object(),
        paths=_paths(tmp_path),
        training_signature={},
        start_position=(0, 0),
        stop=stop,
        validation_runner=_validator([]),
        checkpoint_writer=writer,
    )

    assert result.status == "interrupted"
    assert synchronization_calls == 5
    assert [payload["run_status"] for payload in writer.payloads("latest.pt")] == [
        "interrupted",
    ]


def test_stop_during_completed_checkpoint_write_never_persists_or_returns_completed(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=1, interval=1)
    model = _TemporalModel()
    stop = CooperativeStop()
    paths = _paths(tmp_path)
    writes: list[tuple[str, str]] = []

    def request_stop_during_completed_write(
        payload: Mapping[str, object],
        path: Path,
    ) -> None:
        atomic_torch_save(payload, path)
        status = str(payload["run_status"])
        writes.append((path.name, status))
        if status == "completed":
            stop.request("SIGTERM")

    result = run_train_epoch_loop(
        args,
        context=_context(),
        model=model,
        engine=_FakeEngine(model),
        loaders=_LoopLoaders(epochs=1, steps=1),
        metric_suite=object(),
        paths=paths,
        training_signature={},
        start_position=(0, 0),
        stop=stop,
        validation_runner=_validator([]),
        checkpoint_writer=request_stop_during_completed_write,
    )
    persisted = load_checkpoint(paths.checkpoint_dir / "latest.pt")

    assert any(status == "completed" for _, status in writes)
    assert result.status == "interrupted"
    assert persisted["run_status"] == "interrupted"
    assert ("latest.pt", "completed") not in writes


def test_completed_checkpoint_writer_failure_discards_all_candidates(tmp_path: Path) -> None:
    from tardis.cli.train import CooperativeStop, run_train_epoch_loop

    args = _args(tmp_path, epochs=1, steps=1, interval=1)
    model = _TemporalModel()
    paths = _paths(tmp_path)
    atomic_torch_save(
        {"schema_version": 1, "run_status": "running", "marker": "canonical-latest"},
        paths.checkpoint_dir / "latest.pt",
    )
    atomic_torch_save(
        {"schema_version": 1, "run_status": "running", "marker": "canonical-best"},
        paths.checkpoint_dir / "best.pt",
    )

    def fail_after_writing_best_candidate(
        payload: Mapping[str, object],
        path: Path,
    ) -> None:
        atomic_torch_save(payload, path)
        if path.name == "best.pt.candidate":
            raise OSError("injected completion writer failure")

    with pytest.raises(OSError, match="injected completion writer failure"):
        run_train_epoch_loop(
            args,
            context=_context(),
            model=model,
            engine=_FakeEngine(model),
            loaders=_LoopLoaders(epochs=1, steps=1),
            metric_suite=object(),
            paths=paths,
            training_signature={},
            start_position=(0, 0),
            stop=CooperativeStop(),
            validation_runner=_validator([]),
            checkpoint_writer=fail_after_writing_best_candidate,
        )

    assert not list(paths.checkpoint_dir.glob("*.candidate"))
    assert load_checkpoint(paths.checkpoint_dir / "latest.pt")["marker"] == "canonical-latest"
    assert load_checkpoint(paths.checkpoint_dir / "best.pt")["marker"] == "canonical-best"


def test_partial_completion_promotion_restores_canonical_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tardis.cli.train import _promote_completion_candidates

    paths = _paths(tmp_path)
    checkpoint_dir = paths.checkpoint_dir
    for name, marker in (
        ("latest.pt", "canonical-latest"),
        ("best.pt", "canonical-best"),
        ("latest.pt.candidate", "candidate-latest"),
        ("best.pt.candidate", "candidate-best"),
    ):
        atomic_torch_save(
            {"schema_version": 1, "run_status": "running", "marker": marker},
            checkpoint_dir / name,
        )
    original_replace = Path.replace

    def fail_latest_candidate_promotion(self: Path, target: str | Path) -> Path:
        if self.name == "latest.pt.candidate":
            raise OSError("injected completion promotion failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_latest_candidate_promotion)

    with pytest.raises(OSError, match="injected completion promotion failure"):
        _promote_completion_candidates(paths, _context(), improved=True)

    assert {path.name for path in checkpoint_dir.iterdir()} == {"latest.pt", "best.pt"}
    assert load_checkpoint(checkpoint_dir / "latest.pt")["marker"] == "canonical-latest"
    assert load_checkpoint(checkpoint_dir / "best.pt")["marker"] == "canonical-best"


def test_compile_mode_is_part_of_resume_compatibility_signature(tmp_path: Path) -> None:
    from tardis.cli.train import _training_signature, _validate_resume_checkpoint

    eager_args = _args(tmp_path)
    compiled_args = SimpleNamespace(**vars(eager_args))
    compiled_args.compile_model = True
    runtime = SimpleNamespace(
        factory_options={"architecture": "tiny"},
        metric_suite=SimpleNamespace(provenance_ids={"tc": "tiny-tc"}),
    )
    sources = {source: {"revision": "1" * 40} for source in VALIDATION_SOURCES}
    eager_signature = _training_signature(eager_args, runtime, sources, world_size=1)
    compiled_signature = _training_signature(compiled_args, runtime, sources, world_size=1)
    paths = _paths(tmp_path)
    checkpoint = {
        "run_id": paths.run_id,
        "world_size": 1,
        "training_signature": eager_signature,
        "rank_random_states": {"0": {}},
    }

    assert eager_signature["args"]["compile_model"] is False
    assert compiled_signature["args"]["compile_model"] is True
    with pytest.raises(ValueError, match="training signature is incompatible"):
        _validate_resume_checkpoint(
            checkpoint,
            paths=paths,
            signature=compiled_signature,
            world_size=1,
        )


def test_source_identity_matches_local_paths_by_directory_not_catalog_line_order() -> None:
    from tardis.cli.train import _source_identity

    revisions = {source: source[0] * 40 for source in ("dataverse", "openvid", "seedance")}
    records = {
        source: (
            VideoRecord(
                id=f"{source}-0",
                caption="prompt",
                media_locator="https://media.invalid/video.mp4",
                source=source,
                metadata={"revision": revision},
            ),
        )
        for source, revision in revisions.items()
    }
    splits = {
        split: {source: source_records for source, source_records in records.items()}
        for split in ("train", "validation", "test")
    }
    loaders = RemoteDataLoaders(
        catalog=RemoteCatalog(records),
        splits=splits,
        train_dataset=object(),
        train=object(),
        validation={},
        test={},
    )
    paths = (
        "/root/autodl-tmp/TARDIS/datasets/OpenVid-1M",
        "/root/autodl-tmp/TARDIS/datasets/seedance-2-prompts-datasets",
        "/root/autodl-tmp/TARDIS/datasets/Vchitect_T2V_DataVerse",
    )

    identity = _source_identity(loaders, paths, dataset="openvid")

    assert identity["openvid"]["path"].endswith("OpenVid-1M")
    assert set(identity) == {"openvid"}


class _ManagedContext:
    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.initialized = 0
        self.closed = 0

    @property
    def is_main(self) -> bool:
        return True

    def initialize(self) -> None:
        self.initialized += 1

    def barrier(self) -> None:
        pass

    def close(self) -> None:
        self.closed += 1


class _Monitor:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def sample_once(self) -> None:
        pass

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def summary(self) -> ResourceSummary:
        return ResourceSummary(1, 2.0, 3.0, 4.0, 5.0)


def test_run_training_closes_monitor_context_and_marks_manifest_failed_on_exception(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import TrainServices, run_training

    args = _args(tmp_path)
    context = _ManagedContext()
    monitor = _Monitor()
    runtime_calls: list[bool] = []

    def fail_runtime(_args: object, *, restore_checkpoint: bool) -> object:
        runtime_calls.append(restore_checkpoint)
        raise RuntimeError("assembly failed")

    services = TrainServices(
        context_factory=lambda _device_type: context,
        runtime_builder=fail_runtime,
        monitor_factory=lambda _device: monitor,
    )

    with pytest.raises(RuntimeError, match="assembly failed"):
        run_training(args, services=services)

    assert runtime_calls == [False]
    assert context.initialized == context.closed == 1
    assert monitor.started == monitor.stopped == 1
    manifests = list((tmp_path / "outputs" / "train" / "dataverse").glob("*/manifest.json"))
    resources = list((tmp_path / "outputs" / "train" / "dataverse").glob("*/resources.json"))
    assert len(manifests) == len(resources) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["args"]["epochs"] == 3


def test_distributed_wrapper_requires_find_unused_parameters() -> None:
    from tardis.cli.train import wrap_distributed_model

    model = _TemporalModel()
    calls: list[tuple[nn.Module, dict[str, object]]] = []
    sentinel = nn.Identity()

    def factory(module: nn.Module, **kwargs: object) -> nn.Module:
        calls.append((module, kwargs))
        return sentinel

    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )

    assert wrap_distributed_model(model, context, ddp_factory=factory) is sentinel
    assert calls == [(model, {"find_unused_parameters": True})]
