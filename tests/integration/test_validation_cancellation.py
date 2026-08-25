from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tardis.cli.common import parse_args
from tardis.cli.train import CooperativeStop, run_train_epoch_loop
from tardis.data.assembly import RemoteDataLoaders
from tardis.data.dataset import ClipBatch
from tardis.training.engine import ModelEMA, TrainStepResult
from tardis.training.validation import (
    METRIC_BASELINES,
    ValidationCheckpointSelector,
    ValidationMetric,
)
from tardis.utils.checkpoint import atomic_torch_save, load_checkpoint
from tardis.utils.distributed import DistributedContext
from tardis.utils.manifest import RunPaths

_SOURCES = ("dataverse",)


def _wait_for_file(path: Path, *, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _complete_validation_metrics() -> dict[str, dict[str, float]]:
    return {
        f"{source}_validation": {
            metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric
        }
        for source in _SOURCES
    }


@dataclass(frozen=True, slots=True)
class _Record:
    id: str


class _BenchmarkDataset:
    def __init__(
        self,
        records: tuple[_Record, ...],
        *,
        source: str,
        rank: int,
        world_size: int,
    ) -> None:
        self.records = records
        self.source = source
        self.split = "validation"
        self.rank = rank
        self.world_size = world_size

    def __len__(self) -> int:
        return len(self.records[self.rank :: self.world_size])


class _BenchmarkLoader:
    def __init__(self, dataset: _BenchmarkDataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[ClipBatch]:
        for record in self.dataset.records[self.dataset.rank :: self.dataset.world_size]:
            yield ClipBatch(
                prompts=[record.id],
                video=torch.zeros(1, 2, 3, 2, 2),
                sources=(self.dataset.source,),
                record_ids=(record.id,),
                sample_seeds=(17,),
            )


class _EpochDataset:
    def set_epoch(self, _epoch: int) -> None:
        pass


class _NeverLoader:
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("test loaders must not be iterated during checkpoint validation")
        yield


class _MetricSuite:
    def __init__(self) -> None:
        self.updates = 0
        self.reductions = 0
        self.resets = 0

    def update(self, _generated: torch.Tensor, _reference: torch.Tensor, _prompt: str) -> None:
        self.updates += 1

    def all_reduce(self) -> None:
        self.reductions += 1

    def compute(self) -> dict[str, dict[str, float]]:
        macro = {metric.value: METRIC_BASELINES[metric] for metric in ValidationMetric}
        return {"macro": macro, "micro": dict(macro)}

    def reset(self) -> None:
        self.resets += 1


class _StoppingModel(nn.Module):
    def __init__(self, stop: CooperativeStop, *, request_stop: bool) -> None:
        super().__init__()
        self.temporal = nn.Parameter(torch.tensor(1.0))
        self.stop = stop
        self.request_stop = request_stop
        self.generate_calls = 0

    def generate(
        self,
        prompts: list[str],
        num_frames: int,
        fps: int,
        generator: torch.Generator,
    ) -> SimpleNamespace:
        del prompts, fps, generator
        self.generate_calls += 1
        if self.request_stop and self.generate_calls == 1:
            self.stop.request("SIGTERM")
        return SimpleNamespace(video=torch.zeros(1, num_frames, 3, 2, 2))


class _Engine:
    def __init__(self, model: nn.Module, rank: int) -> None:
        self.unwrapped_model = model
        self.execution_model = model
        self.ema = ModelEMA(model, decay=0.9)
        self.ema.shadow["temporal"].fill_(7.0)
        self.selector = ValidationCheckpointSelector()
        self.micro_step = 0
        self.optimizer_step = 0
        self.accumulation_index = 0
        self.nonfinite_ledger: list[tuple[int, tuple[str, ...]]] = []
        self.rank = rank

    def train_microbatch(
        self,
        _batch: object,
        *,
        batch_ids: tuple[str, ...] = (),
    ) -> TrainStepResult:
        del batch_ids
        self.micro_step += 1
        self.optimizer_step += 1
        return TrainStepResult(
            total_loss=0.0,
            losses={"diffusion": 0.0},
            optimizer_updated=True,
            skipped_nonfinite=False,
            gradient_norm=0.0,
            learning_rate=0.0,
            micro_step=self.micro_step,
            optimizer_step=self.optimizer_step,
            stage="metric_alignment",
        )

    def state_dict(self, *, epoch: int, next_batch_index: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "distributed-validation-test",
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
        }

    def stochastic_state_dict(self) -> dict[str, object]:
        return {"rank": self.rank, "micro_step": self.micro_step}


def _loaders(rank: int, world_size: int) -> RemoteDataLoaders:
    records = {
        source: tuple(_Record(f"{source}-{index}") for index in range(3)) for source in _SOURCES
    }
    validation = {
        source: _BenchmarkLoader(
            _BenchmarkDataset(
                source_records,
                source=source,
                rank=rank,
                world_size=world_size,
            )
        )
        for source, source_records in records.items()
    }
    splits = {
        "train": {source: () for source in _SOURCES},
        "validation": records,
        "test": {source: () for source in _SOURCES},
    }
    return RemoteDataLoaders(
        catalog=object(),
        splits=splits,
        train_dataset=_EpochDataset(),
        train=[
            ClipBatch(
                prompts=[f"train-{rank}"],
                video=torch.zeros(1, 2, 3, 2, 2),
                sources=("dataverse",),
                record_ids=(f"train-{rank}",),
                sample_seeds=(rank,),
            )
        ],
        validation=validation,
        test={source: _NeverLoader() for source in _SOURCES},
    )


def _run_validation_cancel_rank(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        root_path = Path(root)
        stop = CooperativeStop()
        model = _StoppingModel(stop, request_stop=rank == 1).train()
        engine = _Engine(model, rank)
        suite = _MetricSuite()
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        args = parse_args(
            [
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--validation-interval",
                "1",
                "--num-workers",
                "0",
            ]
        )
        paths = RunPaths("20260803_010203_000004", root_path / "output", root_path / "checkpoint")
        result = run_train_epoch_loop(
            args,
            context=context,
            model=model,
            engine=engine,
            loaders=_loaders(rank, world_size),
            metric_suite=suite,
            paths=paths,
            training_signature={},
            start_position=(0, 0),
            stop=stop,
        )
        (root_path / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "position": result.position,
                    "validation_epochs": result.validation_epochs,
                    "generate_calls": model.generate_calls,
                    "model_training": model.training,
                    "temporal": model.temporal.item(),
                    "metric_updates": suite.updates,
                    "metric_reductions": suite.reductions,
                    "metric_resets": suite.resets,
                    "best_epoch": engine.selector.best_epoch,
                }
            ),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_completion_write_cancel_rank(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        root_path = Path(root)
        stop = CooperativeStop()
        model = _StoppingModel(stop, request_stop=False).train()
        engine = _Engine(model, rank)
        suite = _MetricSuite()
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        args = parse_args(
            [
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--validation-interval",
                "1",
                "--num-workers",
                "0",
            ]
        )
        paths = RunPaths("20260803_010203_000004", root_path / "output", root_path / "checkpoint")

        def request_stop_during_completed_write(
            payload: Mapping[str, object],
            path: Path,
        ) -> None:
            atomic_torch_save(payload, path)
            if payload["run_status"] == "completed":
                stop.request("SIGTERM")

        result = run_train_epoch_loop(
            args,
            context=context,
            model=model,
            engine=engine,
            loaders=_loaders(rank, world_size),
            metric_suite=suite,
            paths=paths,
            training_signature={},
            start_position=(0, 0),
            stop=stop,
            checkpoint_writer=request_stop_during_completed_write,
        )
        (root_path / f"writer-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "position": result.position,
                    "stop_reason": result.stop_reason,
                }
            ),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_non_main_completion_write_cancel_rank(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    original_all_reduce = dist.all_reduce
    try:
        root_path = Path(root)
        writer_active = root_path / "rank0-writer-active"
        rank1_stopped = root_path / "rank1-stop-requested"
        stop = CooperativeStop()
        model = _StoppingModel(stop, request_stop=False).train()
        engine = _Engine(model, rank)
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        args = parse_args(
            [
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--validation-interval",
                "1",
                "--num-workers",
                "0",
            ]
        )
        paths = RunPaths("20260803_010203_000004", root_path / "output", root_path / "checkpoint")
        all_reduce_calls = 0

        def intercept_all_reduce(
            tensor: torch.Tensor,
            *call_args: object,
            **call_kwargs: object,
        ) -> object:
            nonlocal all_reduce_calls
            all_reduce_calls += 1
            if rank == 1 and all_reduce_calls == 6:
                assert tensor.item() == 0
                _wait_for_file(writer_active)
                stop.request("SIGTERM")
                rank1_stopped.write_text("requested", encoding="utf-8")
            return original_all_reduce(tensor, *call_args, **call_kwargs)

        dist.all_reduce = intercept_all_reduce

        def block_rank0_completed_writer(
            payload: Mapping[str, object],
            path: Path,
        ) -> None:
            if (
                rank == 0
                and payload["run_status"] == "completed"
                and path.name == "latest.pt.candidate"
            ):
                writer_active.write_text("active", encoding="utf-8")
                _wait_for_file(rank1_stopped)
            atomic_torch_save(payload, path)

        def complete_validation(
            *_args: object,
            **_kwargs: object,
        ) -> dict[str, dict[str, float]]:
            return _complete_validation_metrics()

        result = run_train_epoch_loop(
            args,
            context=context,
            model=model,
            engine=engine,
            loaders=_loaders(rank, world_size),
            metric_suite=_MetricSuite(),
            paths=paths,
            training_signature={},
            start_position=(0, 0),
            stop=stop,
            validation_runner=complete_validation,
            checkpoint_writer=block_rank0_completed_writer,
        )
        (root_path / f"non-main-writer-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "position": result.position,
                    "stop_reason": result.stop_reason,
                    "all_reduce_calls": all_reduce_calls,
                }
            ),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.all_reduce = original_all_reduce
        dist.destroy_process_group()


def _run_completion_write_failure_rank(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        root_path = Path(root)
        stop = CooperativeStop()
        model = _StoppingModel(stop, request_stop=False).train()
        engine = _Engine(model, rank)
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        args = parse_args(
            [
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--validation-interval",
                "1",
                "--num-workers",
                "0",
            ]
        )
        paths = RunPaths("20260803_010203_000004", root_path / "output", root_path / "checkpoint")

        def fail_after_writing_best_candidate(
            payload: Mapping[str, object],
            path: Path,
        ) -> None:
            atomic_torch_save(payload, path)
            if path.name == "best.pt.candidate":
                raise OSError("injected completion writer failure")

        try:
            run_train_epoch_loop(
                args,
                context=context,
                model=model,
                engine=engine,
                loaders=_loaders(rank, world_size),
                metric_suite=_MetricSuite(),
                paths=paths,
                training_signature={},
                start_position=(0, 0),
                stop=stop,
                validation_runner=lambda *_args, **_kwargs: _complete_validation_metrics(),
                checkpoint_writer=fail_after_writing_best_candidate,
            )
        except BaseException as error:
            (root_path / f"writer-failure-rank-{rank}.json").write_text(
                json.dumps({"type": type(error).__name__, "message": str(error)}),
                encoding="utf-8",
            )
        else:
            raise AssertionError("completion writer failure did not converge to this rank")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_completion_promotion_failure_rank(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    original_replace = Path.replace
    try:
        root_path = Path(root)
        stop = CooperativeStop()
        model = _StoppingModel(stop, request_stop=False).train()
        engine = _Engine(model, rank)
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        args = parse_args(
            [
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--validation-interval",
                "1",
                "--num-workers",
                "0",
            ]
        )
        paths = RunPaths("20260803_010203_000004", root_path / "output", root_path / "checkpoint")

        def fail_latest_candidate_promotion(self: Path, target: str | Path) -> Path:
            if rank == 0 and self.name == "latest.pt.candidate":
                raise OSError("injected completion promotion failure")
            return original_replace(self, target)

        Path.replace = fail_latest_candidate_promotion
        try:
            run_train_epoch_loop(
                args,
                context=context,
                model=model,
                engine=engine,
                loaders=_loaders(rank, world_size),
                metric_suite=_MetricSuite(),
                paths=paths,
                training_signature={},
                start_position=(0, 0),
                stop=stop,
                validation_runner=lambda *_args, **_kwargs: _complete_validation_metrics(),
            )
        except BaseException as error:
            (root_path / f"promotion-failure-rank-{rank}.json").write_text(
                json.dumps({"type": type(error).__name__, "message": str(error)}),
                encoding="utf-8",
            )
        else:
            raise AssertionError("completion promotion failure did not converge to this rank")
    finally:
        Path.replace = original_replace
        dist.destroy_process_group()


@pytest.mark.integration
def test_validation_stop_converges_all_ranks_to_one_interrupted_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    world_size = 2
    rendezvous = str((tmp_path / "gloo-init").resolve())
    mp.spawn(
        _run_validation_cancel_rank,
        args=(world_size, rendezvous, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    checkpoint = load_checkpoint(tmp_path / "checkpoint" / "latest.pt")

    assert {report["status"] for report in reports} == {"interrupted"}
    assert {tuple(report["position"]) for report in reports} == {(1, 0)}
    assert all(report["validation_epochs"] == [] for report in reports)
    assert all(report["generate_calls"] == 1 for report in reports)
    assert all(report["model_training"] is True for report in reports)
    assert all(report["temporal"] == pytest.approx(1.0) for report in reports)
    assert all(report["metric_updates"] == report["metric_resets"] == 1 for report in reports)
    assert all(report["metric_reductions"] == 0 for report in reports)
    assert all(report["best_epoch"] is None for report in reports)
    assert checkpoint["run_status"] == "interrupted"
    assert checkpoint["validation_metrics"] is None
    assert checkpoint["epoch"] == 1
    assert checkpoint["next_batch_index"] == 0
    assert set(checkpoint["rank_random_states"]) == {"0", "1"}


@pytest.mark.integration
def test_completed_writer_stop_converges_all_ranks_without_persisting_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    world_size = 2
    rendezvous = str((tmp_path / "gloo-writer-init").resolve())
    mp.spawn(
        _run_completion_write_cancel_rank,
        args=(world_size, rendezvous, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    reports = [
        json.loads((tmp_path / f"writer-rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    checkpoint = load_checkpoint(tmp_path / "checkpoint" / "latest.pt")

    assert {report["status"] for report in reports} == {"interrupted"}
    assert {tuple(report["position"]) for report in reports} == {(1, 0)}
    assert checkpoint["run_status"] == "interrupted"
    assert not list((tmp_path / "checkpoint").glob("*.candidate"))


@pytest.mark.integration
def test_non_main_stop_during_rank0_completed_writer_is_not_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    world_size = 2
    rendezvous = str((tmp_path / "gloo-non-main-writer-init").resolve())
    mp.spawn(
        _run_non_main_completion_write_cancel_rank,
        args=(world_size, rendezvous, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    reports = [
        json.loads((tmp_path / f"non-main-writer-rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    checkpoint = load_checkpoint(tmp_path / "checkpoint" / "latest.pt")

    assert {report["status"] for report in reports} == {"interrupted"}
    assert {tuple(report["position"]) for report in reports} == {(1, 0)}
    assert {report["all_reduce_calls"] for report in reports} == {7}
    assert checkpoint["run_status"] == "interrupted"
    assert not list((tmp_path / "checkpoint").glob("*.candidate"))


@pytest.mark.integration
def test_rank0_completed_writer_failure_converges_without_replacing_canonical_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    checkpoint_dir = tmp_path / "checkpoint"
    atomic_torch_save(
        {"schema_version": 1, "run_status": "running", "marker": "canonical-latest"},
        checkpoint_dir / "latest.pt",
    )
    atomic_torch_save(
        {"schema_version": 1, "run_status": "running", "marker": "canonical-best"},
        checkpoint_dir / "best.pt",
    )
    world_size = 2
    rendezvous = str((tmp_path / "gloo-writer-failure-init").resolve())
    mp.spawn(
        _run_completion_write_failure_rank,
        args=(world_size, rendezvous, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    reports = [
        json.loads((tmp_path / f"writer-failure-rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]

    assert reports == [
        {"type": "OSError", "message": "injected completion writer failure"},
        {"type": "RuntimeError", "message": "completion checkpoint writer failed on rank zero"},
    ]
    assert not list(checkpoint_dir.glob("*.candidate"))
    assert load_checkpoint(checkpoint_dir / "latest.pt")["marker"] == "canonical-latest"
    assert load_checkpoint(checkpoint_dir / "best.pt")["marker"] == "canonical-best"


@pytest.mark.integration
def test_rank0_completion_promotion_failure_converges_all_ranks_without_hanging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    checkpoint_dir = tmp_path / "checkpoint"
    atomic_torch_save(
        {"schema_version": 1, "run_status": "running", "marker": "canonical-latest"},
        checkpoint_dir / "latest.pt",
    )
    atomic_torch_save(
        {"schema_version": 1, "run_status": "running", "marker": "canonical-best"},
        checkpoint_dir / "best.pt",
    )
    world_size = 2
    rendezvous = str((tmp_path / "gloo-promotion-failure-init").resolve())
    process_context = mp.spawn(
        _run_completion_promotion_failure_rank,
        args=(world_size, rendezvous, str(tmp_path)),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 10.0
    while not process_context.join(timeout=0.1):
        if time.monotonic() < deadline:
            continue
        for process in process_context.processes:
            process.terminate()
        for process in process_context.processes:
            process.join(timeout=1.0)
        pytest.fail("distributed completion promotion failure did not converge within 10 seconds")

    reports = [
        json.loads((tmp_path / f"promotion-failure-rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]

    assert reports == [
        {"type": "OSError", "message": "injected completion promotion failure"},
        {"type": "RuntimeError", "message": "completion checkpoint promotion failed on rank zero"},
    ]
    assert {path.name for path in checkpoint_dir.iterdir()} == {"latest.pt", "best.pt"}
    assert load_checkpoint(checkpoint_dir / "latest.pt")["marker"] == "canonical-latest"
    assert load_checkpoint(checkpoint_dir / "best.pt")["marker"] == "canonical-best"
