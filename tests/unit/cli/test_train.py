from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tardis.cli.common import parse_args
from tardis.data.assembly import RangeClientFactory, RemoteDataLoaderOptions, RemoteDataLoaders
from tardis.data.dataset import ClipBatch, ClipDecodeOptions
from tardis.data.splits import StablePartition
from tardis.models.tardis import TARDISTrainingBatch
from tardis.training.engine import ModelEMA, TrainEngineOptions
from tardis.utils.distributed import DistributedContext


def test_train_helper_import_does_not_load_weight_dependencies() -> None:
    script = """
import sys
import tardis.cli.train
print(any(name in sys.modules for name in ("diffusers", "transformers", "open_clip", "lpips")))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_train_parser_selects_exactly_one_dataset() -> None:
    assert parse_args([]).dataset == "dataverse"
    assert parse_args(["--dataset", "seedance"]).dataset == "seedance"

    with pytest.raises(SystemExit):
        parse_args(["--dataset", "unknown"])


def test_train_parser_exposes_keyframe_only_mode() -> None:
    assert parse_args([]).train_mode == "full_temporal"
    assert parse_args(["--train-mode", "keyframe_only"]).train_mode == "keyframe_only"

    with pytest.raises(SystemExit):
        parse_args(["--train-mode", "unknown"])


def test_amp_master_weights_promote_only_trainable_parameters_to_fp32() -> None:
    from tardis.cli.train import prepare_amp_master_weights

    model = nn.Module()
    model.frozen = nn.Linear(4, 4).half()
    model.trainable = nn.Linear(4, 4).half()
    model.frozen.requires_grad_(False)

    prepare_amp_master_weights(model)

    assert all(parameter.dtype == torch.float16 for parameter in model.frozen.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.trainable.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast requires a GPU")
def test_validation_generation_autocast_bridges_half_priors_and_fp32_temporal_weights() -> None:
    from tardis.cli.train import validation_generation_autocast

    model = nn.Module()
    model.priors = nn.Linear(4, 4).cuda().half().requires_grad_(False)
    model.temporal = nn.Linear(4, 4).cuda().float()
    latent = model.priors(torch.randn(2, 4, device="cuda", dtype=torch.float16))

    with validation_generation_autocast(model, torch.device("cuda")):
        output = model.temporal(latent)

    assert output.dtype == torch.float16


@pytest.mark.parametrize("total", [6, 7, 11, 12, 101, 640])
def test_curriculum_durations_prioritize_closed_loop_metric_training(total: int) -> None:
    from tardis.cli.train import curriculum_durations

    first = curriculum_durations(total)
    second = curriculum_durations(total)

    assert len(first) == 6
    assert all(duration > 0 for duration in first)
    assert sum(first) == total
    if total >= 12:
        assert first[-1] == max(first)
        assert sum(first[3:]) > sum(first[:3])
    assert first == second


def test_transport_only_curriculum_keeps_probe_in_stage_one() -> None:
    from tardis.cli.train import curriculum_durations_for_profile
    from tardis.training.curriculum import CurriculumSchedule, CurriculumStage

    durations = curriculum_durations_for_profile(96, "transport")
    schedule = CurriculumSchedule(durations=durations)

    assert schedule.at_step(0).stage is CurriculumStage.TRANSPORT_WARMUP
    assert schedule.at_step(95).stage is CurriculumStage.TRANSPORT_WARMUP


def test_metric_alignment_curriculum_unlocks_then_spends_probe_on_metrics() -> None:
    from tardis.cli.train import curriculum_durations_for_profile
    from tardis.training.curriculum import CurriculumSchedule, CurriculumStage

    durations = curriculum_durations_for_profile(32, "metric_alignment")
    schedule = CurriculumSchedule(durations=durations)

    assert durations == (1, 1, 1, 1, 1, 27)
    assert schedule.at_step(0).stage is CurriculumStage.TRANSPORT_WARMUP
    assert schedule.at_step(5).stage is CurriculumStage.METRIC_ALIGNMENT
    assert schedule.at_step(31).stage is CurriculumStage.METRIC_ALIGNMENT
    assert parse_args(["--curriculum-profile", "metric_alignment"]).curriculum_profile == (
        "metric_alignment"
    )


def test_closed_loop_motion_curriculum_spends_probe_on_generated_state() -> None:
    from tardis.cli.train import curriculum_durations_for_profile
    from tardis.training.curriculum import CurriculumSchedule, CurriculumStage

    durations = curriculum_durations_for_profile(32, "closed_loop_motion")
    schedule = CurriculumSchedule(durations=durations)

    assert durations == (1, 1, 1, 27, 1, 1)
    assert schedule.at_step(2).stage is CurriculumStage.RESIDUAL_TEACHER
    assert schedule.at_step(3).stage is CurriculumStage.CLOSED_LOOP
    assert schedule.at_step(29).stage is CurriculumStage.CLOSED_LOOP
    assert parse_args(["--curriculum-profile", "closed_loop_motion"]).curriculum_profile == (
        "closed_loop_motion"
    )


def test_all_training_loss_weights_are_cli_configurable() -> None:
    args = parse_args(
        [
            "--residual-loss-weight",
            "0.11",
            "--transport-loss-weight",
            "1.5",
            "--flow-loss-weight",
            "2.0",
            "--visibility-loss-weight",
            "3.0",
            "--lite-loss-weight",
            "0.05",
            "--diffusion-loss-weight",
            "0.0",
            "--router-loss-weight",
            "0.12",
            "--survival-loss-weight",
            "0.13",
            "--budget-loss-weight",
            "0.14",
            "--warp-loss-weight",
            "0.15",
            "--drift-loss-weight",
            "0.16",
            "--crcd-loss-weight",
            "0.17",
            "--text-loss-weight",
            "0.18",
            "--curriculum-profile",
            "transport",
        ]
    )

    assert args.residual_loss_weight == pytest.approx(0.11)
    assert args.transport_loss_weight == pytest.approx(1.5)
    assert args.flow_loss_weight == pytest.approx(2.0)
    assert args.visibility_loss_weight == pytest.approx(3.0)
    assert args.lite_loss_weight == pytest.approx(0.05)
    assert args.diffusion_loss_weight == pytest.approx(0.0)
    assert args.router_loss_weight == pytest.approx(0.12)
    assert args.survival_loss_weight == pytest.approx(0.13)
    assert args.budget_loss_weight == pytest.approx(0.14)
    assert args.warp_loss_weight == pytest.approx(0.15)
    assert args.drift_loss_weight == pytest.approx(0.16)
    assert args.crcd_loss_weight == pytest.approx(0.17)
    assert args.text_loss_weight == pytest.approx(0.18)
    assert args.curriculum_profile == "transport"


def test_training_metric_alignment_defaults_follow_competition_ratio() -> None:
    args = parse_args([])

    assert args.tc_loss_weight == pytest.approx(5.0)
    assert args.lpips_loss_weight == pytest.approx(3.0)
    assert args.lpips_frame_chunk_size == 4
    assert args.checkpoint_interval_steps == 256


def test_train_parser_rejects_nonpositive_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_steps"):
        parse_args(["--checkpoint-interval-steps", "0"])


def test_train_parser_exposes_weights_only_warm_start() -> None:
    args = parse_args(
        [
            "--warm-start",
            "/tmp/dataverse/run/best.pt",
            "--no-warm-start-use-ema",
        ]
    )

    assert args.warm_start == Path("/tmp/dataverse/run/best.pt")
    assert args.warm_start_use_ema is False


def test_train_parser_requires_explicit_cross_dataset_warm_start_opt_in() -> None:
    assert parse_args([]).allow_cross_dataset_warm_start is False
    assert (
        parse_args(["--allow-cross-dataset-warm-start"])
        .allow_cross_dataset_warm_start
        is True
    )


def test_train_parser_rejects_resume_and_warm_start_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(
            [
                "--resume",
                "/tmp/dataverse/run/latest.pt",
                "--warm-start",
                "/tmp/dataverse/parent/best.pt",
            ]
        )


@pytest.mark.parametrize("total", [-1, 0, 5])
def test_curriculum_durations_reject_fewer_than_six_optimizer_steps(total: int) -> None:
    from tardis.cli.train import curriculum_durations

    with pytest.raises(ValueError, match="at least 6"):
        curriculum_durations(total)


def test_train_engine_options_map_every_field_and_count_optimizer_steps() -> None:
    from tardis.cli.train import train_engine_options_from_args

    args = parse_args(
        [
            "--epochs",
            "3",
            "--steps-per-epoch",
            "8",
            "--gradient-accumulation-steps",
            "4",
            "--learning-rate",
            "0.002",
            "--weight-decay",
            "0.03",
            "--warmup-steps",
            "7",
            "--gradient-clip-norm",
            "1.5",
            "--precision",
            "fp16",
            "--ema-decay",
            "0.95",
        ]
    )

    options = train_engine_options_from_args(args)

    assert isinstance(options, TrainEngineOptions)
    assert asdict(options) == {
        "learning_rate": 0.002,
        "weight_decay": 0.03,
        "gradient_accumulation_steps": 4,
        "gradient_clip_norm": 1.5,
        "warmup_steps": 7,
        "total_optimizer_steps": 6,
        "precision": "fp16",
        "ema_decay": 0.95,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"epochs": 0}, "epochs"),
        ({"steps_per_epoch": 0}, "steps_per_epoch"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"steps_per_epoch": 10, "gradient_accumulation_steps": 4}, "divisible"),
    ],
)
def test_train_engine_options_fail_fast_without_cross_epoch_accumulation(
    overrides: dict[str, int],
    message: str,
) -> None:
    from tardis.cli.train import train_engine_options_from_args

    values: dict[str, object] = {
        "epochs": 3,
        "steps_per_epoch": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-2,
        "warmup_steps": 5,
        "gradient_clip_norm": 1.0,
        "precision": "bf16",
        "ema_decay": 0.999,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        train_engine_options_from_args(SimpleNamespace(**values))


def test_build_train_dataloaders_maps_local_data_options(tmp_path: Path) -> None:
    from tardis.cli.train import build_train_dataloaders

    dataset_paths = tuple(
        tmp_path / name
        for name in (
            "Vchitect_T2V_DataVerse",
            "OpenVid-1M",
            "seedance-2-prompts-datasets",
        )
    )
    for path in dataset_paths:
        path.mkdir()
    datasets_file = tmp_path / "datasets.txt"
    datasets_file.write_text(
        "\n".join(str(path) for path in dataset_paths),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--datasets-file",
            str(datasets_file),
            "--dataset",
            "openvid",
            "--height",
            "96",
            "--width",
            "128",
            "--num-frames",
            "12",
            "--steps-per-epoch",
            "8",
            "--micro-batch-size",
            "3",
            "--validation-batch-size",
            "5",
            "--validation-size",
            "17",
            "--test-size",
            "23",
            "--split-seed",
            "41",
            "--seed",
            "43",
            "--mirror-endpoint",
            "https://mirror.invalid",
            "--request-timeout-seconds",
            "12.5",
            "--max-retries",
            "6",
            "--num-workers",
            "2",
            "--prefetch-factor",
            "7",
            "--catalog-record-limit",
            "31",
            "--openvid-archive-limit",
            "2",
            "--dataverse-record-ids",
            "0000000300.mp4,0000000534.mp4",
        ]
    )
    context = DistributedContext(rank=1, local_rank=1, world_size=4, device=torch.device("cpu"))
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    result = build_train_dataloaders(args, context, builder=fake_builder)

    assert result is sentinel
    partition = captured["partition"]
    assert isinstance(partition, StablePartition)
    assert partition == StablePartition(seed=41, validation_size=17, test_size=23)
    train_clip = captured["train_clip_options"]
    benchmark_clip = captured["evaluation_clip_options"]
    assert isinstance(train_clip, ClipDecodeOptions)
    assert isinstance(benchmark_clip, ClipDecodeOptions)
    assert (train_clip.num_frames, train_clip.height, train_clip.width, train_clip.mode) == (
        12,
        96,
        128,
        "train",
    )
    assert train_clip.timeout_seconds == pytest.approx(12.5)
    assert train_clip.random_flip is True
    assert benchmark_clip.mode == "benchmark"
    assert benchmark_clip.timeout_seconds == pytest.approx(12.5)
    assert benchmark_clip.random_flip is False
    loader = captured["loader_options"]
    assert isinstance(loader, RemoteDataLoaderOptions)
    assert loader == RemoteDataLoaderOptions(
        steps_per_epoch=8,
        global_batch_size=12,
        evaluation_batch_size=5,
        gradient_accumulation_steps=2,
        seed=43,
        num_workers=2,
        prefetch_factor=7,
        max_sample_retries=6,
        multiprocessing_context="spawn",
    )
    client = captured["client_factory"]
    assert isinstance(client, RangeClientFactory)
    assert client.timeout_seconds == pytest.approx(12.5)
    assert client.max_retries == 6
    assert captured["rank"] == 1
    assert captured["world_size"] == 4
    assert captured["endpoint"] == "https://mirror.invalid"
    assert captured["dataset_roots"] == {
        "dataverse": dataset_paths[0].resolve(),
        "openvid": dataset_paths[1].resolve(),
        "seedance": dataset_paths[2].resolve(),
    }
    assert captured["catalog_record_limit"] == 31
    assert captured["openvid_archive_limit"] == 2
    assert captured["record_ids_by_source"] == {"dataverse": ("0000000300.mp4", "0000000534.mp4")}
    assert captured["selected_source"] == "openvid"


def test_keyframe_only_loader_decodes_one_training_frame_but_full_validation(
    tmp_path: Path,
) -> None:
    from tardis.cli.train import build_train_dataloaders

    dataset_paths = tuple(
        tmp_path / name
        for name in (
            "Vchitect_T2V_DataVerse",
            "OpenVid-1M",
            "seedance-2-prompts-datasets",
        )
    )
    for path in dataset_paths:
        path.mkdir()
    datasets_file = tmp_path / "datasets.txt"
    datasets_file.write_text(
        "\n".join(str(path) for path in dataset_paths),
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--datasets-file",
            str(datasets_file),
            "--train-mode",
            "keyframe_only",
            "--num-frames",
            "16",
        ]
    )
    captured: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    build_train_dataloaders(
        args,
        DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu")),
        builder=fake_builder,
    )

    train_clip = captured["train_clip_options"]
    validation_clip = captured["evaluation_clip_options"]
    assert isinstance(train_clip, ClipDecodeOptions)
    assert isinstance(validation_clip, ClipDecodeOptions)
    assert train_clip.num_frames == 1
    assert validation_clip.num_frames == 16


def test_clip_batch_to_training_batch_preserves_prompts_and_uses_non_blocking_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tardis.cli.train import clip_batch_to_training_batch

    prompts = ["first", "second"]
    batch = ClipBatch(
        prompts=prompts,
        video=torch.zeros(2, 3, 3, 4, 4),
        sources=("dataverse", "openvid"),
        record_ids=("record-a", "record-b"),
        sample_seeds=(11, 13),
    )
    original_to = torch.Tensor.to
    calls: list[dict[str, object]] = []

    def tracked_to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(dict(kwargs))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)

    converted, record_ids = clip_batch_to_training_batch(batch, torch.device("cpu"))

    assert isinstance(converted, TARDISTrainingBatch)
    assert converted.prompts is prompts
    assert converted.video.shape == batch.video.shape
    assert calls == [{"device": torch.device("cpu"), "non_blocking": True}]
    assert record_ids is batch.record_ids


class _TemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.priors = nn.Linear(2, 2, bias=False)
        self.temporal = nn.Linear(2, 2, bias=False)
        self.priors.requires_grad_(False)
        self.priors.weight.data.fill_(5.0)
        self.temporal.weight.data.fill_(1.0)


def test_ema_temporal_parameters_overlays_only_trainable_state_and_restores_on_exception() -> None:
    from tardis.cli.train import ema_temporal_parameters

    model = _TemporalModel()
    ema = ModelEMA(model, decay=0.9)
    ema.shadow["temporal.weight"].fill_(7.0)
    original_prior = model.priors.weight.detach().clone()
    original_temporal = model.temporal.weight.detach().clone()

    with (
        pytest.raises(RuntimeError, match="validation failed"),
        ema_temporal_parameters(model, ema),
    ):
        assert torch.equal(model.temporal.weight, torch.full_like(model.temporal.weight, 7.0))
        assert torch.equal(model.priors.weight, original_prior)
        raise RuntimeError("validation failed")

    assert torch.equal(model.temporal.weight, original_temporal)
    assert torch.equal(model.priors.weight, original_prior)


def test_ema_temporal_parameters_restores_student_after_normal_exit() -> None:
    from tardis.cli.train import ema_temporal_parameters

    model = _TemporalModel()
    ema = ModelEMA(model, decay=0.9)
    ema.shadow["temporal.weight"].fill_(7.0)
    original_prior = model.priors.weight.detach().clone()
    original_temporal = model.temporal.weight.detach().clone()

    with ema_temporal_parameters(model, ema):
        assert torch.equal(model.temporal.weight, torch.full_like(model.temporal.weight, 7.0))
        assert torch.equal(model.priors.weight, original_prior)

    assert torch.equal(model.temporal.weight, original_temporal)
    assert torch.equal(model.priors.weight, original_prior)


@pytest.mark.parametrize("corruption", ["missing", "shape", "prior"])
def test_ema_temporal_parameters_validates_all_shadow_state_before_writing(
    corruption: str,
) -> None:
    from tardis.cli.train import ema_temporal_parameters

    model = _TemporalModel()
    ema = ModelEMA(model, decay=0.9)
    ema.shadow["temporal.weight"].fill_(7.0)
    if corruption == "missing":
        del ema.shadow["temporal.weight"]
    elif corruption == "shape":
        ema.shadow["temporal.weight"] = torch.ones(1)
    else:
        ema.shadow["priors.weight"] = torch.ones_like(model.priors.weight)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with pytest.raises(ValueError, match="EMA"), ema_temporal_parameters(model, ema):
        pytest.fail("invalid EMA state must fail before entering the context")

    assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())


class _ValidationModel(nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.calls: list[tuple[list[str], int, int, int, bool]] = []

    def generate(
        self,
        prompts: list[str],
        num_frames: int,
        fps: int,
        generator: torch.Generator,
    ) -> SimpleNamespace:
        self.calls.append((list(prompts), num_frames, fps, generator.initial_seed(), self.training))
        if self.fail:
            raise RuntimeError("generation failed")
        video = torch.full((1, num_frames, 3, 2, 2), float(generator.initial_seed() % 17))
        return SimpleNamespace(video=video)


class _MetricSuite:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.updates: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        self.compute_count = 0

    def update(self, generated: torch.Tensor, reference: torch.Tensor, prompt: str) -> None:
        self.events.append(("update", prompt))
        self.updates.append((generated.clone(), reference.clone(), prompt))

    def all_reduce(self) -> None:
        self.events.append("all_reduce")

    def compute(self) -> dict[str, dict[str, float]]:
        self.events.append("compute")
        self.compute_count += 1
        return {"macro": {"score": float(self.compute_count)}, "micro": {"score": -1.0}}

    def reset(self) -> None:
        self.events.append("reset")


@dataclass(frozen=True)
class _BenchmarkRecord:
    id: str


class _BenchmarkDataset:
    def __init__(self, *, split: str, source: str, record_ids: tuple[str, ...]) -> None:
        self.split = split
        self.source = source
        self.records = tuple(_BenchmarkRecord(record_id) for record_id in record_ids)
        self.rank = 0
        self.world_size = 1

    def __len__(self) -> int:
        return len(self.records)


class _ObservedLoader:
    def __init__(
        self,
        *,
        split: str,
        source: str,
        record_ids: tuple[str, ...],
        batches: list[ClipBatch],
    ) -> None:
        self.dataset = _BenchmarkDataset(
            split=split,
            source=source,
            record_ids=record_ids,
        )
        self.batches = batches
        self.iterations = 0

    def __iter__(self) -> Iterator[ClipBatch]:
        self.iterations += 1
        yield from self.batches


def _validation_batches() -> dict[str, list[ClipBatch]]:
    return {
        "dataverse": [
            ClipBatch(
                prompts=["d0", "d1"],
                video=torch.stack((torch.zeros(2, 3, 2, 2), torch.ones(2, 3, 2, 2))),
                sources=("dataverse", "dataverse"),
                record_ids=("d-0", "d-1"),
                sample_seeds=(101, 103),
            )
        ],
        "openvid": [
            ClipBatch(
                prompts=["o0"],
                video=torch.full((1, 2, 3, 2, 2), 2.0),
                sources=("openvid",),
                record_ids=("o-0",),
                sample_seeds=(107,),
            )
        ],
        "seedance": [
            ClipBatch(
                prompts=["s0"],
                video=torch.full((1, 2, 3, 2, 2), 3.0),
                sources=("seedance",),
                record_ids=("s-0",),
                sample_seeds=(109,),
            )
        ],
    }


def _remote_loader_bundle(source: str = "dataverse") -> RemoteDataLoaders:
    validation_batches = _validation_batches()
    validation_ids = {
        source: tuple(record_id for batch in batches for record_id in batch.record_ids)
        for source, batches in validation_batches.items()
    }
    validation = {
        source: _ObservedLoader(
            split="validation",
            source=source,
            record_ids=validation_ids[source],
            batches=validation_batches[source],
        )
    }
    test = {
        source: _ObservedLoader(
            split="test",
            source=source,
            record_ids=(f"{source}-test-0",),
            batches=[],
        )
    }
    return RemoteDataLoaders(
        catalog=object(),
        splits={
            "train": {source: () for source in validation},
            "validation": {source: loader.dataset.records for source, loader in validation.items()},
            "test": {source: loader.dataset.records for source, loader in test.items()},
        },
        train_dataset=object(),
        train=object(),
        validation=validation,
        test=test,
        selected_source=source,
    )


def test_evaluate_validation_uses_each_record_seed_in_one_dataset_metric_cycle() -> None:
    from tardis.cli.train import evaluate_validation

    model = _ValidationModel().train()
    suite = _MetricSuite()
    loaders = _remote_loader_bundle()
    progress_updates: list[tuple[int, str]] = []

    class Progress:
        def update_validation(self, records: int, source: str) -> None:
            progress_updates.append((records, source))

    result = evaluate_validation(
        model,
        loaders,
        suite,
        fps=8,
        device=torch.device("cpu"),
        seed=3407,
        progress=Progress(),
    )

    assert model.training is True
    assert model.calls == [
        (["d0"], 2, 8, 101, False),
        (["d1"], 2, 8, 103, False),
    ]
    assert [prompt for _, _, prompt in suite.updates] == ["d0", "d1"]
    assert [reference.mean().item() for _, reference, _ in suite.updates] == [0.0, 1.0]
    assert all(generated.ndim == reference.ndim == 4 for generated, reference, _ in suite.updates)
    assert suite.events == [
        ("update", "d0"),
        ("update", "d1"),
        "all_reduce",
        "compute",
        "reset",
    ]
    assert result == {"dataverse_validation": {"score": 1.0}}
    assert all(loader.iterations == 0 for loader in loaders.test.values())
    assert progress_updates == [(2, "dataverse")]


def test_evaluate_validation_rejects_a_bare_source_mapping() -> None:
    from tardis.cli.train import evaluate_validation

    with pytest.raises(TypeError, match="RemoteDataLoaders"):
        evaluate_validation(
            _ValidationModel(),
            _validation_batches(),
            _MetricSuite(),
            fps=8,
            device="cpu",
            seed=1,
        )


def test_evaluate_validation_rejects_a_real_test_split_in_validation_bundle() -> None:
    from tardis.cli.train import evaluate_validation

    loaders = _remote_loader_bundle()
    loaders.validation["dataverse"] = loaders.test["dataverse"]

    with pytest.raises(ValueError, match="split.*validation"):
        evaluate_validation(
            _ValidationModel(),
            loaders,
            _MetricSuite(),
            fps=8,
            device="cpu",
            seed=1,
        )

    assert all(loader.iterations == 0 for loader in loaders.test.values())


@pytest.mark.parametrize(("batch_count", "observed"), [(0, 0), (1, 1), (3, 3)])
def test_evaluate_validation_rejects_incomplete_local_validation_shards(
    batch_count: int,
    observed: int,
) -> None:
    from tardis.cli.train import evaluate_validation

    loaders = _remote_loader_bundle()
    records = _validation_batches()["dataverse"][0]
    single_record = ClipBatch(
        prompts=[records.prompts[0]],
        video=records.video[:1],
        sources=(records.sources[0],),
        record_ids=(records.record_ids[0],),
        sample_seeds=(records.sample_seeds[0],),
    )
    loaders.validation["dataverse"] = _ObservedLoader(
        split="validation",
        source="dataverse",
        record_ids=("d-0", "d-1"),
        batches=[single_record] * batch_count,
    )

    with pytest.raises(ValueError, match=rf"expected 2.*observed {observed}"):
        evaluate_validation(
            _ValidationModel(),
            loaders,
            _MetricSuite(),
            fps=8,
            device="cpu",
            seed=1,
        )


def test_evaluate_validation_rejects_repeated_ids_at_the_expected_shard_size() -> None:
    from tardis.cli.train import evaluate_validation

    loaders = _remote_loader_bundle()
    records = _validation_batches()["dataverse"][0]
    repeated_record = ClipBatch(
        prompts=[records.prompts[0]],
        video=records.video[:1],
        sources=(records.sources[0],),
        record_ids=(records.record_ids[0],),
        sample_seeds=(records.sample_seeds[0],),
    )
    loaders.validation["dataverse"] = _ObservedLoader(
        split="validation",
        source="dataverse",
        record_ids=("d-0", "d-1"),
        batches=[repeated_record, repeated_record],
    )

    with pytest.raises(ValueError, match="repeated record IDs.*d-0"):
        evaluate_validation(
            _ValidationModel(),
            loaders,
            _MetricSuite(),
            fps=8,
            device="cpu",
            seed=1,
        )


def test_evaluate_validation_rejects_repeated_ids_inside_one_batch() -> None:
    from tardis.cli.train import evaluate_validation

    loaders = _remote_loader_bundle()
    records = _validation_batches()["dataverse"][0]
    loaders.validation["dataverse"] = _ObservedLoader(
        split="validation",
        source="dataverse",
        record_ids=("d-0", "d-1"),
        batches=[
            ClipBatch(
                prompts=[records.prompts[0], records.prompts[0]],
                video=records.video,
                sources=("dataverse", "dataverse"),
                record_ids=("d-0", "d-0"),
                sample_seeds=(records.sample_seeds[0], records.sample_seeds[0]),
            )
        ],
    )

    with pytest.raises(ValueError, match="repeated record IDs.*d-0"):
        evaluate_validation(
            _ValidationModel(),
            loaders,
            _MetricSuite(),
            fps=8,
            device="cpu",
            seed=1,
        )


def test_evaluate_validation_rejects_test_records_relabelled_as_validation() -> None:
    from tardis.cli.train import evaluate_validation

    loaders = _remote_loader_bundle()
    test_records = loaders.test["dataverse"].dataset.records
    loaders.validation["dataverse"] = _ObservedLoader(
        split="validation",
        source="dataverse",
        record_ids=tuple(record.id for record in test_records),
        batches=[],
    )

    with pytest.raises(ValueError, match="canonical validation records"):
        evaluate_validation(
            _ValidationModel(),
            loaders,
            _MetricSuite(),
            fps=8,
            device="cpu",
            seed=1,
        )


def test_evaluate_validation_restores_training_state_after_generation_error() -> None:
    from tardis.cli.train import evaluate_validation

    model = _ValidationModel(fail=True).train()
    suite = _MetricSuite()

    with pytest.raises(RuntimeError, match="generation failed"):
        evaluate_validation(
            model,
            _remote_loader_bundle(),
            suite,
            fps=8,
            device="cpu",
            seed=3407,
        )

    assert model.training is True
    assert suite.events == ["reset"]


def test_evaluate_validation_cancels_at_record_boundary_and_discards_partial_metrics() -> None:
    from tardis.cli.train import CooperativeStop, evaluate_validation

    stop = CooperativeStop()

    class StopAfterFirstRecord(_ValidationModel):
        def generate(
            self,
            prompts: list[str],
            num_frames: int,
            fps: int,
            generator: torch.Generator,
        ) -> SimpleNamespace:
            result = super().generate(prompts, num_frames, fps, generator)
            stop.request("SIGTERM")
            return result

    model = StopAfterFirstRecord().train()
    suite = _MetricSuite()

    with pytest.raises(RuntimeError, match="validation.*interrupt"):
        evaluate_validation(
            model,
            _remote_loader_bundle(),
            suite,
            fps=8,
            device="cpu",
            seed=3407,
            stop=stop,
            context=DistributedContext(
                rank=0,
                local_rank=0,
                world_size=1,
                device=torch.device("cpu"),
            ),
        )

    assert model.training is True
    assert [call[0] for call in model.calls] == [["d0"]]
    assert suite.events == [("update", "d0"), "reset"]


def test_train_module_exports_only_typed_primitives() -> None:
    import tardis.cli.train as train

    assert train.__all__ == [
        "build_train_dataloaders",
        "clip_batch_to_training_batch",
        "curriculum_durations",
        "ema_temporal_parameters",
        "evaluate_validation",
        "train_engine_options_from_args",
    ]
    assert all(callable(getattr(train, name)) for name in train.__all__)
