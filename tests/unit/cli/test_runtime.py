from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tardis.cli.common import ModelOptions, parse_args
from tardis.models.factory import TARDISFactoryOptions, tardis_temporal_state_dict
from tardis.utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, atomic_torch_save
from tardis.utils.defaults import DEFAULT_TARDIS_ARCHITECTURE
from tests.helpers.tardis_model import build_tiny_tardis


def _local_dataset_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (
        tmp_path / "Vchitect_T2V_DataVerse",
        tmp_path / "OpenVid-1M",
        tmp_path / "seedance-2-prompts-datasets",
    )
    for path in paths:
        path.mkdir()
    return paths


def _local_datasets_file(tmp_path: Path) -> Path:
    paths = _local_dataset_dirs(tmp_path)
    source_file = tmp_path / "datasets.txt"
    source_file.write_text("\n".join(str(path) for path in paths), encoding="utf-8")
    return source_file


def test_runtime_import_does_not_eagerly_import_model_weight_dependencies() -> None:
    script = """
import sys
import tardis.cli.runtime
print(any(name in sys.modules for name in (\"diffusers\", \"transformers\", \"open_clip\")))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_common_build_parser_does_not_import_torch_or_weight_dependencies() -> None:
    script = """
import sys
from tardis.cli.common import build_parser
build_parser()
print(any(name in sys.modules for name in ("torch", "diffusers", "transformers", "open_clip")))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("precision", "device", "expected"),
    [
        ("fp32", "cpu", torch.float32),
        ("fp16", "cpu", torch.float32),
        ("bf16", "cpu", torch.float32),
    ],
)
def test_resolve_torch_dtype_has_stable_cpu_fallback(
    precision: str,
    device: str,
    expected: torch.dtype,
) -> None:
    from tardis.cli.runtime import resolve_torch_dtype

    assert resolve_torch_dtype(precision, device) is expected


def test_resolve_torch_dtype_uses_supported_cuda_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    from tardis.cli.runtime import resolve_torch_dtype

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert resolve_torch_dtype("fp16", "cuda:0") is torch.float16
    assert resolve_torch_dtype("bf16", "cuda:0") is torch.bfloat16


def test_dataset_sources_are_exactly_the_three_local_directories(tmp_path: Path) -> None:
    from tardis.cli.runtime import read_dataset_sources

    dataverse, openvid, seedance = _local_dataset_dirs(tmp_path)
    source_file = tmp_path / "datasets.txt"
    source_file.write_text(
        "\n".join(
            [
                "# local authoritative sources",
                f" {dataverse} ",
                str(openvid),
                "",
                str(seedance),
            ]
        ),
        encoding="utf-8",
    )

    sources = read_dataset_sources(source_file, "https://hf-mirror.com")

    assert isinstance(sources, tuple)
    assert sources == (
        str(dataverse.resolve()),
        str(openvid.resolve()),
        str(seedance.resolve()),
    )


def test_dataset_sources_reject_incomplete_local_catalog(tmp_path: Path) -> None:
    from tardis.cli.runtime import read_dataset_sources

    dataverse, _, _ = _local_dataset_dirs(tmp_path)
    source_file = tmp_path / "datasets.txt"
    source_file.write_text(f"{dataverse}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly three"):
        read_dataset_sources(source_file, "https://hf-mirror.com")


def test_dataset_sources_reject_noncanonical_or_missing_local_directory(tmp_path: Path) -> None:
    from tardis.cli.runtime import read_dataset_sources

    dataverse, openvid, _ = _local_dataset_dirs(tmp_path)
    wrong = tmp_path / "seedance"
    wrong.mkdir()
    source_file = tmp_path / "datasets.txt"
    source_file.write_text(
        "\n".join((str(dataverse), str(openvid), str(wrong))),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical names"):
        read_dataset_sources(source_file, "https://hf-mirror.com")


def test_factory_options_map_shared_cli_architecture_fields() -> None:
    from tardis.cli.runtime import factory_options_from_args

    args = parse_args(
        [
            "--height",
            "256",
            "--width",
            "320",
            "--latent-channels",
            "3",
            "--patch-size",
            "4",
            "--hidden-size",
            "64",
            "--num-layers",
            "2",
            "--num-heads",
            "8",
            "--active-ratio",
            "0.5",
            "--motion-max-flow-pixels",
            "3.5",
            "--transport-max-correction-pixels",
            "0.75",
            "--transport-history-fallback-weight",
            "0.8",
            "--router-threshold",
            "0.2",
            "--router-halo-radius",
            "2",
            "--state-anchor-decay",
            "0.9",
            "--scene-cut-threshold",
            "0.85",
            "--oracle-temperature",
            "0.4",
            "--training-noise-scale",
            "0.05",
            "--lite-max-magnitude",
            "0.025",
            "--keyframe-lite-alignment",
            "--sampler-trajectory-alignment",
        ]
    )

    options = factory_options_from_args(args)

    assert options.height == 256
    assert options.width == 320
    assert options.motion_noise_channels == 3
    assert options.patch_size == 4
    assert options.residual_hidden_size == 64
    assert options.residual_layers == 2
    assert options.residual_heads == 8
    assert options.active_ratio == pytest.approx(0.5)
    assert options.motion_max_flow_pixels == pytest.approx(3.5)
    assert options.transport_max_correction_pixels == pytest.approx(0.75)
    assert options.transport_history_fallback_weight == pytest.approx(0.8)
    assert options.router_threshold == pytest.approx(0.2)
    assert options.router_halo_radius == 2
    assert options.state_anchor_decay == pytest.approx(0.9)
    assert options.scene_cut_threshold == pytest.approx(0.85)
    assert options.oracle_temperature == pytest.approx(0.4)
    assert options.training_noise_scale == pytest.approx(0.05)
    assert options.lite_max_magnitude == pytest.approx(0.025)
    assert options.keyframe_lite_alignment is True
    assert options.sampler_trajectory_alignment is True
    assert options.gradient_checkpointing is True


def test_discover_checkpoint_prefers_newest_timestamped_best_checkpoint(tmp_path: Path) -> None:
    from tardis.cli.runtime import discover_checkpoint

    older = tmp_path / "20260801_235959_000000_old" / "best.pt"
    newer = tmp_path / "20260802_000001_000000_new" / "best.pt"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.touch()
    newer.touch()

    assert discover_checkpoint(None, tmp_path) == newer
    assert discover_checkpoint(older, tmp_path) == older


def test_discover_checkpoint_reports_missing_explicit_or_latest_checkpoint(tmp_path: Path) -> None:
    from tardis.cli.runtime import discover_checkpoint

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        discover_checkpoint(tmp_path / "missing.pt", tmp_path)
    with pytest.raises(FileNotFoundError, match="best.pt"):
        discover_checkpoint(None, tmp_path)


def test_dataset_checkpoint_root_isolates_automatic_weight_discovery(tmp_path: Path) -> None:
    from tardis.cli.runtime import dataset_checkpoint_root, discover_checkpoint

    dataverse = tmp_path / "dataverse" / "20260808_010101_000001"
    openvid = tmp_path / "openvid" / "20260808_020202_000002"
    dataverse.mkdir(parents=True)
    openvid.mkdir(parents=True)
    (dataverse / "best.pt").touch()
    (openvid / "best.pt").touch()

    assert discover_checkpoint(None, dataset_checkpoint_root(tmp_path, "dataverse")) == (
        dataverse / "best.pt"
    )
    assert discover_checkpoint(None, dataset_checkpoint_root(tmp_path, "openvid")) == (
        openvid / "best.pt"
    )

    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_checkpoint_root(tmp_path, "unknown")


def test_load_temporal_checkpoint_applies_ema_only_to_trainable_temporal_parameters(
    tmp_path: Path,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    source = build_tiny_tardis().model
    target = build_tiny_tardis().model
    model_state = tardis_temporal_state_dict(source)
    target_priors = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if name.startswith("priors.")
    }
    shadow = {
        name: parameter.detach().clone().add(2)
        for name, parameter in target.named_parameters()
        if parameter.requires_grad and not name.startswith("priors.")
    }
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": model_state,
            "ema": {"decay": 0.9, "shadow": shadow},
        },
        checkpoint,
    )

    loaded = load_temporal_checkpoint(target, checkpoint, use_ema=True)

    assert loaded.path == checkpoint
    assert loaded.used_ema is True
    for name, parameter in target.named_parameters():
        if name in shadow:
            assert torch.equal(parameter, shadow[name])
    for name, value in target.state_dict().items():
        if name.startswith("priors."):
            assert torch.equal(value, target_priors[name])


def test_load_temporal_checkpoint_migrates_only_legacy_keyframe_text_parameters(
    tmp_path: Path,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    source = build_tiny_tardis().model
    target = build_tiny_tardis().model
    state = tardis_temporal_state_dict(source)
    del state["lite_corrector.text_projection.weight"]
    del state["lite_corrector.text_projection.bias"]
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model": state},
        checkpoint,
    )

    load_temporal_checkpoint(target, checkpoint)

    assert torch.count_nonzero(target.lite_corrector.text_projection.weight) == 0
    assert torch.count_nonzero(target.lite_corrector.text_projection.bias) == 0


def test_load_temporal_checkpoint_does_not_silently_accept_arbitrary_missing_keys(
    tmp_path: Path,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    target = build_tiny_tardis().model
    state = tardis_temporal_state_dict(target)
    del state["motion_scaffold.condition.0.weight"]
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model": state},
        checkpoint,
    )

    with pytest.raises(ValueError, match="key mismatch"):
        load_temporal_checkpoint(target, checkpoint)


def test_load_temporal_checkpoint_rejects_prior_state_in_any_temporal_payload(
    tmp_path: Path,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    target = build_tiny_tardis().model
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": {"priors.codec_module.weight": torch.ones(1)},
            "ema": {"shadow": {}},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="frozen prior"):
        load_temporal_checkpoint(target, checkpoint)


@pytest.mark.parametrize("use_ema", [False, True])
def test_load_temporal_checkpoint_rejects_ema_prior_state_in_both_modes(
    tmp_path: Path,
    use_ema: bool,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    target = build_tiny_tardis().model
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": tardis_temporal_state_dict(target),
            "ema": {"shadow": {"priors.codec_module.encoder.weight": torch.ones(1)}},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="frozen prior"):
        load_temporal_checkpoint(target, checkpoint, use_ema=use_ema)


def _model_state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_model_state_equal(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    assert model.state_dict().keys() == snapshot.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, snapshot[name]), name


def test_load_temporal_checkpoint_is_atomic_when_model_state_shape_is_invalid(
    tmp_path: Path,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    target = build_tiny_tardis().model
    before = _model_state_snapshot(target)
    state = tardis_temporal_state_dict(target)
    key = "residual_dit.blocks.0.modulation_projection.weight"
    state[key] = state[key][:-1].clone()
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model": state},
        checkpoint,
    )

    with pytest.raises((ValueError, RuntimeError), match="shape|size mismatch"):
        load_temporal_checkpoint(target, checkpoint)

    _assert_model_state_equal(target, before)


def test_load_temporal_checkpoint_is_atomic_when_late_ema_shape_is_invalid(
    tmp_path: Path,
) -> None:
    from tardis.cli.runtime import load_temporal_checkpoint

    target = build_tiny_tardis().model
    before = _model_state_snapshot(target)
    shadow = {
        name: parameter.detach().clone().add(2)
        for name, parameter in target.named_parameters()
        if parameter.requires_grad and not name.startswith("priors.")
    }
    key = "residual_dit.blocks.0.modulation_projection.weight"
    shadow[key] = shadow[key][:-1].clone()
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": tardis_temporal_state_dict(target),
            "ema": {"shadow": shadow},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="shape"):
        load_temporal_checkpoint(target, checkpoint, use_ema=True)

    _assert_model_state_equal(target, before)


def test_load_temporal_checkpoint_rolls_back_partial_loader_failure_with_cpu_temporal_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tardis.cli.runtime as runtime

    target = build_tiny_tardis().model
    before = _model_state_snapshot(target)
    state = tardis_temporal_state_dict(target)
    changed_key = "motion_scaffold.condition.0.weight"
    state[changed_key] = state[changed_key].add(3)
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model": state},
        checkpoint,
    )
    captured_snapshot: dict[str, torch.Tensor] = {}
    real_snapshot = runtime._model_state_snapshot

    def capture_snapshot(model: object) -> dict[str, torch.Tensor]:
        snapshot = real_snapshot(model)
        captured_snapshot.update(snapshot)
        return snapshot

    def fail_after_one_write(
        model: torch.nn.Module,
        prepared: dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            model.state_dict()[changed_key].copy_(prepared[changed_key])
        raise RuntimeError("injected temporal load failure")

    monkeypatch.setattr(runtime, "_model_state_snapshot", capture_snapshot)
    monkeypatch.setattr(runtime, "load_tardis_temporal_state_dict", fail_after_one_write)

    with pytest.raises(RuntimeError, match="injected temporal load failure"):
        runtime.load_temporal_checkpoint(target, checkpoint)

    assert captured_snapshot
    assert all(not name.startswith("priors.") for name in captured_snapshot)
    assert all(value.device.type == "cpu" for value in captured_snapshot.values())
    _assert_model_state_equal(target, before)


def test_factory_options_from_args_uses_factory_defaults_for_omitted_architecture_values() -> None:
    from tardis.cli.runtime import factory_options_from_args

    defaults = TARDISFactoryOptions()
    options = factory_options_from_args(SimpleNamespace(height=64, width=80, latent_channels=3))

    assert options.height == 64
    assert options.width == 80
    assert options.motion_noise_channels == 3
    assert options.state_channels == defaults.state_channels
    assert options.residual_hidden_size == defaults.residual_hidden_size
    assert options.residual_layers == defaults.residual_layers
    assert options.residual_heads == defaults.residual_heads
    assert options.diffusion_time_sampling == defaults.diffusion_time_sampling
    assert options.sampler_trajectory_alignment == defaults.sampler_trajectory_alignment


def test_common_model_architecture_defaults_match_lightweight_shared_defaults() -> None:
    model = ModelOptions()
    model_fields = {field.name for field in fields(ModelOptions)}

    assert {"height", "width", "latent_channels"}.issubset(model_fields)
    assert model.height == DEFAULT_TARDIS_ARCHITECTURE.height
    assert model.width == DEFAULT_TARDIS_ARCHITECTURE.width
    assert model.latent_channels == DEFAULT_TARDIS_ARCHITECTURE.motion_noise_channels
    assert model.patch_size == DEFAULT_TARDIS_ARCHITECTURE.patch_size
    assert model.hidden_size == DEFAULT_TARDIS_ARCHITECTURE.residual_hidden_size
    assert model.num_layers == DEFAULT_TARDIS_ARCHITECTURE.residual_layers
    assert model.num_heads == DEFAULT_TARDIS_ARCHITECTURE.residual_heads
    assert model.active_ratio == DEFAULT_TARDIS_ARCHITECTURE.active_ratio
    assert model.diffusion_time_sampling == DEFAULT_TARDIS_ARCHITECTURE.diffusion_time_sampling


def test_delivery_architecture_defaults_match_frozen_checkpoint_profile() -> None:
    model = ModelOptions()

    assert model.transport_history_fallback_weight == 1.0
    assert model.lite_max_magnitude == 0.75
    assert model.keyframe_lite_alignment is True
    assert model.diffusion_time_sampling == "endpoint"
    assert model.sampler_trajectory_alignment is True


def test_cli_accepts_deployment_aligned_diffusion_time_sampling() -> None:
    args = parse_args(["--diffusion-time-sampling", "endpoint"])

    assert args.diffusion_time_sampling == "endpoint"


def test_cli_accepts_sampler_trajectory_alignment() -> None:
    args = parse_args(
        [
            "--diffusion-time-sampling",
            "endpoint",
            "--sampler-trajectory-alignment",
        ]
    )

    assert args.sampler_trajectory_alignment is True


@pytest.mark.parametrize("entrypoint", ("infer", "apply"))
def test_generation_cli_accepts_sampler_trajectory_alignment(entrypoint: str) -> None:
    if entrypoint == "infer":
        from tardis.cli.infer import parse_args as parse_generation_args
    else:
        from tardis.cli.apply import parse_args as parse_generation_args

    args = parse_generation_args(
        [
            "--diffusion-time-sampling",
            "endpoint",
            "--sampler-trajectory-alignment",
        ]
    )

    assert args.sampler_trajectory_alignment is True


def test_all_formal_shell_entries_forward_sampler_training_contract() -> None:
    for script_name in ("train.sh", "infer.sh", "apply.sh"):
        script = (Path("scripts") / script_name).read_text(encoding="utf-8")
        assert (
            '--diffusion-time-sampling "${TARDIS_DIFFUSION_TIME_SAMPLING:-endpoint}"'
            in script
        )
        assert 'TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT:-1' in script
        assert 'TARDIS_KEYFRAME_LITE_ALIGNMENT:-1' in script
        assert 'TARDIS_TRANSPORT_HISTORY_FALLBACK_WEIGHT:-1.0' in script
        assert 'TARDIS_LITE_MAX_MAGNITUDE:-0.75' in script


def test_ci_installs_project_dependencies_before_importing_torch_tests() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m pip install -e '.[dev]'" in workflow
    assert "--no-deps" not in workflow


def test_pytest_uses_importlib_mode_for_duplicate_test_basenames() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "--import-mode=importlib" in config["tool"]["pytest"]["ini_options"]["addopts"]


class _FakeLPIPS:
    provenance_id = "test/lpips"

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3))


class _FakeVideoFeature:
    feature_dim = 2
    provenance_id = "test/video"

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        mean = video.mean()
        return torch.stack((mean, mean.square())).reshape(1, 2)


class _FakeClip:
    feature_dim = 2
    provenance_id = "test/clip"

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        mean = video.mean(dim=(1, 2, 3))
        return torch.stack((torch.ones_like(mean), mean), dim=1)

    def encode_text(self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor([[1.0, 0.0]])


def test_build_metric_suite_injects_adapters_and_preserves_all_provenance() -> None:
    from tardis.cli.runtime import MetricFeatureAdapters, build_metric_suite
    from tardis.metrics.paired import SSIMMetric, TemporalConsistencyMetric

    suite = build_metric_suite(
        device="cpu",
        feature_adapters=MetricFeatureAdapters(
            lpips=_FakeLPIPS(),
            fid=_FakeVideoFeature(),
            fvd=_FakeVideoFeature(),
            clipscore=_FakeClip(),
            tc=TemporalConsistencyMetric(),
            ssim=SSIMMetric(),
        ),
    )

    assert suite.lpips.feature is not None
    assert suite.lpips.provenance_id == "test/lpips"
    assert suite.fid.provenance_id == "test/video"
    assert suite.fvd.provenance_id == "test/video"
    assert suite.clipscore.provenance_id == "test/clip"
    assert set(suite.provenance_ids) == {"tc", "lpips", "fvd", "fid", "clipscore", "ssim"}


def test_build_metric_suite_rejects_mapping_adapter_injection() -> None:
    from tardis.cli.runtime import build_metric_suite

    with pytest.raises(TypeError, match="MetricFeatureAdapters"):
        build_metric_suite(device="cpu", feature_adapters={})


def test_build_production_runtime_assembles_typed_shared_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tardis.cli.runtime as runtime

    datasets_file = _local_datasets_file(tmp_path)
    args = parse_args(
        [
            "--datasets-file",
            str(datasets_file),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--checkpoint-root",
            str(tmp_path),
        ]
    )
    assembly = build_tiny_tardis()
    calls: dict[str, object] = {}

    def fake_factory(**kwargs: object) -> object:
        calls.update(kwargs)
        return assembly.model

    monkeypatch.setattr(runtime, "build_production_tardis", fake_factory)
    monkeypatch.setattr(runtime, "load_temporal_checkpoint", lambda *args, **kwargs: None)

    shared = runtime.build_production_runtime(
        args,
        restore_checkpoint=False,
        motion_teacher=assembly.motion_teacher,
        feature_adapters=runtime.MetricFeatureAdapters(
            lpips=_FakeLPIPS(),
            fid=_FakeVideoFeature(),
            fvd=_FakeVideoFeature(),
            clipscore=_FakeClip(),
        ),
    )

    assert isinstance(shared, runtime.ProductionRuntime)
    assert shared.device == torch.device("cpu")
    assert shared.torch_dtype is torch.float32
    assert len(shared.dataset_sources) == 1
    assert Path(shared.dataset_sources[0]).name == "Vchitect_T2V_DataVerse"
    assert calls["model_id"] == "stabilityai/sd-turbo"
    assert calls["torch_dtype"] is torch.float32
    assert calls["device"] == torch.device("cpu")
    assert shared.factory_options.height == args.height


def test_build_production_runtime_discovers_checkpoint_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tardis.cli.runtime as runtime

    datasets_file = _local_datasets_file(tmp_path)
    args = parse_args(
        [
            "--datasets-file",
            str(datasets_file),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--checkpoint-root",
            str(tmp_path),
        ]
    )
    assembly = build_tiny_tardis()
    events: list[str] = []
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()

    monkeypatch.setattr(
        runtime,
        "discover_checkpoint",
        lambda *args, **kwargs: events.append("discover") or checkpoint,
    )
    monkeypatch.setattr(
        runtime,
        "build_production_tardis",
        lambda **kwargs: events.append("build") or assembly.model,
    )
    monkeypatch.setattr(
        runtime,
        "load_temporal_checkpoint",
        lambda *args, **kwargs: events.append("load") or None,
    )
    monkeypatch.setattr(
        runtime,
        "build_metric_suite",
        lambda **kwargs: events.append("metrics") or object(),
    )

    runtime.build_production_runtime(
        args,
        motion_teacher=assembly.motion_teacher,
        feature_adapters=runtime.MetricFeatureAdapters(
            lpips=_FakeLPIPS(),
            fid=_FakeVideoFeature(),
            fvd=_FakeVideoFeature(),
            clipscore=_FakeClip(),
        ),
    )

    assert events == ["discover", "build", "load", "metrics"]
