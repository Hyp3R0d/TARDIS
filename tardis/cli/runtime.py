"""Typed production assembly for the shared TARDIS runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from tardis.cli.common import DataOptions
from tardis.data.catalog import LOCAL_DATASET_DIRECTORIES, normalize_local_dataset_roots
from tardis.metrics.base import FramePairFeature, VideoTextFeature
from tardis.metrics.frechet import VideoFeature
from tardis.metrics.paired import SSIMMetric, TemporalConsistencyMetric
from tardis.metrics.suite import MetricSuite
from tardis.models.contracts import MotionTeacher
from tardis.models.factory import (
    PriorLoader,
    TARDISFactoryOptions,
    build_production_tardis,
    load_tardis_temporal_state_dict,
    migrate_tardis_temporal_state_dict,
    tardis_forward_migration_names,
)
from tardis.models.motion import FlowMotionTeacher
from tardis.models.tardis import TARDISModel
from tardis.utils.checkpoint import checkpoint_sha256, find_latest_checkpoint
from tardis.utils.checkpoint import load_checkpoint as load_checkpoint_payload


@dataclass(frozen=True, slots=True)
class MetricFeatureAdapters:
    """Optional injectable implementations for all six suite components."""

    lpips: FramePairFeature | None = None
    fid: VideoFeature | None = None
    fvd: VideoFeature | None = None
    clipscore: VideoTextFeature | None = None
    tc: TemporalConsistencyMetric | None = None
    ssim: SSIMMetric | None = None


@dataclass(frozen=True, slots=True)
class LoadedTemporalCheckpoint:
    """Identity and restore mode for a loaded temporal checkpoint."""

    path: Path
    sha256: str
    used_ema: bool


@dataclass(frozen=True, slots=True)
class ProductionRuntime:
    """Shared resources consumed by TARDIS train, infer, and apply commands."""

    model: TARDISModel
    motion_teacher: MotionTeacher
    metric_suite: MetricSuite
    dataset_sources: tuple[str, ...]
    factory_options: TARDISFactoryOptions
    device: torch.device
    torch_dtype: torch.dtype
    checkpoint: LoadedTemporalCheckpoint | None


@dataclass(frozen=True, slots=True)
class GenerationRuntime:
    """Model-only runtime used by prompt-only generation interfaces."""

    model: TARDISModel
    factory_options: TARDISFactoryOptions
    device: torch.device
    torch_dtype: torch.dtype
    checkpoint: LoadedTemporalCheckpoint


def resolve_torch_dtype(precision: str, device: torch.device | str) -> torch.dtype:
    """Resolve CLI precision with stable full-precision CPU fallbacks."""

    normalized = precision.strip().lower()
    target = torch.device(device)
    if normalized == "fp32":
        return torch.float32
    if normalized == "fp16":
        return torch.float16 if target.type == "cuda" else torch.float32
    if normalized == "bf16":
        if target.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    raise ValueError("precision must be one of: fp32, fp16, bf16")


def read_dataset_sources(
    path: Path | str,
    mirror_endpoint: str = DataOptions().mirror_endpoint,
) -> tuple[str, ...]:
    """Read and validate the immutable three-source local dataset catalog."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"dataset source file does not exist: {source_path}")
    lines = (
        line.strip()
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return validate_dataset_sources(tuple(lines), mirror_endpoint)


def validate_dataset_sources(
    sources: Sequence[str],
    mirror_endpoint: str = DataOptions().mirror_endpoint,
) -> tuple[str, ...]:
    """Validate source count, local paths, and the exact required datasets."""

    del mirror_endpoint
    roots = normalize_local_dataset_roots(sources)
    return tuple(str(roots[source]) for source in LOCAL_DATASET_DIRECTORIES)


def select_dataset_source(sources: Sequence[str], dataset: str) -> tuple[str]:
    """Return the canonical local path for one run-selected dataset."""

    roots = normalize_local_dataset_roots(sources)
    if dataset not in roots:
        raise ValueError(f"unknown dataset: {dataset!r}")
    return (str(roots[dataset]),)


def dataset_checkpoint_root(checkpoint_root: Path | str, dataset: str) -> Path:
    """Scope automatic checkpoint discovery to one dataset namespace."""

    if dataset not in LOCAL_DATASET_DIRECTORIES:
        raise ValueError(f"unknown dataset: {dataset!r}")
    return Path(checkpoint_root) / dataset


def factory_options_from_args(args: argparse.Namespace) -> TARDISFactoryOptions:
    """Map the shared CLI model fields to the temporal factory contract."""

    defaults = TARDISFactoryOptions()
    return TARDISFactoryOptions(
        height=getattr(args, "height", defaults.height),
        width=getattr(args, "width", defaults.width),
        motion_noise_channels=getattr(
            args,
            "latent_channels",
            defaults.motion_noise_channels,
        ),
        patch_size=getattr(args, "patch_size", defaults.patch_size),
        residual_hidden_size=getattr(
            args,
            "hidden_size",
            defaults.residual_hidden_size,
        ),
        residual_layers=getattr(args, "num_layers", defaults.residual_layers),
        residual_heads=getattr(args, "num_heads", defaults.residual_heads),
        active_ratio=getattr(args, "active_ratio", defaults.active_ratio),
        motion_max_flow_pixels=getattr(
            args,
            "motion_max_flow_pixels",
            defaults.motion_max_flow_pixels,
        ),
        transport_max_correction_pixels=getattr(
            args,
            "transport_max_correction_pixels",
            defaults.transport_max_correction_pixels,
        ),
        transport_history_fallback_weight=getattr(
            args,
            "transport_history_fallback_weight",
            defaults.transport_history_fallback_weight,
        ),
        router_threshold=getattr(
            args,
            "router_threshold",
            defaults.router_threshold,
        ),
        router_halo_radius=getattr(
            args,
            "router_halo_radius",
            defaults.router_halo_radius,
        ),
        state_anchor_decay=getattr(
            args,
            "state_anchor_decay",
            defaults.state_anchor_decay,
        ),
        scene_cut_threshold=getattr(
            args,
            "scene_cut_threshold",
            defaults.scene_cut_threshold,
        ),
        oracle_temperature=getattr(
            args,
            "oracle_temperature",
            defaults.oracle_temperature,
        ),
        training_noise_scale=getattr(
            args,
            "training_noise_scale",
            defaults.training_noise_scale,
        ),
        lite_max_magnitude=getattr(
            args,
            "lite_max_magnitude",
            defaults.lite_max_magnitude,
        ),
        keyframe_lite_alignment=getattr(
            args,
            "keyframe_lite_alignment",
            defaults.keyframe_lite_alignment,
        ),
        keyframe_residual_generation=getattr(
            args,
            "keyframe_residual_generation",
            defaults.keyframe_residual_generation,
        ),
        diffusion_steps=getattr(
            args,
            "diffusion_steps",
            defaults.diffusion_steps,
        ),
        diffusion_time_sampling=getattr(
            args,
            "diffusion_time_sampling",
            defaults.diffusion_time_sampling,
        ),
        sampler_trajectory_alignment=getattr(
            args,
            "sampler_trajectory_alignment",
            defaults.sampler_trajectory_alignment,
        ),
        transport_quotient=getattr(
            args,
            "transport_quotient",
            defaults.transport_quotient,
        ),
        quotient_regularization=getattr(
            args,
            "quotient_regularization",
            defaults.quotient_regularization,
        ),
        quotient_rank_threshold=getattr(
            args,
            "quotient_rank_threshold",
            defaults.quotient_rank_threshold,
        ),
        innovation_proper_time=getattr(
            args,
            "innovation_proper_time",
            defaults.innovation_proper_time,
        ),
        proper_time_maximum_hazard=getattr(
            args,
            "proper_time_maximum_hazard",
            defaults.proper_time_maximum_hazard,
        ),
        gradient_checkpointing=getattr(
            args,
            "gradient_checkpointing",
            defaults.gradient_checkpointing,
        ),
    )


def discover_checkpoint(
    checkpoint: Path | str | None,
    checkpoint_root: Path | str,
) -> Path:
    """Resolve an explicit checkpoint or the newest timestamped ``best.pt``."""

    if checkpoint is not None:
        explicit = Path(checkpoint)
        if not explicit.is_file():
            raise FileNotFoundError(f"checkpoint does not exist or is not a file: {explicit}")
        return explicit
    discovered = find_latest_checkpoint(Path(checkpoint_root), "best.pt")
    if discovered is None:
        raise FileNotFoundError(
            f"no timestamped best.pt checkpoint was found below {Path(checkpoint_root)}"
        )
    return discovered


def load_temporal_checkpoint(
    model: TARDISModel,
    checkpoint: Path | str,
    *,
    use_ema: bool = False,
) -> LoadedTemporalCheckpoint:
    """Restore temporal state and optionally overlay the checkpoint EMA shadow."""

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist or is not a file: {path}")
    digest = checkpoint_sha256(path)
    payload = load_checkpoint_payload(path, map_location="cpu")
    raw_model = payload.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("checkpoint model field must be a temporal state mapping")
    model_state = _prepare_temporal_state(model, _tensor_mapping(raw_model, "checkpoint model"))

    raw_ema = payload.get("ema")
    raw_shadow = raw_ema.get("shadow") if isinstance(raw_ema, Mapping) else None
    if isinstance(raw_shadow, Mapping):
        prior_names = sorted(
            name for name in raw_shadow if isinstance(name, str) and name.startswith("priors.")
        )
        if prior_names:
            raise ValueError(f"checkpoint EMA cannot contain frozen prior state: {prior_names}")

    shadow: dict[str, torch.Tensor] | None = None
    if use_ema:
        if not isinstance(raw_ema, Mapping):
            raise ValueError("checkpoint ema field must be a mapping when EMA is requested")
        if not isinstance(raw_shadow, Mapping):
            raise ValueError("checkpoint ema.shadow field must be a mapping")
        shadow = _prepare_ema_shadow(model, _tensor_mapping(raw_shadow, "checkpoint ema.shadow"))

    original_state = _model_state_snapshot(model)
    try:
        load_tardis_temporal_state_dict(model, model_state)
        if shadow is not None:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.is_floating_point() and not name.startswith("priors."):
                        parameter.copy_(shadow[name])
    except BaseException:
        _restore_model_state(model, original_state)
        raise
    return LoadedTemporalCheckpoint(path, digest, shadow is not None)


def build_metric_suite(
    *,
    device: torch.device | str,
    feature_adapters: MetricFeatureAdapters | None = None,
) -> MetricSuite:
    """Construct the six real production metrics with injectable test adapters."""

    if feature_adapters is not None and not isinstance(feature_adapters, MetricFeatureAdapters):
        raise TypeError("feature_adapters must be a MetricFeatureAdapters instance or None")
    adapters = MetricFeatureAdapters() if feature_adapters is None else feature_adapters
    metric_device = torch.device(device)
    if (
        adapters.lpips is None
        or adapters.fid is None
        or adapters.fvd is None
        or adapters.clipscore is None
    ):
        from tardis.metrics.features import (
            AlexNetLPIPS,
            I3DKineticsFeatures,
            InceptionV3PoolFeatures,
            OpenCLIPFeatures,
        )

        lpips_feature = adapters.lpips or AlexNetLPIPS(device=metric_device)
        fid_feature = adapters.fid or InceptionV3PoolFeatures(device=metric_device)
        fvd_feature = adapters.fvd or I3DKineticsFeatures(device=metric_device)
        clip_feature = adapters.clipscore or OpenCLIPFeatures(device=metric_device)
    else:
        lpips_feature = adapters.lpips
        fid_feature = adapters.fid
        fvd_feature = adapters.fvd
        clip_feature = adapters.clipscore

    from tardis.metrics.frechet import FIDMetric, FVDMetric
    from tardis.metrics.paired import CLIPScoreMetric, LPIPSMetric

    return MetricSuite(
        tc=adapters.tc or TemporalConsistencyMetric(),
        lpips=LPIPSMetric(lpips_feature),
        fvd=FVDMetric(fvd_feature),
        fid=FIDMetric(fid_feature),
        clipscore=CLIPScoreMetric(clip_feature),
        ssim=adapters.ssim or SSIMMetric(),
    )


def build_production_runtime(
    args: argparse.Namespace,
    *,
    checkpoint: Path | str | None = None,
    restore_checkpoint: bool = True,
    use_ema: bool = False,
    local_files_only: bool = False,
    prior_loader: PriorLoader | None = None,
    motion_teacher: MotionTeacher | None = None,
    feature_adapters: MetricFeatureAdapters | None = None,
) -> ProductionRuntime:
    """Assemble the device-ready model, motion teacher, metrics, and source catalog."""

    all_dataset_sources = read_dataset_sources(args.datasets_file, args.mirror_endpoint)
    dataset_sources = select_dataset_source(all_dataset_sources, str(args.dataset))
    device = torch.device(args.device)
    torch_dtype = resolve_torch_dtype(args.precision, device)
    factory_options = factory_options_from_args(args)
    checkpoint_path: Path | None = None
    if restore_checkpoint:
        requested_checkpoint = checkpoint
        if requested_checkpoint is None:
            requested_checkpoint = getattr(args, "checkpoint", None)
        if requested_checkpoint is None:
            requested_checkpoint = getattr(args, "resume", None)
        checkpoint_path = discover_checkpoint(
            requested_checkpoint,
            dataset_checkpoint_root(args.checkpoint_root, str(args.dataset)),
        )
    teacher = FlowMotionTeacher() if motion_teacher is None else motion_teacher
    model = build_production_tardis(
        model_id=args.pretrained_model,
        cache_dir=None,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
        motion_teacher=teacher,
        options=factory_options,
        **({} if prior_loader is None else {"prior_loader": prior_loader}),
    )
    loaded_checkpoint: LoadedTemporalCheckpoint | None = None
    if checkpoint_path is not None:
        loaded_checkpoint = load_temporal_checkpoint(
            model,
            checkpoint_path,
            use_ema=use_ema,
        )
    return ProductionRuntime(
        model=model,
        motion_teacher=teacher,
        metric_suite=build_metric_suite(
            device=device,
            feature_adapters=feature_adapters,
        ),
        dataset_sources=dataset_sources,
        factory_options=factory_options,
        device=device,
        torch_dtype=torch_dtype,
        checkpoint=loaded_checkpoint,
    )


def build_generation_runtime(
    args: argparse.Namespace,
    *,
    checkpoint: Path | str | None = None,
    use_ema: bool = True,
    local_files_only: bool = False,
    prior_loader: PriorLoader | None = None,
    motion_teacher: MotionTeacher | None = None,
) -> GenerationRuntime:
    """Assemble only the checkpointed prompt-to-video model, without data or metrics."""

    device = torch.device(args.device)
    torch_dtype = resolve_torch_dtype(args.precision, device)
    factory_options = factory_options_from_args(args)
    checkpoint_path = discover_checkpoint(
        checkpoint or args.checkpoint,
        dataset_checkpoint_root(args.checkpoint_root, str(args.dataset)),
    )
    teacher = FlowMotionTeacher() if motion_teacher is None else motion_teacher
    model = build_production_tardis(
        model_id=args.pretrained_model,
        cache_dir=None,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
        motion_teacher=teacher,
        options=factory_options,
        **({} if prior_loader is None else {"prior_loader": prior_loader}),
    )
    loaded_checkpoint = load_temporal_checkpoint(model, checkpoint_path, use_ema=use_ema)
    return GenerationRuntime(
        model=model,
        factory_options=factory_options,
        device=device,
        torch_dtype=torch_dtype,
        checkpoint=loaded_checkpoint,
    )


def _tensor_mapping(value: Mapping[object, object], name: str) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, torch.Tensor):
            raise ValueError(f"{name} entries must be named tensors")
        result[key] = item
    return result


def _prepare_temporal_state(
    model: TARDISModel,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state = migrate_tardis_temporal_state_dict(model, state)
    expected = {
        name: value for name, value in model.state_dict().items() if not name.startswith("priors.")
    }
    prior_names = sorted(name for name in state if name.startswith("priors."))
    if prior_names:
        raise ValueError(f"checkpoint cannot contain frozen prior state: {prior_names}")
    _validate_state_keys(state, expected, "temporal checkpoint")
    return _prepare_tensors(state, expected, "temporal checkpoint")


def _prepare_ema_shadow(
    model: TARDISModel,
    shadow: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and not name.startswith("priors.")
    }
    prior_names = sorted(name for name in shadow if name.startswith("priors."))
    if prior_names:
        raise ValueError(f"checkpoint EMA cannot contain frozen prior state: {prior_names}")
    received = set(shadow)
    expected_names = set(expected)
    missing = expected_names - received
    unexpected = received - expected_names
    allowed_missing = tardis_forward_migration_names(expected)
    if not missing.issubset(allowed_missing) or unexpected:
        raise ValueError(
            "checkpoint EMA key mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    migrated = dict(shadow)
    for name in missing:
        source_name = _ema_forward_migration_source(name)
        source_value = None if source_name is None else shadow.get(source_name)
        if isinstance(source_value, torch.Tensor) and source_value.shape == expected[name].shape:
            migrated[name] = source_value.detach().to(device="cpu", copy=True)
        else:
            migrated[name] = expected[name].detach().to(device="cpu", copy=True)
    shadow = migrated
    return _prepare_tensors(shadow, expected, "checkpoint EMA")


def _ema_forward_migration_source(name: str) -> str | None:
    if ".output_projection." in name:
        return None
    if name.startswith("keyframe_residual_dit."):
        return name.replace("keyframe_residual_dit.", "residual_dit.", 1)
    if name.startswith("transition_lite_corrector."):
        return name.replace("transition_lite_corrector.", "lite_corrector.", 1)
    return None


def _validate_state_keys(
    state: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    name: str,
) -> None:
    received = set(state)
    expected_names = set(expected)
    missing = sorted(expected_names - received)
    unexpected = sorted(received - expected_names)
    if missing or unexpected:
        raise ValueError(f"{name} key mismatch; missing={missing}, unexpected={unexpected}")


def _prepare_tensors(
    state: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    name: str,
) -> dict[str, torch.Tensor]:
    prepared: dict[str, torch.Tensor] = {}
    for key, target in expected.items():
        value = state[key]
        if value.shape != target.shape:
            raise ValueError(
                f"{name} tensor {key!r} has shape {tuple(value.shape)}; "
                f"expected {tuple(target.shape)}"
            )
        prepared[key] = value.to(device="cpu", dtype=target.dtype)
    return prepared


def _model_state_snapshot(model: TARDISModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in model.state_dict().items()
        if not name.startswith("priors.")
    }


def _restore_model_state(model: TARDISModel, snapshot: Mapping[str, torch.Tensor]) -> None:
    current = model.state_dict()
    with torch.no_grad():
        for name, value in snapshot.items():
            current[name].copy_(value)


__all__ = [
    "LoadedTemporalCheckpoint",
    "MetricFeatureAdapters",
    "ProductionRuntime",
    "build_metric_suite",
    "build_production_runtime",
    "discover_checkpoint",
    "dataset_checkpoint_root",
    "factory_options_from_args",
    "load_temporal_checkpoint",
    "read_dataset_sources",
    "resolve_torch_dtype",
    "validate_dataset_sources",
    "select_dataset_source",
]
