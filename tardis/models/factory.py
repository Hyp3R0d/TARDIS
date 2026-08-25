"""Typed construction of TARDIS residual diffusion in innovation subspaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from tardis.models.clock import InnovationProperTime
from tardis.models.contracts import MotionTeacher
from tardis.models.motion import PromptMotionScaffold
from tardis.models.priors import FrozenPriorBundle, load_sd_turbo_prior_bundle
from tardis.models.quotient import TransportOrbitProjector
from tardis.models.residual import LiteResidualCorrector, SparseResidualDiT
from tardis.models.router import VisibilityCalibratedInnovationRouter
from tardis.models.state import CausalStateUpdater
from tardis.models.tardis import AblationVariant, TARDISConfig, TARDISModel
from tardis.models.transport import MotionStateTransport
from tardis.utils.defaults import DEFAULT_TARDIS_ARCHITECTURE

PriorLoader = Callable[..., FrozenPriorBundle]


@dataclass(frozen=True, slots=True)
class TARDISFactoryOptions:
    """Architecture options for the trainable TARDIS temporal network."""

    height: int = DEFAULT_TARDIS_ARCHITECTURE.height
    width: int = DEFAULT_TARDIS_ARCHITECTURE.width
    motion_noise_channels: int = DEFAULT_TARDIS_ARCHITECTURE.motion_noise_channels
    state_channels: int = DEFAULT_TARDIS_ARCHITECTURE.state_channels
    motion_hidden_size: int = DEFAULT_TARDIS_ARCHITECTURE.motion_hidden_size
    motion_token_dim: int = DEFAULT_TARDIS_ARCHITECTURE.motion_token_dim
    motion_token_stride: int = DEFAULT_TARDIS_ARCHITECTURE.motion_token_stride
    motion_max_flow_pixels: float = DEFAULT_TARDIS_ARCHITECTURE.motion_max_flow_pixels
    motion_time_frequencies: int = DEFAULT_TARDIS_ARCHITECTURE.motion_time_frequencies
    router_hidden_size: int = DEFAULT_TARDIS_ARCHITECTURE.router_hidden_size
    router_threshold: float = DEFAULT_TARDIS_ARCHITECTURE.router_threshold
    router_halo_radius: int = DEFAULT_TARDIS_ARCHITECTURE.router_halo_radius
    patch_size: int = DEFAULT_TARDIS_ARCHITECTURE.patch_size
    residual_hidden_size: int = DEFAULT_TARDIS_ARCHITECTURE.residual_hidden_size
    residual_layers: int = DEFAULT_TARDIS_ARCHITECTURE.residual_layers
    residual_heads: int = DEFAULT_TARDIS_ARCHITECTURE.residual_heads
    residual_max_grid_size: int | None = DEFAULT_TARDIS_ARCHITECTURE.residual_max_grid_size
    state_token_stride: int = DEFAULT_TARDIS_ARCHITECTURE.state_token_stride
    state_anchor_decay: float = DEFAULT_TARDIS_ARCHITECTURE.state_anchor_decay
    transport_max_correction_pixels: float = (
        DEFAULT_TARDIS_ARCHITECTURE.transport_max_correction_pixels
    )
    transport_history_fallback_weight: float = (
        DEFAULT_TARDIS_ARCHITECTURE.transport_history_fallback_weight
    )
    transport_quotient: bool = DEFAULT_TARDIS_ARCHITECTURE.transport_quotient
    quotient_regularization: float = DEFAULT_TARDIS_ARCHITECTURE.quotient_regularization
    quotient_rank_threshold: float = DEFAULT_TARDIS_ARCHITECTURE.quotient_rank_threshold
    innovation_proper_time: bool = DEFAULT_TARDIS_ARCHITECTURE.innovation_proper_time
    proper_time_maximum_hazard: float = DEFAULT_TARDIS_ARCHITECTURE.proper_time_maximum_hazard
    lite_hidden_channels: int = DEFAULT_TARDIS_ARCHITECTURE.lite_hidden_channels
    lite_max_magnitude: float = DEFAULT_TARDIS_ARCHITECTURE.lite_max_magnitude
    keyframe_lite_alignment: bool = DEFAULT_TARDIS_ARCHITECTURE.keyframe_lite_alignment
    keyframe_residual_generation: bool = (
        DEFAULT_TARDIS_ARCHITECTURE.keyframe_residual_generation
    )
    diffusion_steps: int = DEFAULT_TARDIS_ARCHITECTURE.diffusion_steps
    diffusion_time_sampling: str = DEFAULT_TARDIS_ARCHITECTURE.diffusion_time_sampling
    sampler_trajectory_alignment: bool = (
        DEFAULT_TARDIS_ARCHITECTURE.sampler_trajectory_alignment
    )
    active_ratio: float = DEFAULT_TARDIS_ARCHITECTURE.active_ratio
    gradient_checkpointing: bool = False
    scene_cut_threshold: float = DEFAULT_TARDIS_ARCHITECTURE.scene_cut_threshold
    oracle_temperature: float = DEFAULT_TARDIS_ARCHITECTURE.oracle_temperature
    training_noise_scale: float = DEFAULT_TARDIS_ARCHITECTURE.training_noise_scale
    prior_anchored_training: bool = True
    ablation: AblationVariant = AblationVariant(DEFAULT_TARDIS_ARCHITECTURE.ablation)

    def __post_init__(self) -> None:
        positive = (
            self.height,
            self.width,
            self.motion_noise_channels,
            self.state_channels,
            self.motion_hidden_size,
            self.motion_token_dim,
            self.motion_token_stride,
            self.motion_time_frequencies,
            self.router_hidden_size,
            self.patch_size,
            self.residual_hidden_size,
            self.residual_layers,
            self.residual_heads,
            self.state_token_stride,
            self.lite_hidden_channels,
            self.diffusion_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("TARDIS factory dimensions must be positive")
        if self.residual_hidden_size % self.residual_heads:
            raise ValueError("residual_hidden_size must be divisible by residual_heads")
        if not 0 < self.active_ratio <= 1:
            raise ValueError("active_ratio must be in (0, 1]")
        if (
            not isinstance(self.gradient_checkpointing, bool)
            or not isinstance(self.transport_quotient, bool)
            or not isinstance(self.innovation_proper_time, bool)
            or not isinstance(self.prior_anchored_training, bool)
            or not isinstance(self.keyframe_lite_alignment, bool)
            or not isinstance(self.keyframe_residual_generation, bool)
            or not isinstance(self.sampler_trajectory_alignment, bool)
        ):
            raise TypeError("TARDIS architecture flags must be boolean")
        if self.router_threshold < 0 or self.router_halo_radius < 0:
            raise ValueError("router threshold and halo radius cannot be negative")
        if self.motion_max_flow_pixels <= 0 or self.lite_max_magnitude <= 0:
            raise ValueError("motion and lite bounds must be positive")
        if self.transport_max_correction_pixels < 0:
            raise ValueError("transport correction bound cannot be negative")
        if not 0 <= self.transport_history_fallback_weight <= 1:
            raise ValueError("transport history fallback weight must be in [0, 1]")
        if self.quotient_regularization <= 0 or self.quotient_rank_threshold <= 0:
            raise ValueError("quotient regularization and rank threshold must be positive")
        if self.proper_time_maximum_hazard <= 0:
            raise ValueError("proper_time_maximum_hazard must be positive")
        if not 0 <= self.state_anchor_decay < 1:
            raise ValueError("state_anchor_decay must be in [0, 1)")
        if self.diffusion_time_sampling not in {"uniform", "high_noise", "endpoint"}:
            raise ValueError(
                "diffusion_time_sampling must be one of: uniform, high_noise, endpoint"
            )


def build_tardis_from_priors(
    priors: FrozenPriorBundle,
    *,
    motion_teacher: MotionTeacher | None,
    options: TARDISFactoryOptions,
) -> TARDISModel:
    """Compose the trainable temporal path around an already-loaded prior bundle."""

    scale = priors.spatial_scale
    if options.height % scale or options.width % scale:
        raise ValueError("height and width must be divisible by the codec spatial scale")
    latent_height = options.height // scale
    latent_width = options.width // scale
    if latent_height % options.patch_size or latent_width % options.patch_size:
        raise ValueError("latent height and width must be divisible by patch_size")
    max_grid_size = options.residual_max_grid_size or max(
        latent_height // options.patch_size,
        latent_width // options.patch_size,
    )
    if max_grid_size < max(latent_height, latent_width) // options.patch_size:
        raise ValueError("residual_max_grid_size is smaller than the latent patch grid")

    state_updater = CausalStateUpdater(
        latent_channels=priors.latent_channels,
        state_channels=options.state_channels,
        anchor_decay=options.state_anchor_decay,
    )
    motion_scaffold = PromptMotionScaffold(
        text_dim=priors.text_dim,
        state_channels=options.state_channels,
        noise_channels=options.motion_noise_channels,
        hidden_size=options.motion_hidden_size,
        motion_token_dim=options.motion_token_dim,
        token_stride=options.motion_token_stride,
        max_flow_pixels=options.motion_max_flow_pixels,
        num_time_frequencies=options.motion_time_frequencies,
    )
    transport = MotionStateTransport(
        channels=priors.latent_channels,
        max_correction_pixels=options.transport_max_correction_pixels,
        history_fallback_weight=options.transport_history_fallback_weight,
    )
    quotient = TransportOrbitProjector(
        regularization=options.quotient_regularization,
        rank_threshold=options.quotient_rank_threshold,
    )
    router = VisibilityCalibratedInnovationRouter(
        latent_channels=priors.latent_channels,
        motion_channels=2,
        state_channels=options.state_channels,
        text_dim=priors.text_dim,
        hidden_size=options.router_hidden_size,
        patch_size=options.patch_size,
        active_ratio=options.active_ratio,
        threshold=options.router_threshold,
        halo_radius=options.router_halo_radius,
    )
    innovation_clock = InnovationProperTime(
        patch_size=options.patch_size,
        active_ratio=options.active_ratio,
        threshold=options.router_threshold,
        halo_radius=options.router_halo_radius,
        maximum_hazard=options.proper_time_maximum_hazard,
    )
    lite_corrector = LiteResidualCorrector(
        latent_channels=priors.latent_channels,
        condition_channels=2 + options.state_channels,
        hidden_channels=options.lite_hidden_channels,
        max_magnitude=options.lite_max_magnitude,
        text_dim=priors.text_dim,
    )
    transition_lite_corrector = LiteResidualCorrector(
        latent_channels=priors.latent_channels,
        condition_channels=2 + options.state_channels,
        hidden_channels=options.lite_hidden_channels,
        max_magnitude=options.lite_max_magnitude,
        text_dim=priors.text_dim,
    )
    keyframe_residual_dit = SparseResidualDiT(
        latent_channels=priors.latent_channels,
        patch_size=options.patch_size,
        hidden_size=options.residual_hidden_size,
        num_layers=options.residual_layers,
        num_heads=options.residual_heads,
        text_dim=priors.text_dim,
        motion_dim=options.motion_token_dim,
        state_dim=options.state_channels,
        max_grid_size=max_grid_size,
        gradient_checkpointing=options.gradient_checkpointing,
    )
    residual_dit = SparseResidualDiT(
        latent_channels=priors.latent_channels,
        patch_size=options.patch_size,
        hidden_size=options.residual_hidden_size,
        num_layers=options.residual_layers,
        num_heads=options.residual_heads,
        text_dim=priors.text_dim,
        motion_dim=options.motion_token_dim,
        state_dim=options.state_channels,
        max_grid_size=max_grid_size,
        gradient_checkpointing=options.gradient_checkpointing,
    )
    return TARDISModel(
        priors=priors,
        motion_teacher=motion_teacher,
        motion_scaffold=motion_scaffold,
        transport=transport,
        quotient=quotient,
        router=router,
        innovation_clock=innovation_clock,
        lite_corrector=lite_corrector,
        transition_lite_corrector=transition_lite_corrector,
        keyframe_residual_dit=keyframe_residual_dit,
        residual_dit=residual_dit,
        state_updater=state_updater,
        config=TARDISConfig(
            height=options.height,
            width=options.width,
            motion_noise_channels=options.motion_noise_channels,
            state_token_stride=options.state_token_stride,
            scene_cut_threshold=options.scene_cut_threshold,
            oracle_temperature=options.oracle_temperature,
            training_noise_scale=options.training_noise_scale,
            transport_quotient=options.transport_quotient,
            innovation_proper_time=options.innovation_proper_time,
            prior_anchored_training=options.prior_anchored_training,
            keyframe_lite_alignment=options.keyframe_lite_alignment,
            keyframe_residual_generation=options.keyframe_residual_generation,
            diffusion_steps=options.diffusion_steps,
            diffusion_time_sampling=options.diffusion_time_sampling,
            sampler_trajectory_alignment=options.sampler_trajectory_alignment,
        ),
        ablation=options.ablation,
    )


def build_production_tardis(
    *,
    model_id: str,
    cache_dir: Path | str | None,
    torch_dtype: torch.dtype,
    device: torch.device,
    local_files_only: bool,
    motion_teacher: MotionTeacher | None,
    options: TARDISFactoryOptions,
    prior_loader: PriorLoader = load_sd_turbo_prior_bundle,
) -> TARDISModel:
    """Load the shared semantic prior and assemble a device-ready TARDIS model."""

    normalized_cache = None if cache_dir is None else str(cache_dir)
    priors = prior_loader(
        model_id,
        cache_dir=normalized_cache,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
    )
    if not isinstance(priors, FrozenPriorBundle):
        raise TypeError("prior_loader must return FrozenPriorBundle")
    model = build_tardis_from_priors(
        priors,
        motion_teacher=motion_teacher,
        options=options,
    )
    model.to(device=device, dtype=torch_dtype)
    model.priors.train(False)
    return model


def tardis_temporal_state_dict(model: TARDISModel) -> dict[str, torch.Tensor]:
    """Clone only trainable temporal-network state, excluding frozen priors."""

    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("priors.")
    }


def migrate_tardis_temporal_state_dict(
    model: TARDISModel,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply only the registered forward migrations for temporal checkpoints.

    Registered forward migrations preserve the legacy function because every new
    output projection is zero-initialized. Every unrelated missing or unexpected
    key remains a hard error.
    """

    expected = {
        name: value for name, value in model.state_dict().items() if not name.startswith("priors.")
    }
    received = set(state)
    prior_keys = sorted(name for name in received if name.startswith("priors."))
    if prior_keys:
        raise ValueError(f"checkpoint cannot contain frozen prior state: {prior_keys}")
    missing = set(expected) - received
    unexpected = received - set(expected)
    allowed_missing = tardis_forward_migration_names(expected)
    if not missing.issubset(allowed_missing):
        raise ValueError(
            "temporal checkpoint key mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if unexpected:
        raise ValueError(
            "temporal checkpoint key mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    migrated = dict(state)
    for name in missing:
        source_name = _forward_migration_source(name)
        source_value = None if source_name is None else state.get(source_name)
        if source_value is not None and source_value.shape == expected[name].shape:
            migrated[name] = source_value.detach().to(device="cpu", copy=True)
        else:
            migrated[name] = expected[name].detach().to(device="cpu", copy=True)
    return migrated


def tardis_forward_migration_names(names: Mapping[str, object] | set[str]) -> set[str]:
    """Return the exact temporal keys introduced by registered forward migrations."""

    available = set(names)
    explicit = {
        "lite_corrector.text_projection.weight",
        "lite_corrector.text_projection.bias",
    }
    prefixes = ("keyframe_residual_dit.", "transition_lite_corrector.")
    return {
        name
        for name in available
        if name in explicit or name.startswith(prefixes)
    }


def _forward_migration_source(name: str) -> str | None:
    if ".output_projection." in name:
        return None
    if name.startswith("keyframe_residual_dit."):
        return name.replace("keyframe_residual_dit.", "residual_dit.", 1)
    if name.startswith("transition_lite_corrector."):
        return name.replace("transition_lite_corrector.", "lite_corrector.", 1)
    return None


def load_tardis_temporal_state_dict(
    model: TARDISModel,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Strictly restore temporal state without allowing frozen-prior replacement."""

    state = migrate_tardis_temporal_state_dict(model, state)
    received = set(state)
    expected = {name for name in model.state_dict() if not name.startswith("priors.")}
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    if missing or unexpected:
        raise ValueError(
            f"temporal checkpoint key mismatch; missing={missing}, unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(dict(state), strict=False)
    nonprior_missing = [
        name for name in incompatible.missing_keys if not name.startswith("priors.")
    ]
    if nonprior_missing or incompatible.unexpected_keys:
        raise RuntimeError("temporal checkpoint failed strict model-state restoration")
