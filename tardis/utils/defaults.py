"""Lightweight architecture defaults shared by CLI and model assembly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TARDISArchitectureDefaults:
    height: int = 512
    width: int = 512
    motion_noise_channels: int = 4
    state_channels: int = 128
    motion_hidden_size: int = 256
    motion_token_dim: int = 128
    motion_token_stride: int = 4
    motion_max_flow_pixels: float = 8.0
    motion_time_frequencies: int = 8
    router_hidden_size: int = 128
    router_threshold: float = 0.1
    router_halo_radius: int = 1
    patch_size: int = 2
    residual_hidden_size: int = 512
    residual_layers: int = 8
    residual_heads: int = 8
    residual_max_grid_size: int | None = None
    state_token_stride: int = 4
    state_anchor_decay: float = 0.95
    transport_max_correction_pixels: float = 0.25
    transport_history_fallback_weight: float = 1.0
    transport_quotient: bool = True
    quotient_regularization: float = 1.0e-4
    quotient_rank_threshold: float = 1.0e-5
    innovation_proper_time: bool = True
    proper_time_maximum_hazard: float = 20.0
    lite_hidden_channels: int = 128
    lite_max_magnitude: float = 0.75
    keyframe_lite_alignment: bool = True
    keyframe_residual_generation: bool = True
    diffusion_steps: int = 2
    diffusion_time_sampling: str = "endpoint"
    sampler_trajectory_alignment: bool = True
    active_ratio: float = 0.35
    scene_cut_threshold: float = 0.98
    oracle_temperature: float = 0.25
    training_noise_scale: float = 0.1
    ablation: str = "A10"


DEFAULT_TARDIS_ARCHITECTURE = TARDISArchitectureDefaults()


__all__ = ["DEFAULT_TARDIS_ARCHITECTURE", "TARDISArchitectureDefaults"]
