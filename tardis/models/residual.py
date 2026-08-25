"""Sparse transport-residual DiT and low-frequency correction branch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint

from tardis.models.router import InnovationSelection


@dataclass(frozen=True, slots=True)
class PatchGrid:
    channels: int
    height: int
    width: int
    patch_size: int

    @property
    def token_count(self) -> int:
        return self.height * self.width

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch_size * self.patch_size


@dataclass(frozen=True, slots=True)
class SparseResidualOutput:
    residual: torch.Tensor
    active_tokens: torch.Tensor
    active_counts: torch.Tensor
    memory_key_padding_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ResidualDenoisingContext:
    """All causal conditions shared by the residual teacher and student."""

    noisy_residual: torch.Tensor
    diffusion_noise: torch.Tensor
    transported_prior: torch.Tensor
    diffusion_time: torch.Tensor
    event_probability: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor
    motion_tokens: torch.Tensor
    state_tokens: torch.Tensor
    selection: InnovationSelection

    def predict(
        self,
        denoiser: SparseResidualDiT,
        *,
        noisy_residual: torch.Tensor | None = None,
        diffusion_time: torch.Tensor | None = None,
    ) -> SparseResidualOutput:
        return cast(
            SparseResidualOutput,
            denoiser(
            noisy_residual=self.noisy_residual if noisy_residual is None else noisy_residual,
            transported_prior=self.transported_prior,
            diffusion_time=self.diffusion_time if diffusion_time is None else diffusion_time,
            event_probability=self.event_probability,
            text_tokens=self.text_tokens,
            text_mask=self.text_mask,
            motion_tokens=self.motion_tokens,
            state_tokens=self.state_tokens,
                selection=self.selection,
            ),
        )


def patchify(tensor: torch.Tensor, *, patch_size: int) -> tuple[torch.Tensor, PatchGrid]:
    """Convert ``[B,C,H,W]`` into row-major non-overlapping patch tokens."""

    if tensor.ndim != 4:
        raise ValueError("patchify expects [B,C,H,W]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    batch, channels, height, width = tensor.shape
    if height % patch_size or width % patch_size:
        raise ValueError("spatial dimensions must be divisible by patch_size")
    grid_height = height // patch_size
    grid_width = width // patch_size
    tokens = (
        tensor.reshape(batch, channels, grid_height, patch_size, grid_width, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_height * grid_width, channels * patch_size * patch_size)
    )
    return tokens, PatchGrid(channels, grid_height, grid_width, patch_size)


def unpatchify(tokens: torch.Tensor, grid: PatchGrid) -> torch.Tensor:
    """Invert :func:`patchify` exactly."""

    if tokens.ndim != 3:
        raise ValueError("unpatchify expects [B,N,D]")
    if tokens.shape[1:] != (grid.token_count, grid.patch_dim):
        raise ValueError("token geometry does not match PatchGrid")
    batch = tokens.shape[0]
    return (
        tokens.reshape(
            batch,
            grid.height,
            grid.width,
            grid.channels,
            grid.patch_size,
            grid.patch_size,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(
            batch,
            grid.channels,
            grid.height * grid.patch_size,
            grid.width * grid.patch_size,
        )
    )


def gather_tokens(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Gather a fixed-width sparse batch and zero its padded slots."""

    if tokens.ndim != 3 or indices.ndim != 2 or valid_tokens.shape != indices.shape:
        raise ValueError("tokens/indices/valid_tokens must be [B,N,D], [B,K], [B,K]")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError("tokens and indices must share batch size")
    if indices.numel() and (indices.min() < 0 or indices.max() >= tokens.shape[1]):
        raise ValueError("gather index lies outside token sequence")
    expanded = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    gathered = tokens.gather(1, expanded)
    return gathered * valid_tokens.unsqueeze(-1).to(gathered.dtype)


def scatter_tokens(
    base_tokens: torch.Tensor,
    updates: torch.Tensor,
    indices: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Replace valid gathered slots while preserving every inactive token exactly."""

    if updates.shape[:2] != indices.shape or updates.shape[-1] != base_tokens.shape[-1]:
        raise ValueError("updates must have shape [B,K,D] compatible with base tokens")
    gathered_base = gather_tokens(base_tokens, indices, torch.ones_like(valid_tokens))
    delta = (updates - gathered_base) * valid_tokens.unsqueeze(-1).to(updates.dtype)
    expanded = indices.unsqueeze(-1).expand_as(delta)
    result = base_tokens.clone()
    result.scatter_add_(1, expanded, delta)
    return result


class _AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.modulation_projection = nn.Linear(hidden_size, hidden_size * 9)
        self.modulation = nn.Sequential(nn.SiLU(), self.modulation_projection)
        nn.init.zeros_(self.modulation_projection.weight)
        nn.init.zeros_(self.modulation_projection.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        *,
        valid_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        shifts = self.modulation(condition).chunk(9, dim=-1)
        shift_self, scale_self, gate_self = shifts[0:3]
        shift_cross, scale_cross, gate_cross = shifts[3:6]
        shift_mlp, scale_mlp, gate_mlp = shifts[6:9]
        safe_valid = valid_tokens.clone()
        empty_rows = ~safe_valid.any(dim=1)
        safe_valid[empty_rows, 0] = True

        hidden = self.norm1(tokens)
        hidden = hidden * (1 + scale_self[:, None]) + shift_self[:, None]
        attended, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        tokens = tokens + gate_self[:, None] * attended

        hidden = self.norm2(tokens)
        hidden = hidden * (1 + scale_cross[:, None]) + shift_cross[:, None]
        attended, _ = self.cross_attention(
            hidden,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + gate_cross[:, None] * attended

        hidden = self.norm3(tokens)
        hidden = hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        tokens = tokens + gate_mlp[:, None] * self.mlp(hidden)
        return tokens * valid_tokens.unsqueeze(-1).to(tokens.dtype)


class SparseResidualDiT(nn.Module):
    """Run residual denoising only in the DIS-selected innovation subspace."""

    def __init__(
        self,
        *,
        latent_channels: int,
        patch_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        text_dim: int,
        motion_dim: int,
        state_dim: int,
        max_grid_size: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if (
            min(
                latent_channels,
                patch_size,
                hidden_size,
                num_layers,
                num_heads,
                text_dim,
                motion_dim,
                state_dim,
                max_grid_size,
            )
            <= 0
        ):
            raise ValueError("SparseResidualDiT dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.max_grid_size = max_grid_size
        self.gradient_checkpointing = gradient_checkpointing
        patch_dim = latent_channels * patch_size * patch_size
        self.input_projection = nn.Linear(patch_dim * 2, hidden_size)
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.motion_projection = nn.Linear(motion_dim, hidden_size)
        self.state_projection = nn.Linear(state_dim, hidden_size)
        self.proper_time_projection = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [_AdaLNZeroBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, patch_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        *,
        noisy_residual: torch.Tensor,
        transported_prior: torch.Tensor,
        diffusion_time: torch.Tensor,
        event_probability: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        selection: InnovationSelection,
    ) -> SparseResidualOutput:
        if noisy_residual.shape != transported_prior.shape or noisy_residual.ndim != 4:
            raise ValueError("residual and prior must share [B,C,H,W]")
        if noisy_residual.shape[1] != self.latent_channels:
            raise ValueError("latent channel count does not match model")
        if event_probability.shape != (
            noisy_residual.shape[0],
            1,
            *noisy_residual.shape[-2:],
        ):
            raise ValueError("event_probability must have shape [B,1,H,W]")
        residual_patches, grid = patchify(noisy_residual, patch_size=self.patch_size)
        prior_patches, _ = patchify(transported_prior, patch_size=self.patch_size)
        if max(grid.height, grid.width) > self.max_grid_size:
            raise ValueError("patch grid exceeds max_grid_size")
        if selection.indices.shape[0] != noisy_residual.shape[0]:
            raise ValueError("selection batch size does not match residual")

        full_tokens = self.input_projection(torch.cat((residual_patches, prior_patches), dim=-1))
        positions = _sinusoidal_2d_positions(
            grid.height,
            grid.width,
            self.hidden_size,
            device=full_tokens.device,
            dtype=full_tokens.dtype,
        )
        full_tokens = full_tokens + positions.unsqueeze(0)
        active_tokens = gather_tokens(
            full_tokens,
            selection.indices,
            selection.valid_tokens,
        )
        patch_probability = (
            functional.avg_pool2d(
                event_probability,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            .flatten(2)
            .transpose(1, 2)
        )
        active_probability = gather_tokens(
            patch_probability,
            selection.indices,
            selection.valid_tokens,
        )
        active_tokens = active_tokens + self.proper_time_projection(active_probability)

        memory = torch.cat(
            (
                self.text_projection(text_tokens),
                self.motion_projection(motion_tokens),
                self.state_projection(state_tokens),
            ),
            dim=1,
        )
        memory_key_padding_mask = torch.cat(
            (
                ~text_mask,
                torch.zeros(
                    motion_tokens.shape[:2],
                    device=motion_tokens.device,
                    dtype=torch.bool,
                ),
                torch.zeros(
                    state_tokens.shape[:2],
                    device=state_tokens.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        condition = self.time_mlp(_time_embedding(diffusion_time, self.hidden_size))
        for raw_block in self.blocks:
            block = cast(_AdaLNZeroBlock, raw_block)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                active_tokens = cast(
                    torch.Tensor,
                    checkpoint(
                        partial(
                            _run_residual_block,
                            block,
                            valid_tokens=selection.valid_tokens,
                            memory=memory,
                            memory_key_padding_mask=memory_key_padding_mask,
                        ),
                        active_tokens,
                        condition,
                        use_reentrant=False,
                    ),
                )
            else:
                active_tokens = block(
                    active_tokens,
                    condition,
                    valid_tokens=selection.valid_tokens,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        active_tokens = self.final_norm(active_tokens)
        active_tokens = active_tokens * selection.valid_tokens.unsqueeze(-1).to(active_tokens.dtype)
        active_patches = self.output_projection(active_tokens)
        active_patches = active_patches * selection.valid_tokens.unsqueeze(-1).to(
            active_patches.dtype
        )
        residual_tokens = scatter_tokens(
            torch.zeros_like(residual_patches),
            active_patches,
            selection.indices,
            selection.valid_tokens,
        )
        return SparseResidualOutput(
            residual=unpatchify(residual_tokens, grid),
            active_tokens=active_tokens,
            active_counts=selection.active_counts,
            memory_key_padding_mask=memory_key_padding_mask,
        )


def _run_residual_block(
    block: _AdaLNZeroBlock,
    tokens: torch.Tensor,
    condition: torch.Tensor,
    *,
    valid_tokens: torch.Tensor,
    memory: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    return block.forward(
        tokens,
        condition,
        valid_tokens=valid_tokens,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
    )


class LiteResidualCorrector(nn.Module):
    """Bounded depthwise low-frequency update over all latent locations."""

    def __init__(
        self,
        *,
        latent_channels: int,
        condition_channels: int,
        hidden_channels: int,
        max_magnitude: float,
        text_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(latent_channels, condition_channels, hidden_channels) <= 0 or max_magnitude <= 0:
            raise ValueError("LiteResidualCorrector settings must be positive")
        if text_dim is not None and text_dim <= 0:
            raise ValueError("text_dim must be positive when provided")
        self.max_magnitude = max_magnitude
        self.text_dim = text_dim
        input_channels = latent_channels + condition_channels + 1
        self.input_projection = nn.Conv2d(input_channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=5,
            padding=2,
            groups=hidden_channels,
        )
        self.text_projection = (
            nn.Linear(text_dim, hidden_channels) if text_dim is not None else None
        )
        if self.text_projection is not None:
            # Keep legacy checkpoints functionally identical until the new branch learns.
            nn.init.zeros_(self.text_projection.weight)
            nn.init.zeros_(self.text_projection.bias)
        self.output_projection = nn.Conv2d(hidden_channels, latent_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        transported_prior: torch.Tensor,
        condition: torch.Tensor,
        innovation_probability: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            condition.shape[0] != transported_prior.shape[0]
            or condition.shape[-2:] != transported_prior.shape[-2:]
        ):
            raise ValueError("lite condition must share batch and spatial geometry")
        if innovation_probability.shape != (
            transported_prior.shape[0],
            1,
            *transported_prior.shape[-2:],
        ):
            raise ValueError("innovation_probability must have shape [B,1,H,W]")
        features = torch.cat((transported_prior, condition, innovation_probability), dim=1)
        features = functional.silu(self.input_projection(features))
        features = functional.silu(self.depthwise(features))
        if text_embeddings is not None or text_mask is not None:
            if self.text_projection is None:
                raise ValueError("text conditioning is unavailable for this corrector")
            if text_embeddings is None or text_mask is None:
                raise ValueError("text_embeddings and text_mask must be provided together")
            if (
                text_embeddings.ndim != 3
                or text_embeddings.shape[0] != transported_prior.shape[0]
                or text_embeddings.shape[-1] != self.text_dim
                or text_mask.shape != text_embeddings.shape[:2]
                or text_mask.dtype != torch.bool
            ):
                raise ValueError("text conditioning must be [B,L,D] with a bool [B,L] mask")
            weights = text_mask.unsqueeze(-1).to(text_embeddings.dtype)
            pooled_text = (text_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
            features = features + self.text_projection(pooled_text)[:, :, None, None]
        features = functional.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        return torch.tanh(self.output_projection(features)) * self.max_magnitude


def _time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError("diffusion_time must have shape [B]")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = functional.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


def _sinusoidal_2d_positions(
    height: int,
    width: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("hidden_size must be divisible by four for 2D positions")
    quarter = dimension // 4
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(quarter, device=device, dtype=dtype)
        / max(quarter - 1, 1)
    )
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_angles = x.reshape(-1, 1) * frequencies[None]
    y_angles = y.reshape(-1, 1) * frequencies[None]
    return torch.cat((x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()), dim=-1)
