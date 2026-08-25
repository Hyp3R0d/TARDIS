"""Training-only flow supervision and deployable prompt-conditioned motion scaffold."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from tardis.models.contracts import MotionTargets


@dataclass(frozen=True, slots=True)
class MotionScaffoldOutput:
    backward_flow: torch.Tensor
    visibility_logits: torch.Tensor
    motion_tokens: torch.Tensor
    flow_pyramid: tuple[torch.Tensor, ...]


class FlowMotionTeacher:
    """Generate adjacent optical-flow and cycle-consistency pseudo-labels."""

    def __init__(
        self,
        *,
        pyramid_scale: float = 0.5,
        levels: int = 3,
        window_size: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        cycle_temperature: float = 1.5,
    ) -> None:
        if not 0 < pyramid_scale < 1:
            raise ValueError("pyramid_scale must be in (0, 1)")
        if levels <= 0 or window_size <= 0 or iterations <= 0:
            raise ValueError("flow iteration settings must be positive")
        if cycle_temperature <= 0:
            raise ValueError("cycle_temperature must be positive")
        self.pyramid_scale = pyramid_scale
        self.levels = levels
        self.window_size = window_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.cycle_temperature = cycle_temperature

    @torch.no_grad()
    def estimate(
        self,
        video: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> MotionTargets:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("video must have shape [B,T,3,H,W]")
        if video.shape[1] < 2:
            raise ValueError("motion supervision requires at least two frames")
        output_height, output_width = output_size
        if output_height <= 0 or output_width <= 0:
            raise ValueError("output_size must be positive")
        cpu_video = video.detach().float().cpu().clamp(-1, 1)
        rgb = cpu_video.add(1).mul(127.5).permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)

        # The temporal network operates on the VAE latent grid. Estimating the
        # teacher flow at that grid avoids running two full-resolution CPU
        # Farneback passes for every adjacent frame while preserving the flow
        # units consumed by the transport module.
        output_height, output_width = output_size

        forward_pairs: list[torch.Tensor] = []
        backward_pairs: list[torch.Tensor] = []
        for batch_index in range(video.shape[0]):
            batch_forward: list[torch.Tensor] = []
            batch_backward: list[torch.Tensor] = []
            for frame_index in range(video.shape[1] - 1):
                previous = self._teacher_gray(
                    rgb[batch_index, frame_index],
                    output_size=(output_height, output_width),
                )
                current = self._teacher_gray(
                    rgb[batch_index, frame_index + 1],
                    output_size=(output_height, output_width),
                )
                forward = self._farneback(previous, current)
                backward = self._farneback(current, previous)
                batch_forward.append(torch.from_numpy(forward).permute(2, 0, 1))
                batch_backward.append(torch.from_numpy(backward).permute(2, 0, 1))
            forward_pairs.append(torch.stack(batch_forward))
            backward_pairs.append(torch.stack(batch_backward))
        forward_flow = torch.stack(forward_pairs).float()
        backward_flow = torch.stack(backward_pairs).float()

        flat_forward = forward_flow.flatten(0, 1)
        flat_backward = backward_flow.flatten(0, 1)
        visibility = self._backward_cycle_visibility(flat_forward, flat_backward)
        visibility = visibility.reshape(
            video.shape[0], video.shape[1] - 1, 1, output_height, output_width
        )
        return MotionTargets(
            forward_flow=forward_flow.to(video.device),
            backward_flow=backward_flow.to(video.device),
            visibility=visibility.clamp(0, 1).to(video.device),
        )

    @staticmethod
    def _teacher_gray(
        frame: np.ndarray,
        *,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        output_height, output_width = output_size
        if gray.shape == (output_height, output_width):
            return gray
        return np.asarray(
            cv2.resize(gray, (output_width, output_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )

    def _farneback(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        initial = np.zeros((*source.shape, 2), dtype=np.float32)
        result = cv2.calcOpticalFlowFarneback(
            source,
            target,
            initial,
            self.pyramid_scale,
            self.levels,
            self.window_size,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0,
        )
        return np.asarray(result, dtype=np.float32)

    def _backward_cycle_visibility(
        self,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = backward_flow.shape
        y, x = torch.meshgrid(
            torch.arange(height, dtype=backward_flow.dtype),
            torch.arange(width, dtype=backward_flow.dtype),
            indexing="ij",
        )
        previous_x = x[None] + backward_flow[:, 0]
        previous_y = y[None] + backward_flow[:, 1]
        normalized_x = 2 * (previous_x + 0.5) / width - 1
        normalized_y = 2 * (previous_y + 0.5) / height - 1
        grid = torch.stack((normalized_x, normalized_y), dim=-1)
        sampled_forward = functional.grid_sample(
            forward_flow,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        cycle_error = (backward_flow + sampled_forward).square().sum(1, keepdim=True).sqrt()
        in_bounds = (
            (previous_x >= 0)
            & (previous_x <= width - 1)
            & (previous_y >= 0)
            & (previous_y <= height - 1)
        ).unsqueeze(1)
        confidence = torch.exp(-cycle_error / self.cycle_temperature)
        return confidence * in_bounds.to(confidence.dtype)

    @staticmethod
    def _resize_flow(flow: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        batch, pairs, _, source_height, source_width = flow.shape
        output_height, output_width = output_size
        resized = functional.interpolate(
            flow.flatten(0, 1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        resized[:, 0].mul_(output_width / source_width)
        resized[:, 1].mul_(output_height / source_height)
        result: torch.Tensor = resized.reshape(batch, pairs, 2, output_height, output_width)
        return result


class _MotionResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm1(features)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(functional.silu(hidden))
        hidden = self.conv2(functional.silu(self.norm2(hidden)))
        result: torch.Tensor = features + hidden
        return result


class PromptMotionScaffold(nn.Module):
    """Predict causal motion structure from prompt, time, finite state, and noise."""

    def __init__(
        self,
        *,
        text_dim: int,
        state_channels: int,
        noise_channels: int,
        hidden_size: int,
        motion_token_dim: int,
        token_stride: int,
        max_flow_pixels: float,
        num_time_frequencies: int,
    ) -> None:
        super().__init__()
        if min(text_dim, state_channels, noise_channels, hidden_size, motion_token_dim) <= 0:
            raise ValueError("motion scaffold dimensions must be positive")
        if token_stride <= 0 or max_flow_pixels <= 0 or num_time_frequencies <= 0:
            raise ValueError("motion scaffold scale settings must be positive")
        self.max_flow_pixels = max_flow_pixels
        self.num_time_frequencies = num_time_frequencies
        condition_dim = hidden_size
        self.condition = nn.Sequential(
            nn.Linear(text_dim + 2 * num_time_frequencies, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, condition_dim),
        )
        self.stem = nn.Conv2d(state_channels + noise_channels, hidden_size, 3, padding=1)
        self.blocks = nn.ModuleList(
            [_MotionResidualBlock(hidden_size, condition_dim) for _ in range(3)]
        )
        self.flow_head = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.visibility_head = nn.Conv2d(hidden_size, 1, 3, padding=1)
        self.token_pool = nn.AvgPool2d(token_stride, stride=token_stride)
        self.token_projection = nn.Conv2d(hidden_size, motion_token_dim, 1)
        # A new temporal model must preserve the semantic keyframe before it has
        # learned any motion. This makes the frozen image prior a protected lower
        # bound instead of injecting random optical flow at initialization.
        nn.init.zeros_(self.flow_head.weight)
        if self.flow_head.bias is not None:
            nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.visibility_head.weight)
        if self.visibility_head.bias is not None:
            nn.init.constant_(self.visibility_head.bias, 8.0)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        time: torch.Tensor,
        state: torch.Tensor,
        motion_noise: torch.Tensor,
    ) -> MotionScaffoldOutput:
        if text_embeddings.ndim != 3 or text_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("text inputs must be [B,L,D] and mask [B,L]")
        if (
            state.ndim != 4
            or motion_noise.ndim != 4
            or state.shape[0::2] != motion_noise.shape[0::2]
        ):
            raise ValueError("state and motion_noise must share [B,H,W]")
        if time.shape != (state.shape[0],):
            raise ValueError("time must have shape [B]")
        weights = text_mask.unsqueeze(-1).to(text_embeddings.dtype)
        pooled_text = (text_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
        condition = self.condition(torch.cat((pooled_text, self._time_embedding(time)), dim=-1))
        features = self.stem(torch.cat((state, motion_noise), dim=1))
        for block in self.blocks:
            features = block(features, condition)
        flow = torch.tanh(self.flow_head(features)) * self.max_flow_pixels
        visibility_logits = self.visibility_head(features)
        tokens = self.token_projection(self.token_pool(features)).flatten(2).transpose(1, 2)
        flow_pyramid = [flow]
        current = flow
        for _ in range(2):
            current = (
                functional.interpolate(
                    current,
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                * 0.5
            )
            flow_pyramid.append(current)
        return MotionScaffoldOutput(
            backward_flow=flow,
            visibility_logits=visibility_logits,
            motion_tokens=tokens,
            flow_pyramid=tuple(flow_pyramid),
        )

    def _time_embedding(self, time: torch.Tensor) -> torch.Tensor:
        frequencies = torch.exp(
            torch.linspace(
                0,
                math.log(1000.0),
                self.num_time_frequencies,
                device=time.device,
                dtype=time.dtype,
            )
        )
        angles = time[:, None] * frequencies[None] * (2 * math.pi)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _group_count(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1
