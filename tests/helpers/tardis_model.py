from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from tardis.models.clock import InnovationProperTime
from tardis.models.contracts import MotionTargets
from tardis.models.motion import PromptMotionScaffold
from tardis.models.priors import FrozenPriorBundle
from tardis.models.quotient import TransportOrbitProjector
from tardis.models.residual import LiteResidualCorrector, SparseResidualDiT
from tardis.models.router import VisibilityCalibratedInnovationRouter
from tardis.models.state import CausalStateUpdater
from tardis.models.tardis import TARDISConfig, TARDISModel
from tardis.models.transport import MotionStateTransport


class TinyCodec(nn.Module):
    latent_channels = 4
    spatial_scale = 2

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, 3, 1)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        flattened = video.reshape(batch * frames, channels, height, width)
        latent = functional.avg_pool2d(self.encoder(flattened), 2)
        return latent.reshape(batch, frames, 4, height // 2, width // 2)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = latents.shape
        flattened = latents.reshape(batch * frames, channels, height, width)
        pixels = functional.interpolate(
            self.decoder(flattened),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return pixels.reshape(batch, frames, 3, height * 2, width * 2)


class TinyTextConditioner(nn.Module):
    embedding_dim = 12
    max_length = 6

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, self.embedding_dim)

    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.zeros(len(prompts), self.max_length, dtype=torch.long)
        mask = torch.zeros(len(prompts), self.max_length, dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            values = list(prompt.encode())[: self.max_length]
            tokens[row, : len(values)] = torch.tensor(values)
            mask[row, : len(values)] = True
        return self.embedding(tokens), mask


class TinyFirstFrameGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(12, 4)

    def generate_first_latent(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        generator: torch.Generator,
        height: int,
        width: int,
    ) -> torch.Tensor:
        del generator
        weights = text_mask.unsqueeze(-1).to(text_embeddings.dtype)
        pooled = (text_embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)
        latent = self.projection(pooled)[:, :, None, None]
        return latent.expand(-1, -1, height // 2, width // 2).contiguous()


class ZeroMotionTeacher:
    def __init__(self) -> None:
        self.calls = 0

    def estimate(
        self,
        video: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> MotionTargets:
        self.calls += 1
        batch, frames = video.shape[:2]
        height, width = output_size
        flow = torch.zeros(batch, frames - 1, 2, height, width, device=video.device)
        visibility = torch.ones(batch, frames - 1, 1, height, width, device=video.device)
        return MotionTargets(flow, flow.clone(), visibility)


@dataclass(frozen=True)
class TinyTARDISAssembly:
    model: TARDISModel
    priors: FrozenPriorBundle
    motion_teacher: ZeroMotionTeacher


def build_tiny_tardis(
    *,
    active_ratio: float = 0.25,
    keyframe_lite_alignment: bool = False,
    keyframe_residual_generation: bool = False,
    diffusion_steps: int = 1,
    diffusion_time_sampling: str = "uniform",
    sampler_trajectory_alignment: bool = False,
) -> TinyTARDISAssembly:
    torch.manual_seed(17)
    priors = FrozenPriorBundle(
        TinyCodec(),
        TinyTextConditioner(),
        TinyFirstFrameGenerator(),
    )
    motion_teacher = ZeroMotionTeacher()
    state_updater = CausalStateUpdater(
        latent_channels=4,
        state_channels=8,
        anchor_decay=0.75,
    )
    model = TARDISModel(
        priors=priors,
        motion_teacher=motion_teacher,
        motion_scaffold=PromptMotionScaffold(
            text_dim=12,
            state_channels=8,
            noise_channels=3,
            hidden_size=16,
            motion_token_dim=10,
            token_stride=2,
            max_flow_pixels=2.0,
            num_time_frequencies=2,
        ),
        transport=MotionStateTransport(channels=4, max_correction_pixels=0.25),
        quotient=TransportOrbitProjector(),
        router=VisibilityCalibratedInnovationRouter(
            latent_channels=4,
            motion_channels=2,
            state_channels=8,
            text_dim=12,
            hidden_size=16,
            patch_size=2,
            active_ratio=active_ratio,
            threshold=0.1,
            halo_radius=0,
        ),
        innovation_clock=InnovationProperTime(
            patch_size=2,
            active_ratio=active_ratio,
            threshold=0.1,
            halo_radius=0,
        ),
        lite_corrector=LiteResidualCorrector(
            latent_channels=4,
            condition_channels=10,
            hidden_channels=8,
            max_magnitude=0.1,
            text_dim=12,
        ),
        transition_lite_corrector=LiteResidualCorrector(
            latent_channels=4,
            condition_channels=10,
            hidden_channels=8,
            max_magnitude=0.1,
            text_dim=12,
        ),
        keyframe_residual_dit=SparseResidualDiT(
            latent_channels=4,
            patch_size=2,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            text_dim=12,
            motion_dim=10,
            state_dim=8,
            max_grid_size=8,
        ),
        residual_dit=SparseResidualDiT(
            latent_channels=4,
            patch_size=2,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            text_dim=12,
            motion_dim=10,
            state_dim=8,
            max_grid_size=8,
        ),
        state_updater=state_updater,
        config=TARDISConfig(
            height=16,
            width=16,
            motion_noise_channels=3,
            state_token_stride=2,
            scene_cut_threshold=0.99,
            keyframe_lite_alignment=keyframe_lite_alignment,
            keyframe_residual_generation=keyframe_residual_generation,
            diffusion_steps=diffusion_steps,
            diffusion_time_sampling=diffusion_time_sampling,
            sampler_trajectory_alignment=sampler_trajectory_alignment,
        ),
    )
    return TinyTARDISAssembly(model, priors, motion_teacher)
