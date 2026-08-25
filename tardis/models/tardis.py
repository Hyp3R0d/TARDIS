"""Integrated TARDIS transport-aligned residual network over innovation subspaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import torch
from torch import nn

from tardis.models.clock import InnovationClockOutput, InnovationProperTime
from tardis.models.contracts import MotionTargets, MotionTeacher
from tardis.models.motion import MotionScaffoldOutput, PromptMotionScaffold
from tardis.models.priors import FrozenPriorBundle
from tardis.models.quotient import (
    QuotientDecomposition,
    TransportOrbitBasis,
    TransportOrbitProjector,
)
from tardis.models.residual import (
    LiteResidualCorrector,
    ResidualDenoisingContext,
    SparseResidualDiT,
    SparseResidualOutput,
)
from tardis.models.router import (
    InnovationSelection,
    RouterOutput,
    VisibilityCalibratedInnovationRouter,
    oracle_innovation,
)
from tardis.models.state import CausalState, CausalStateUpdater
from tardis.models.transport import MotionStateTransport, TransportOutput


class AblationVariant(StrEnum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    A7 = "A7"
    A8 = "A8"
    A9 = "A9"
    A10 = "A10"


@dataclass(frozen=True, slots=True)
class TARDISAblationFlags:
    previous_frame_conditioning: bool
    temporal_residual: bool
    source_motion_transport: bool
    analytical_visibility: bool
    learned_vcir: bool
    dual_frequency_residual: bool
    fixed_budget_routing: bool
    innovation_proper_time: bool
    crcd: bool
    metric_alignment: bool

    @classmethod
    def for_variant(cls, variant: AblationVariant) -> TARDISAblationFlags:
        level = int(variant.value[1:])
        enabled = tuple(level >= threshold for threshold in range(1, 11))
        return cls(*enabled)


@dataclass(frozen=True, slots=True)
class TARDISConfig:
    height: int = 512
    width: int = 512
    motion_noise_channels: int = 4
    state_token_stride: int = 4
    scene_cut_threshold: float = 0.98
    oracle_temperature: float = 0.25
    training_noise_scale: float = 0.1
    transport_quotient: bool = True
    innovation_proper_time: bool = True
    prior_anchored_training: bool = True
    keyframe_lite_alignment: bool = False
    keyframe_residual_generation: bool = False
    diffusion_steps: int = 1
    diffusion_time_sampling: str = "uniform"
    sampler_trajectory_alignment: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.height,
                self.width,
                self.motion_noise_channels,
                self.state_token_stride,
                self.diffusion_steps,
            )
            <= 0
        ):
            raise ValueError("TARDIS dimensions must be positive")
        if not 0 < self.scene_cut_threshold <= 1:
            raise ValueError("scene_cut_threshold must be in (0, 1]")
        if self.oracle_temperature <= 0 or self.training_noise_scale < 0:
            raise ValueError("oracle temperature and training noise scale are invalid")
        if any(
            not isinstance(value, bool)
            for value in (
                self.transport_quotient,
                self.innovation_proper_time,
                self.prior_anchored_training,
                self.keyframe_lite_alignment,
                self.keyframe_residual_generation,
                self.sampler_trajectory_alignment,
            )
        ):
            raise TypeError("TARDIS mechanism flags must be boolean")
        if self.diffusion_time_sampling not in {"uniform", "high_noise", "endpoint"}:
            raise ValueError(
                "diffusion_time_sampling must be one of: uniform, high_noise, endpoint"
            )
        if self.sampler_trajectory_alignment and self.diffusion_time_sampling != "endpoint":
            raise ValueError(
                "sampler trajectory alignment requires endpoint diffusion time sampling"
            )


@dataclass(frozen=True, slots=True)
class TARDISTrainingBatch:
    prompts: list[str]
    video: torch.Tensor


@dataclass(frozen=True, slots=True)
class TransitionConditions:
    text_embeddings: torch.Tensor
    text_mask: torch.Tensor
    time: torch.Tensor
    motion_noise: torch.Tensor
    diffusion_noise: torch.Tensor
    diffusion_time: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None
    teacher_backward_flow: torch.Tensor | None = None
    teacher_visibility: torch.Tensor | None = None
    teacher_motion_ratio: float = 1.0
    scene_cut_mask: torch.Tensor | None = None
    use_oracle_routing: bool = False
    detach_state: bool = False


@dataclass(frozen=True, slots=True)
class TARDISTransitionOutput:
    latent: torch.Tensor
    transport: TransportOutput
    lite_residual: torch.Tensor
    sparse_residual: torch.Tensor
    router: RouterOutput
    clock: InnovationClockOutput
    predicted_clock: InnovationClockOutput
    oracle_clock: InnovationClockOutput | None
    oracle_probability: torch.Tensor | None
    motion: MotionScaffoldOutput
    state: CausalState
    scene_cut_mask: torch.Tensor
    quotient_basis: TransportOrbitBasis | None
    tangent_target: torch.Tensor | None
    quotient_tangent_rank: torch.Tensor
    residual_target: torch.Tensor | None
    noisy_residual: torch.Tensor
    residual_context: ResidualDenoisingContext


@dataclass(frozen=True, slots=True)
class TARDISTrainOutput:
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    transitions: tuple[TARDISTransitionOutput, ...]
    motion_targets: MotionTargets
    teacher_forcing_mask: torch.Tensor
    stage: str
    keyframe_prior: torch.Tensor | None = None
    keyframe_lite_residual: torch.Tensor | None = None
    keyframe_residual_base: torch.Tensor | None = None
    keyframe_residual: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TARDISKeyframeTrainOutput:
    predicted_latent: torch.Tensor
    target_latent: torch.Tensor
    keyframe_prior: torch.Tensor
    keyframe_lite_residual: torch.Tensor
    keyframe_residual_base: torch.Tensor
    keyframe_residual: torch.Tensor
    stage: str = "keyframe_only"


@dataclass(frozen=True, slots=True)
class TARDISVideoOutput:
    video: torch.Tensor
    latents: torch.Tensor
    transitions: tuple[TARDISTransitionOutput, ...]
    fps: int


class TARDISModel(nn.Module):
    """Generate only innovation that cannot be reused after causal state transport."""

    def __init__(
        self,
        *,
        priors: FrozenPriorBundle,
        motion_teacher: MotionTeacher | None,
        motion_scaffold: PromptMotionScaffold,
        transport: MotionStateTransport,
        quotient: TransportOrbitProjector,
        router: VisibilityCalibratedInnovationRouter,
        innovation_clock: InnovationProperTime,
        lite_corrector: LiteResidualCorrector,
        transition_lite_corrector: LiteResidualCorrector,
        keyframe_residual_dit: SparseResidualDiT,
        residual_dit: SparseResidualDiT,
        state_updater: CausalStateUpdater,
        config: TARDISConfig,
        ablation: AblationVariant = AblationVariant.A10,
    ) -> None:
        super().__init__()
        if config.height % priors.spatial_scale or config.width % priors.spatial_scale:
            raise ValueError("output dimensions must be divisible by the codec spatial scale")
        self.priors = priors
        self.motion_teacher = motion_teacher
        self.motion_scaffold = motion_scaffold
        self.transport = transport
        self.quotient = quotient
        self.router = router
        self.innovation_clock = innovation_clock
        self.lite_corrector = lite_corrector
        self.transition_lite_corrector = transition_lite_corrector
        self.keyframe_residual_dit = keyframe_residual_dit
        self.residual_dit = residual_dit
        self.state_updater = state_updater
        self.config = config
        self.ablation = ablation
        self.ablation_flags = TARDISAblationFlags.for_variant(ablation)

    def forward(
        self,
        batch: TARDISTrainingBatch,
        stage: str,
        *,
        teacher_forcing_ratio: float = 1.0,
        generator: torch.Generator | None = None,
        train_mode: str = "full_temporal",
    ) -> TARDISTrainOutput | TARDISKeyframeTrainOutput:
        """DDP-compatible entry point for the causal training rollout."""

        if train_mode == "keyframe_only":
            return self.forward_keyframe_train(batch, generator=generator)
        if train_mode != "full_temporal":
            raise ValueError(f"unsupported TARDIS train mode: {train_mode!r}")
        return self.forward_train(
            batch,
            stage,
            teacher_forcing_ratio=teacher_forcing_ratio,
            generator=generator,
        )

    def forward_keyframe_train(
        self,
        batch: TARDISTrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> TARDISKeyframeTrainOutput:
        """Train the deployed prompt keyframe path without temporal rollout work."""

        if not self.config.keyframe_lite_alignment or not self.config.keyframe_residual_generation:
            raise RuntimeError(
                "keyframe-only training requires keyframe lite alignment and residual generation"
            )
        if batch.video.ndim != 5 or batch.video.shape[2] != 3 or batch.video.shape[1] < 1:
            raise ValueError("keyframe training video must have shape [B,T>=1,3,H,W]")
        if batch.video.shape[0] != len(batch.prompts):
            raise ValueError("training prompts must match the keyframe batch")
        if batch.video.shape[-2:] != (self.config.height, self.config.width):
            raise ValueError("training video dimensions do not match TARDISConfig")

        target_latent = self.priors.encode_video(batch.video[:, :1])[:, 0]
        text_embeddings, text_mask = self.priors.encode_text(batch.prompts)
        prior_generator = generator
        if prior_generator is None:
            prior_generator = torch.Generator(device=target_latent.device).manual_seed(
                torch.initial_seed()
            )
        keyframe_prior = self.priors.generate_first_latent(
            text_embeddings,
            text_mask,
            generator=prior_generator,
            height=self.config.height,
            width=self.config.width,
        )
        keyframe_residual_base, keyframe_lite_residual = self._align_keyframe(
            keyframe_prior,
            text_embeddings=text_embeddings,
            text_mask=text_mask,
        )
        keyframe_residual = self._generate_keyframe_residual(
            keyframe_residual_base,
            text_embeddings=text_embeddings,
            text_mask=text_mask,
            generator=generator,
            target_latent=target_latent,
        )
        return TARDISKeyframeTrainOutput(
            predicted_latent=keyframe_residual_base + keyframe_residual,
            target_latent=target_latent,
            keyframe_prior=keyframe_prior,
            keyframe_lite_residual=keyframe_lite_residual,
            keyframe_residual_base=keyframe_residual_base,
            keyframe_residual=keyframe_residual,
        )

    def transition(
        self,
        previous: CausalState,
        conditions: TransitionConditions,
    ) -> TARDISTransitionOutput:
        self._validate_transition(previous, conditions)
        motion = self.motion_scaffold(
            conditions.text_embeddings,
            conditions.text_mask,
            time=conditions.time,
            state=previous.spatial_condition(),
            motion_noise=conditions.motion_noise,
        )
        scaffold_visibility = motion.visibility_logits.sigmoid()
        teacher_ratio = conditions.teacher_motion_ratio
        if conditions.teacher_backward_flow is not None:
            teacher_flow = conditions.teacher_backward_flow.to(
                device=motion.backward_flow.device,
                dtype=motion.backward_flow.dtype,
            )
            flow = torch.lerp(
                motion.backward_flow,
                teacher_flow,
                teacher_ratio,
            )
        else:
            flow = motion.backward_flow
        if conditions.teacher_visibility is not None:
            teacher_visibility = conditions.teacher_visibility.to(
                device=scaffold_visibility.device,
                dtype=scaffold_visibility.dtype,
            )
            visibility = torch.lerp(
                scaffold_visibility,
                teacher_visibility,
                teacher_ratio,
            )
        else:
            visibility = scaffold_visibility
        if not self.ablation_flags.source_motion_transport:
            flow = torch.zeros_like(flow)
        if not self.ablation_flags.analytical_visibility:
            visibility = torch.ones_like(visibility)
        use_transport_quotient = (
            self.config.transport_quotient and self.ablation_flags.analytical_visibility
        )

        transport = self.transport(
            previous.latent,
            flow,
            visibility,
            state={
                "short": previous.short,
                "anchor": previous.anchor,
                "innovation_hazard": previous.innovation_hazard,
            },
        )
        transported_state = CausalState(
            latent=transport.prior,
            short=transport.warped_state["short"],
            anchor=transport.warped_state["anchor"],
            innovation_hazard=transport.warped_state["innovation_hazard"],
            frame_index=previous.frame_index,
        )
        quotient_basis: TransportOrbitBasis | None = None
        if use_transport_quotient:
            quotient_basis = self.quotient.build_basis(
                transport.prior,
                transport.effective_visibility,
            )
        target_decomposition: QuotientDecomposition | None = None
        quotient_residual: torch.Tensor | None = None
        if conditions.target_latent is not None:
            transport_residual = conditions.target_latent - transport.prior.detach()
            if use_transport_quotient:
                assert quotient_basis is not None
                target_decomposition = self.quotient.decompose(
                    transport_residual,
                    quotient_basis,
                )
                quotient_residual = target_decomposition.innovation
            else:
                quotient_residual = transport_residual
        router = self.router(
            transport.prior,
            transport.effective_visibility,
            flow,
            transported_state.short,
            conditions.text_embeddings,
            conditions.text_mask,
        )

        oracle_probability: torch.Tensor | None = None
        if conditions.target_latent is not None:
            oracle_probability = oracle_innovation(
                conditions.target_latent,
                transport.prior,
                transport.effective_visibility,
                residual_temperature=self.config.oracle_temperature,
                quotient_residual=quotient_residual,
            )
        use_proper_time = (
            self.config.innovation_proper_time and self.ablation_flags.innovation_proper_time
        )
        transported_hazard = (
            transported_state.innovation_hazard
            if use_proper_time
            else torch.zeros_like(transported_state.innovation_hazard)
        )
        budget_ratio = self.router.active_ratio if self.ablation_flags.fixed_budget_routing else 1.0
        predicted_clock = self.innovation_clock(
            transported_hazard,
            router.pixel_probability,
            transport.effective_visibility,
            active_ratio=budget_ratio,
        )
        oracle_clock: InnovationClockOutput | None = None
        if oracle_probability is not None:
            oracle_clock = self.innovation_clock(
                transported_hazard,
                oracle_probability,
                transport.effective_visibility,
                active_ratio=budget_ratio,
            )
        if conditions.use_oracle_routing:
            if oracle_clock is None:
                raise ValueError("oracle routing requires target_latent")
            clock = oracle_clock
        else:
            clock = predicted_clock
        router = replace(router, selection=clock.selection)

        update_probability = (
            oracle_probability
            if conditions.use_oracle_routing and oracle_probability is not None
            else router.pixel_probability
        )
        lite_condition = torch.cat((flow, transported_state.short), dim=1)
        transition_lite_enabled = self.ablation_flags.dual_frequency_residual
        if transition_lite_enabled:
            transition_corrector = (
                self.transition_lite_corrector
                if self.config.keyframe_lite_alignment
                or self.config.keyframe_residual_generation
                else self.lite_corrector
            )
            raw_lite_residual = transition_corrector(
                transport.prior,
                lite_condition,
                update_probability,
                text_embeddings=conditions.text_embeddings,
                text_mask=conditions.text_mask,
            )
            if use_transport_quotient:
                assert quotient_basis is not None
                lite_residual = self.quotient.decompose(
                    raw_lite_residual,
                    quotient_basis,
                ).tangent
            else:
                lite_residual = raw_lite_residual * (1 - update_probability)
        else:
            lite_residual = torch.zeros_like(transport.prior)

        tangent_target: torch.Tensor | None = None
        residual_target: torch.Tensor | None = None
        if use_transport_quotient:
            assert quotient_basis is not None
            noise_decomposition = self.quotient.decompose(
                conditions.diffusion_noise,
                quotient_basis,
            )
            diffusion_noise = noise_decomposition.innovation
            quotient_tangent_rank = noise_decomposition.tangent_rank
        else:
            diffusion_noise = conditions.diffusion_noise
            quotient_tangent_rank = torch.zeros_like(transport.effective_visibility)
        noisy_residual = diffusion_noise
        if conditions.target_latent is not None:
            if target_decomposition is not None:
                tangent_target = target_decomposition.tangent
                residual_target = target_decomposition.innovation
                quotient_tangent_rank = target_decomposition.tangent_rank
            else:
                residual_target = (
                    conditions.target_latent - transport.prior.detach() - lite_residual
                )
            diffusion_time = (
                conditions.time if conditions.diffusion_time is None else conditions.diffusion_time
            )
            interpolation = diffusion_time[:, None, None, None]
            noisy_residual = (1 - interpolation) * residual_target + interpolation * diffusion_noise
        else:
            diffusion_time = (
                torch.ones_like(conditions.time)
                if conditions.diffusion_time is None
                else conditions.diffusion_time
            )
        residual_context = ResidualDenoisingContext(
            noisy_residual=noisy_residual,
            diffusion_noise=diffusion_noise,
            transported_prior=transport.prior,
            diffusion_time=diffusion_time,
            event_probability=clock.event_probability,
            text_tokens=conditions.text_embeddings,
            text_mask=conditions.text_mask,
            motion_tokens=motion.motion_tokens,
            state_tokens=transported_state.anchor_tokens(stride=self.config.state_token_stride),
            selection=router.selection,
        )
        use_sampling_trajectory = (
            conditions.target_latent is None or self.config.sampler_trajectory_alignment
        )
        sparse_output = (
            self._sample_residual(residual_context, self.residual_dit)
            if use_sampling_trajectory
            else residual_context.predict(self.residual_dit)
        )
        if self.ablation_flags.temporal_residual:
            sparse_residual = sparse_output.residual
            if use_transport_quotient:
                assert quotient_basis is not None
                sparse_residual = self.quotient.decompose(
                    sparse_residual,
                    quotient_basis,
                ).innovation
        else:
            sparse_residual = torch.zeros_like(transport.prior)
        latent = transport.prior + lite_residual + sparse_residual

        scene_cut_mask = self._scene_cut_mask(update_probability, conditions.scene_cut_mask)
        service_probability = clock.event_probability * clock.service_mask.to(
            clock.event_probability.dtype
        )
        state = self.state_updater.update(
            transported_state,
            latent,
            innovation_probability=service_probability,
            innovation_hazard=clock.settled_hazard,
            reset_mask=scene_cut_mask,
            detach=conditions.detach_state,
        )
        return TARDISTransitionOutput(
            latent=latent,
            transport=transport,
            lite_residual=lite_residual,
            sparse_residual=sparse_residual,
            router=router,
            clock=clock,
            predicted_clock=predicted_clock,
            oracle_clock=oracle_clock,
            oracle_probability=oracle_probability,
            motion=motion,
            state=state,
            scene_cut_mask=scene_cut_mask,
            quotient_basis=quotient_basis,
            tangent_target=tangent_target,
            quotient_tangent_rank=quotient_tangent_rank,
            residual_target=residual_target,
            noisy_residual=noisy_residual,
            residual_context=residual_context,
        )

    def forward_train(
        self,
        batch: TARDISTrainingBatch,
        stage: str,
        *,
        teacher_forcing_ratio: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> TARDISTrainOutput:
        if not stage.strip():
            raise ValueError("training stage must be non-empty")
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if batch.video.ndim != 5 or batch.video.shape[2] != 3:
            raise ValueError("training video must have shape [B,T,3,H,W]")
        if batch.video.shape[0] != len(batch.prompts) or batch.video.shape[1] < 2:
            raise ValueError("training prompts must match a video batch with at least two frames")
        if batch.video.shape[-2:] != (self.config.height, self.config.width):
            raise ValueError("training video dimensions do not match TARDISConfig")
        if self.motion_teacher is None:
            raise RuntimeError("forward_train requires a motion teacher")

        target_latents = self.priors.encode_video(batch.video)
        text_embeddings, text_mask = self.priors.encode_text(batch.prompts)
        motion_targets = self.motion_teacher.estimate(
            batch.video,
            output_size=(int(target_latents.shape[-2]), int(target_latents.shape[-1])),
        )
        batch_size = batch.video.shape[0]
        initial_teacher_forcing = _teacher_forcing_mask(
            batch_size,
            1,
            ratio=teacher_forcing_ratio,
            device=target_latents.device,
            generator=generator,
        )[:, 0]
        initial_latent = target_latents[:, 0]
        keyframe_prior: torch.Tensor | None = None
        keyframe_lite_residual: torch.Tensor | None = None
        keyframe_residual_base: torch.Tensor | None = None
        keyframe_residual: torch.Tensor | None = None
        needs_prompt_latent = (
            self.config.keyframe_lite_alignment
            or self.config.keyframe_residual_generation
            or (
            self.config.prior_anchored_training
            and not bool(initial_teacher_forcing.all().item())
            )
        )
        if needs_prompt_latent:
            prior_generator = generator
            if prior_generator is None:
                prior_generator = torch.Generator(device=target_latents.device).manual_seed(
                    torch.initial_seed()
                )
            prompt_latent = self.priors.generate_first_latent(
                text_embeddings,
                text_mask,
                generator=prior_generator,
                height=self.config.height,
                width=self.config.width,
            )
            if self.config.keyframe_lite_alignment:
                keyframe_prior = prompt_latent
                prompt_latent, keyframe_lite_residual = self._align_keyframe(
                    prompt_latent,
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                )
            if self.config.keyframe_residual_generation:
                if keyframe_prior is None:
                    keyframe_prior = prompt_latent
                keyframe_residual_base = prompt_latent
                keyframe_residual = self._generate_keyframe_residual(
                    keyframe_residual_base,
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                    generator=generator,
                    target_latent=target_latents[:, 0],
                )
                prompt_latent = prompt_latent + keyframe_residual
                # Deployment always starts from the generated keyframe. Keeping this
                # state during training removes the frame-zero exposure mismatch.
                initial_latent = prompt_latent
            elif self.config.prior_anchored_training:
                initial_latent = torch.where(
                    initial_teacher_forcing[:, None, None, None],
                    initial_latent,
                    prompt_latent,
                )
        state = self.state_updater.initialize(initial_latent, detach=False)
        predicted = [initial_latent]
        transitions: list[TARDISTransitionOutput] = []
        teacher_oracle_routing = stage in {
            "transport_warmup",
            "router_calibration",
            "residual_teacher",
        }
        teacher_forcing_mask = _teacher_forcing_mask(
            batch_size,
            batch.video.shape[1] - 1,
            ratio=teacher_forcing_ratio,
            device=target_latents.device,
            generator=generator,
        )
        for frame_index in range(1, batch.video.shape[1]):
            time = torch.full(
                (batch_size,),
                frame_index / max(batch.video.shape[1] - 1, 1),
                device=target_latents.device,
                dtype=target_latents.dtype,
            )
            output = self.transition(
                state,
                TransitionConditions(
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                    time=time,
                    motion_noise=_randn_spatial(
                        target_latents,
                        channels=self.config.motion_noise_channels,
                        generator=generator,
                    )
                    * self.config.training_noise_scale,
                    diffusion_noise=_randn_like(
                        target_latents[:, frame_index],
                        generator=generator,
                    ),
                    diffusion_time=self._sample_training_diffusion_time(
                        batch_size,
                        reference=target_latents,
                        generator=generator,
                    ),
                    target_latent=target_latents[:, frame_index],
                    teacher_backward_flow=(
                        motion_targets.backward_flow[:, frame_index - 1]
                    ),
                    teacher_visibility=(
                        motion_targets.visibility[:, frame_index - 1]
                    ),
                    teacher_motion_ratio=teacher_forcing_ratio,
                    scene_cut_mask=torch.zeros(
                        batch_size,
                        device=target_latents.device,
                        dtype=torch.bool,
                    ),
                    use_oracle_routing=teacher_oracle_routing,
                    detach_state=False,
                ),
            )
            predicted.append(output.latent)
            transitions.append(output)
            reference_state = self.state_updater.initialize(
                target_latents[:, frame_index],
                detach=False,
            )
            state = _blend_states(
                output.state,
                reference_state,
                teacher_forcing_mask[:, frame_index - 1],
            )
        return TARDISTrainOutput(
            predicted_latents=torch.stack(predicted, dim=1),
            target_latents=target_latents,
            transitions=tuple(transitions),
            motion_targets=motion_targets,
            teacher_forcing_mask=teacher_forcing_mask,
            stage=stage,
            keyframe_prior=keyframe_prior,
            keyframe_lite_residual=keyframe_lite_residual,
            keyframe_residual_base=keyframe_residual_base,
            keyframe_residual=keyframe_residual,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        num_frames: int,
        fps: int,
        generator: torch.Generator | Sequence[torch.Generator],
    ) -> TARDISVideoOutput:
        if num_frames <= 0 or fps <= 0:
            raise ValueError("num_frames and fps must be positive")
        text_embeddings, text_mask = self.priors.encode_text(prompts)
        pipeline_generator = (
            generator if isinstance(generator, torch.Generator) else list(generator)
        )
        first_latent = self.priors.generate_first_latent(
            text_embeddings,
            text_mask,
            generator=pipeline_generator,
            height=self.config.height,
            width=self.config.width,
        )
        if self.config.keyframe_lite_alignment:
            first_latent, _ = self._align_keyframe(
                first_latent,
                text_embeddings=text_embeddings,
                text_mask=text_mask,
            )
        if self.config.keyframe_residual_generation:
            first_latent = first_latent + self._generate_keyframe_residual(
                first_latent,
                text_embeddings=text_embeddings,
                text_mask=text_mask,
                generator=generator,
            )
        state = self.state_updater.initialize(first_latent, detach=True)
        latents = [first_latent]
        transitions: list[TARDISTransitionOutput] = []
        for frame_index in range(1, num_frames):
            time = torch.full(
                (len(prompts),),
                frame_index / max(num_frames - 1, 1),
                device=first_latent.device,
                dtype=first_latent.dtype,
            )
            output = self.transition(
                state,
                TransitionConditions(
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                    time=time,
                    motion_noise=_randn_like_channels(
                        first_latent,
                        channels=self.config.motion_noise_channels,
                        generator=generator,
                    )
                    * self.config.training_noise_scale,
                    diffusion_noise=_randn_like_per_sample(
                        first_latent,
                        generator=generator,
                    ),
                    diffusion_time=torch.ones_like(time),
                    scene_cut_mask=torch.zeros(
                        len(prompts),
                        device=first_latent.device,
                        dtype=torch.bool,
                    ),
                    detach_state=True,
                ),
            )
            latents.append(output.latent)
            transitions.append(output)
            state = output.state
        stacked_latents = torch.stack(latents, dim=1)
        return TARDISVideoOutput(
            video=self.priors.decode_video(stacked_latents).clamp(-1, 1),
            latents=stacked_latents,
            transitions=tuple(transitions),
            fps=fps,
        )

    @torch.no_grad()
    def generate_source_conditioned(
        self,
        prompts: list[str],
        source_video: torch.Tensor,
        *,
        fps: int,
        generator: torch.Generator | Sequence[torch.Generator],
        innovation_strength: float = 0.25,
    ) -> TARDISVideoOutput:
        """Generate bounded innovations over source-motion-aligned latent states."""

        if source_video.ndim != 5 or source_video.shape[2] != 3:
            raise ValueError("source_video must have shape [B,T,3,H,W]")
        if source_video.shape[0] != len(prompts):
            raise ValueError("prompts must match the source_video batch size")
        if source_video.shape[1] < 2:
            raise ValueError("source-conditioned generation requires at least two frames")
        if source_video.shape[-2:] != (self.config.height, self.config.width):
            raise ValueError("source_video dimensions do not match TARDISConfig")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= innovation_strength <= 1.0:
            raise ValueError("innovation_strength must be in [0, 1]")
        if self.motion_teacher is None:
            raise RuntimeError("source-conditioned generation requires a motion teacher")

        source_latents = self.priors.encode_video(source_video)
        text_embeddings, text_mask = self.priors.encode_text(prompts)
        motion = self.motion_teacher.estimate(
            source_video,
            output_size=(int(source_latents.shape[-2]), int(source_latents.shape[-1])),
        )
        state = self.state_updater.initialize(source_latents[:, 0], detach=True)
        raw_latents = [source_latents[:, 0]]
        transitions: list[TARDISTransitionOutput] = []
        shared_motion_noise = _randn_like_channels(
            source_latents[:, 0],
            channels=self.config.motion_noise_channels,
            generator=generator,
        ) * self.config.training_noise_scale
        frame_count = int(source_latents.shape[1])
        for frame_index in range(1, frame_count):
            time = source_latents.new_full(
                (source_latents.shape[0],),
                frame_index / max(frame_count - 1, 1),
            )
            output = self.transition(
                state,
                TransitionConditions(
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                    time=time,
                    motion_noise=shared_motion_noise,
                    diffusion_noise=_randn_like_per_sample(
                        source_latents[:, frame_index],
                        generator=generator,
                    ),
                    diffusion_time=torch.ones_like(time),
                    teacher_backward_flow=motion.backward_flow[:, frame_index - 1],
                    teacher_visibility=motion.visibility[:, frame_index - 1],
                    teacher_motion_ratio=1.0,
                    scene_cut_mask=torch.zeros(
                        source_latents.shape[0],
                        device=source_latents.device,
                        dtype=torch.bool,
                    ),
                    detach_state=True,
                ),
            )
            raw_latents.append(output.latent)
            transitions.append(output)
            state = output.state

        raw = torch.stack(raw_latents, dim=1)
        bounded = torch.lerp(source_latents, raw, innovation_strength)
        return TARDISVideoOutput(
            video=self.priors.decode_video(bounded).clamp(-1, 1),
            latents=bounded,
            transitions=tuple(transitions),
            fps=fps,
        )

    def _align_keyframe(
        self,
        keyframe_prior: torch.Tensor,
        *,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state_updater.initialize(keyframe_prior, detach=False)
        zero_flow = keyframe_prior.new_zeros(
            keyframe_prior.shape[0],
            2,
            *keyframe_prior.shape[-2:],
        )
        condition = torch.cat((zero_flow, state.short), dim=1)
        innovation = keyframe_prior.new_ones(
            keyframe_prior.shape[0],
            1,
            *keyframe_prior.shape[-2:],
        )
        residual = self.lite_corrector(
            keyframe_prior,
            condition,
            innovation,
            text_embeddings=text_embeddings,
            text_mask=text_mask,
        )
        return keyframe_prior + residual, residual

    def _generate_keyframe_residual(
        self,
        keyframe_base: torch.Tensor,
        *,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        generator: torch.Generator | Sequence[torch.Generator] | None,
        target_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, _, height, width = keyframe_base.shape
        selection = _full_innovation_selection(
            batch,
            height,
            width,
            patch_size=self.keyframe_residual_dit.patch_size,
            device=keyframe_base.device,
        )
        state = self.state_updater.initialize(keyframe_base, detach=False)
        diffusion_noise = (
            _randn_like_per_sample(keyframe_base, generator=generator)
            if generator is not None
            else torch.randn_like(keyframe_base)
        )
        if target_latent is None:
            diffusion_time = keyframe_base.new_ones(batch)
            noisy_residual = diffusion_noise
        else:
            diffusion_time = self._sample_training_diffusion_time(
                batch,
                reference=keyframe_base,
                generator=generator if isinstance(generator, torch.Generator) else None,
            )
            residual_target = target_latent - keyframe_base.detach()
            interpolation = diffusion_time[:, None, None, None]
            noisy_residual = (
                (1 - interpolation) * residual_target + interpolation * diffusion_noise
            )
        context = ResidualDenoisingContext(
            noisy_residual=noisy_residual,
            diffusion_noise=diffusion_noise,
            transported_prior=keyframe_base,
            diffusion_time=diffusion_time,
            event_probability=keyframe_base.new_ones(batch, 1, height, width),
            text_tokens=text_embeddings,
            text_mask=text_mask,
            motion_tokens=keyframe_base.new_zeros(
                batch,
                1,
                self.keyframe_residual_dit.motion_projection.in_features,
            ),
            state_tokens=state.anchor_tokens(stride=self.config.state_token_stride),
            selection=selection,
        )
        if target_latent is None or self.config.sampler_trajectory_alignment:
            return self._sample_residual(context, self.keyframe_residual_dit).residual
        return context.predict(self.keyframe_residual_dit).residual

    def _sample_training_diffusion_time(
        self,
        batch_size: int,
        *,
        reference: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.config.diffusion_time_sampling == "endpoint":
            return reference.new_ones(batch_size)
        diffusion_time = torch.rand(
            batch_size,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        if self.config.diffusion_time_sampling == "high_noise":
            diffusion_time = 0.5 + 0.5 * diffusion_time
        return diffusion_time

    def _sample_residual(
        self,
        context: ResidualDenoisingContext,
        denoiser: SparseResidualDiT,
    ) -> SparseResidualOutput:
        noisy = context.noisy_residual
        output = context.predict(denoiser)
        trajectory = [output.residual]
        for step in range(1, self.config.diffusion_steps):
            next_time_value = 1 - step / (self.config.diffusion_steps - 1)
            next_time = context.diffusion_time.new_full(
                context.diffusion_time.shape,
                next_time_value,
            )
            interpolation = next_time[:, None, None, None]
            noisy = (1 - interpolation) * output.residual + (
                interpolation * context.diffusion_noise
            )
            output = context.predict(
                denoiser,
                noisy_residual=noisy,
                diffusion_time=next_time,
            )
            trajectory.append(output.residual)
        if self.config.sampler_trajectory_alignment and torch.is_grad_enabled():
            # Preserve the exact deployed final-step value while directly supervising
            # every preceding step, including at zero-initialized denoiser heads.
            gradient_bridge = sum(
                (residual - residual.detach() for residual in trajectory[:-1]),
                torch.zeros_like(output.residual),
            ) / max(len(trajectory) - 1, 1)
            output = replace(output, residual=output.residual + gradient_bridge)
        return output

    def _scene_cut_mask(
        self,
        innovation_probability: torch.Tensor,
        explicit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        inferred = innovation_probability.mean(dim=(1, 2, 3)) >= self.config.scene_cut_threshold
        if explicit_mask is None:
            return inferred
        return inferred | explicit_mask

    def _validate_transition(
        self,
        previous: CausalState,
        conditions: TransitionConditions,
    ) -> None:
        batch, channels, height, width = previous.latent.shape
        if channels != self.priors.latent_channels:
            raise ValueError("causal latent channel count does not match the frozen codec")
        if conditions.text_embeddings.ndim != 3 or conditions.text_embeddings.shape[0] != batch:
            raise ValueError("transition text embeddings must have shape [B,L,D]")
        if conditions.text_mask.shape != conditions.text_embeddings.shape[:2]:
            raise ValueError("transition text mask must have shape [B,L]")
        if conditions.time.shape != (batch,):
            raise ValueError("transition time must have shape [B]")
        expected_motion = (batch, self.config.motion_noise_channels, height, width)
        if conditions.motion_noise.shape != expected_motion:
            raise ValueError(f"motion noise must have shape {expected_motion}")
        if conditions.diffusion_noise.shape != previous.latent.shape:
            raise ValueError("diffusion noise must match the causal latent")
        if conditions.diffusion_time is not None:
            if conditions.diffusion_time.shape != (batch,):
                raise ValueError("diffusion_time must have shape [B]")
            if not bool(
                ((conditions.diffusion_time >= 0) & (conditions.diffusion_time <= 1)).all().item()
            ):
                raise ValueError("diffusion_time must lie in [0, 1]")
        for tensor, name, expected_channels in (
            (conditions.teacher_backward_flow, "teacher backward flow", 2),
            (conditions.teacher_visibility, "teacher visibility", 1),
        ):
            if tensor is not None and tensor.shape != (batch, expected_channels, height, width):
                raise ValueError(f"{name} has incompatible shape")
        if conditions.target_latent is not None and (
            conditions.target_latent.shape != previous.latent.shape
        ):
            raise ValueError("target latent must match the causal latent")
        if conditions.scene_cut_mask is not None and (
            conditions.scene_cut_mask.shape != (batch,)
            or conditions.scene_cut_mask.dtype != torch.bool
        ):
            raise ValueError("scene_cut_mask must be bool [B]")
        if not 0.0 <= conditions.teacher_motion_ratio <= 1.0:
            raise ValueError("teacher_motion_ratio must be in [0, 1]")


def _randn_like_channels(
    reference: torch.Tensor,
    *,
    channels: int,
    generator: torch.Generator | Sequence[torch.Generator],
) -> torch.Tensor:
    shape = (reference.shape[0], channels, *reference.shape[-2:])
    return _randn_per_sample(shape, reference, generator)


def _full_innovation_selection(
    batch_size: int,
    height: int,
    width: int,
    *,
    patch_size: int,
    device: torch.device,
) -> InnovationSelection:
    if height % patch_size or width % patch_size:
        raise ValueError("keyframe latent geometry must be divisible by patch_size")
    patch_height = height // patch_size
    patch_width = width // patch_size
    token_count = patch_height * patch_width
    indices = torch.arange(token_count, device=device).expand(batch_size, -1)
    valid = torch.ones(batch_size, token_count, device=device, dtype=torch.bool)
    return InnovationSelection(
        indices=indices,
        valid_tokens=valid,
        active_counts=torch.full(
            (batch_size,),
            token_count,
            device=device,
            dtype=torch.long,
        ),
        active_mask=torch.ones(
            batch_size,
            1,
            patch_height,
            patch_width,
            device=device,
            dtype=torch.bool,
        ),
    )


def _randn_like_per_sample(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | Sequence[torch.Generator],
) -> torch.Tensor:
    return _randn_per_sample(tuple(reference.shape), reference, generator)


def _randn_per_sample(
    shape: tuple[int, ...],
    reference: torch.Tensor,
    generator: torch.Generator | Sequence[torch.Generator],
) -> torch.Tensor:
    if isinstance(generator, torch.Generator):
        return torch.randn(
            shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
    generators = tuple(generator)
    if len(generators) != shape[0]:
        raise ValueError("per-sample generator count must match the batch size")
    return torch.cat(
        [
            torch.randn(
                (1, *shape[1:]),
                device=reference.device,
                dtype=reference.dtype,
                generator=sample_generator,
            )
            for sample_generator in generators
        ],
        dim=0,
    )


def _randn_spatial(
    reference: torch.Tensor,
    *,
    channels: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape[0],
        channels,
        *reference.shape[-2:],
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _randn_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _teacher_forcing_mask(
    batch_size: int,
    transitions: int,
    *,
    ratio: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if ratio == 0:
        return torch.zeros(batch_size, transitions, device=device, dtype=torch.bool)
    if ratio == 1:
        return torch.ones(batch_size, transitions, device=device, dtype=torch.bool)
    return (
        torch.rand(
            batch_size,
            transitions,
            device=device,
            generator=generator,
        )
        < ratio
    )


def _blend_states(
    generated: CausalState,
    reference: CausalState,
    use_reference: torch.Tensor,
) -> CausalState:
    if use_reference.shape != generated.frame_index.shape or use_reference.dtype != torch.bool:
        raise ValueError("use_reference must be bool [B]")
    spatial_mask = use_reference[:, None, None, None]
    return CausalState(
        latent=torch.where(spatial_mask, reference.latent, generated.latent),
        short=torch.where(spatial_mask, reference.short, generated.short),
        anchor=torch.where(spatial_mask, reference.anchor, generated.anchor),
        innovation_hazard=torch.where(
            spatial_mask,
            reference.innovation_hazard,
            generated.innovation_hazard,
        ),
        frame_index=torch.where(
            use_reference,
            reference.frame_index,
            generated.frame_index,
        ),
    )
