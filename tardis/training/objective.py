"""Curriculum-aware objectives for the TARDIS main network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Protocol, cast

import torch
import torch.nn.functional as functional
from torch import nn

from tardis.models.residual import patchify
from tardis.models.tardis import (
    TARDISKeyframeTrainOutput,
    TARDISModel,
    TARDISTrainingBatch,
    TARDISTrainOutput,
)
from tardis.training.curriculum import CurriculumPoint, active_loss_names
from tardis.training.distillation import CausalResidualDistiller
from tardis.training.engine import ObjectiveOutput
from tardis.training.losses import (
    EmaLossNormalizer,
    LossWeights,
    PerceptualMetric,
    budget_loss,
    crcd_loss,
    diffusion_loss,
    drift_loss,
    flow_loss,
    lite_residual_loss,
    lpips_loss,
    multi_scale_temporal_consistency_loss,
    residual_reconstruction_loss,
    router_loss,
    survival_calibration_loss,
    text_alignment_loss,
    transport_loss,
    visibility_loss,
    warp_loss,
    weighted_loss,
)


class _VideoDecoder(Protocol):
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor: ...


class TARDISObjective:
    """Map transport-innovation rollouts to the staged paper objectives."""

    def __init__(
        self,
        *,
        perceptual_metric: PerceptualMetric,
        weights: LossWeights | None = None,
        normalizer_decay: float = 0.99,
        normalizer_epsilon: float = 1.0e-6,
        temporal_levels: int = 3,
        lpips_frame_chunk_size: int = 4,
        distiller: CausalResidualDistiller | None = None,
    ) -> None:
        if temporal_levels <= 0 or lpips_frame_chunk_size <= 0:
            raise ValueError("temporal levels and LPIPS frame chunk size must be positive")
        self.perceptual_metric = perceptual_metric
        self.weights = weights or LossWeights()
        self.temporal_levels = temporal_levels
        self.lpips_frame_chunk_size = lpips_frame_chunk_size
        self.distiller = distiller or CausalResidualDistiller()
        self.normalizers = {
            name: EmaLossNormalizer(
                decay=normalizer_decay,
                epsilon=normalizer_epsilon,
            )
            for name in asdict(self.weights)
        }

    def __call__(
        self,
        model: nn.Module,
        batch: object,
        point: CurriculumPoint,
        generator: torch.Generator,
    ) -> ObjectiveOutput:
        tardis_model = _unwrap_tardis_model(model)
        if not isinstance(batch, TARDISTrainingBatch):
            raise TypeError("TARDISObjective requires a TARDISTrainingBatch")
        rollout = model(
            batch,
            point.stage.value,
            teacher_forcing_ratio=point.teacher_forcing_ratio,
            generator=generator,
        )
        if not isinstance(rollout, TARDISTrainOutput):
            raise RuntimeError("TARDIS training forward returned an incompatible output")
        active = active_loss_names(point.stage)
        if tardis_model.config.keyframe_residual_generation:
            active.add("keyframe")
        candidates = self._candidate_losses(tardis_model, batch, rollout, active=active)
        missing = active - set(candidates)
        if missing:
            raise RuntimeError(f"objective did not construct active losses: {sorted(missing)}")
        selected = {name: candidates[name] for name in sorted(active)}
        normalized = {
            name: self.normalizers[name].normalize(loss) for name, loss in selected.items()
        }
        return ObjectiveOutput(
            total=weighted_loss(normalized, self.weights),
            losses=selected,
            metrics=_quotient_diagnostics(rollout),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 3,
            "weights": asdict(self.weights),
            "temporal_levels": self.temporal_levels,
            "lpips_frame_chunk_size": self.lpips_frame_chunk_size,
            "normalizers": {
                name: normalizer.state_dict() for name, normalizer in self.normalizers.items()
            },
            "distiller": self.distiller.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {
            "version",
            "weights",
            "temporal_levels",
            "lpips_frame_chunk_size",
            "normalizers",
            "distiller",
        }
        if set(state) != required or state["version"] != 3:
            raise ValueError("objective state has an incompatible schema")
        raw_weights = state["weights"]
        if not isinstance(raw_weights, Mapping) or dict(raw_weights) != asdict(self.weights):
            raise ValueError("objective loss weights do not match")
        if int(cast(int, state["temporal_levels"])) != self.temporal_levels:
            raise ValueError("objective temporal levels do not match")
        if int(cast(int, state["lpips_frame_chunk_size"])) != self.lpips_frame_chunk_size:
            raise ValueError("objective LPIPS frame chunk size does not match")
        raw_normalizers = state["normalizers"]
        if not isinstance(raw_normalizers, Mapping) or set(raw_normalizers) != set(
            self.normalizers
        ):
            raise ValueError("objective normalizer names do not match")
        for name, normalizer in self.normalizers.items():
            value = raw_normalizers[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"objective normalizer {name!r} must be a mapping")
            normalizer.load_state_dict(
                {str(key): float(cast(float, item)) for key, item in value.items()}
            )
        raw_distiller = state["distiller"]
        if not isinstance(raw_distiller, Mapping):
            raise ValueError("objective distiller state must be a mapping")
        self.distiller.load_state_dict(cast(Mapping[str, object], raw_distiller))

    def rank_state_dict(self) -> dict[str, object]:
        """Return only rank-local loss statistics for distributed exact resume."""

        return {
            "version": 1,
            "normalizers": {
                name: normalizer.state_dict() for name, normalizer in self.normalizers.items()
            },
        }

    def load_rank_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore rank-local loss statistics after loading shared objective state."""

        if set(state) != {"version", "normalizers"} or state["version"] != 1:
            raise ValueError("objective rank state has an incompatible schema")
        raw_normalizers = state["normalizers"]
        if not isinstance(raw_normalizers, Mapping) or set(raw_normalizers) != set(
            self.normalizers
        ):
            raise ValueError("objective rank normalizer names do not match")
        for name, normalizer in self.normalizers.items():
            value = raw_normalizers[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"objective rank normalizer {name!r} must be a mapping")
            normalizer.load_state_dict(
                {str(key): float(cast(float, item)) for key, item in value.items()}
            )

    def update_teacher(self, model: nn.Module) -> None:
        if not isinstance(model, TARDISModel):
            raise TypeError("TARDISObjective teacher update requires a TARDISModel")
        self.distiller.update(model.residual_dit)

    def _candidate_losses(
        self,
        model: TARDISModel,
        batch: TARDISTrainingBatch,
        rollout: TARDISTrainOutput,
        *,
        active: set[str],
    ) -> dict[str, torch.Tensor]:
        transitions = rollout.transitions
        if not transitions:
            raise ValueError("TARDIS objective requires at least one transition")
        targets = rollout.target_latents[:, 1:]
        diffusion_terms: list[torch.Tensor] = []
        residual_terms: list[torch.Tensor] = []
        transport_terms: list[torch.Tensor] = []
        flow_terms: list[torch.Tensor] = []
        visibility_terms: list[torch.Tensor] = []
        router_terms: list[torch.Tensor] = []
        survival_terms: list[torch.Tensor] = []
        lite_terms: list[torch.Tensor] = []
        budget_terms: list[torch.Tensor] = []
        warp_terms: list[torch.Tensor] = []
        crcd_terms: list[torch.Tensor] = []

        if "keyframe" in active:
            if rollout.keyframe_residual_base is None or rollout.keyframe_residual is None:
                raise RuntimeError(
                    "keyframe residual generation must expose its corrected base and prediction"
                )
            keyframe_target = (
                rollout.target_latents[:, 0] - rollout.keyframe_residual_base.detach()
            )
            keyframe_loss = residual_reconstruction_loss(
                rollout.keyframe_residual,
                keyframe_target,
            )

        if (
            "lite" in active
            and rollout.keyframe_prior is not None
            and rollout.keyframe_lite_residual is not None
        ):
            keyframe_target = rollout.target_latents[:, 0] - rollout.keyframe_prior
            lite_terms.append(
                residual_reconstruction_loss(
                    rollout.keyframe_lite_residual,
                    keyframe_target,
                )
            )

        for index, transition in enumerate(transitions):
            residual_target = transition.residual_target
            oracle = transition.oracle_probability
            if residual_target is None or oracle is None:
                raise RuntimeError("training transitions must expose residual and oracle targets")
            selection_mask = functional.interpolate(
                transition.router.selection.active_mask.to(residual_target.dtype),
                size=residual_target.shape[-2:],
                mode="nearest",
            )
            diffusion_terms.append(
                diffusion_loss(transition.sparse_residual, residual_target, selection_mask)
            )
            if "residual" in active:
                residual_terms.append(
                    residual_reconstruction_loss(
                        transition.sparse_residual,
                        residual_target,
                        selection_mask,
                    )
                )
            transport_terms.append(
                transport_loss(
                    transition.transport.prior,
                    targets[:, index],
                    rollout.motion_targets.visibility[:, index] * (1 - oracle.detach()),
                )
            )
            flow_terms.append(
                flow_loss(
                    transition.motion.backward_flow,
                    rollout.motion_targets.backward_flow[:, index].to(
                        transition.motion.backward_flow.dtype
                    ),
                    rollout.motion_targets.visibility[:, index],
                )
            )
            visibility_terms.append(
                visibility_loss(
                    transition.motion.visibility_logits,
                    rollout.motion_targets.visibility[:, index].to(
                        transition.motion.visibility_logits.dtype
                    ),
                )
            )
            if "router" in active:
                router_terms.append(router_loss(transition.router.pixel_probability, oracle))
            if "survival" in active:
                if transition.oracle_clock is None:
                    raise RuntimeError("survival calibration requires an oracle innovation clock")
                survival_terms.append(
                    survival_calibration_loss(
                        transition.predicted_clock.event_probability,
                        transition.oracle_clock.event_probability,
                    )
                )
            if "lite" in active:
                if transition.tangent_target is not None:
                    lite_terms.append(
                        residual_reconstruction_loss(
                            transition.lite_residual,
                            transition.tangent_target,
                        )
                    )
                else:
                    full_residual = targets[:, index] - transition.transport.prior
                    lite_terms.append(
                        lite_residual_loss(transition.lite_residual, full_residual, oracle)
                    )
            if "budget" in active:
                budget_terms.append(
                    budget_loss(
                        transition.predicted_clock.patch_probability,
                        model.router.active_ratio,
                    )
                )
            if "warp" in active:
                warp_terms.append(
                    warp_loss(
                        transition.latent,
                        transition.transport.warped_latent,
                        transition.transport.effective_visibility,
                    )
                )
            if "crcd" in active:
                crcd_terms.append(
                    crcd_loss(
                        transition.sparse_residual,
                        self.distiller.build_target(
                            model.residual_dit,
                            transition.residual_context,
                            quotient_projector=(
                                model.quotient if transition.quotient_basis is not None else None
                            ),
                            quotient_basis=transition.quotient_basis,
                        ),
                    )
                )

        losses = {
            "diffusion": _mean(diffusion_terms),
            "transport": _mean(transport_terms),
            "flow": _mean(flow_terms),
            "visibility": _mean(visibility_terms),
        }
        if "keyframe" in active:
            losses["keyframe"] = keyframe_loss
        transition_terms = {
            "residual": residual_terms,
            "router": router_terms,
            "survival": survival_terms,
            "lite": lite_terms,
            "budget": budget_terms,
            "warp": warp_terms,
            "crcd": crcd_terms,
        }
        for name, terms in transition_terms.items():
            if name in active:
                losses[name] = _mean(terms)
        if "drift" in active:
            losses["drift"] = drift_loss(rollout.predicted_latents)
        if active & {"lpips", "tc"}:
            generated_video = _decode_video_for_metric(model.priors, rollout.predicted_latents)
            if "lpips" in active:
                losses["lpips"] = lpips_loss(
                    self.perceptual_metric,
                    generated_video.flatten(0, 1),
                    batch.video.flatten(0, 1),
                    frame_chunk_size=self.lpips_frame_chunk_size,
                )
            if "tc" in active:
                losses["tc"] = multi_scale_temporal_consistency_loss(
                    generated_video,
                    batch.video,
                    levels=self.temporal_levels,
                )
        if "text" in active:
            losses["text"] = _shared_space_text_loss(
                model,
                rollout,
                batch.prompts,
            )
        return losses


class TARDISKeyframeObjective:
    """Optimize the exact deployed keyframe trajectory on one decoded frame."""

    def __init__(
        self,
        *,
        perceptual_metric: PerceptualMetric,
        keyframe_weight: float = 1.0,
        lpips_weight: float = 1.0,
        normalizer_decay: float = 0.99,
        normalizer_epsilon: float = 1.0e-6,
        lpips_frame_chunk_size: int = 4,
    ) -> None:
        if keyframe_weight < 0 or lpips_weight < 0 or keyframe_weight + lpips_weight <= 0:
            raise ValueError("keyframe objective requires non-negative, non-zero loss weights")
        if lpips_frame_chunk_size <= 0:
            raise ValueError("LPIPS frame chunk size must be positive")
        self.perceptual_metric = perceptual_metric
        self.keyframe_weight = keyframe_weight
        self.lpips_weight = lpips_weight
        self.lpips_frame_chunk_size = lpips_frame_chunk_size
        self.normalizers = {
            name: EmaLossNormalizer(decay=normalizer_decay, epsilon=normalizer_epsilon)
            for name in ("keyframe", "lpips")
        }

    def __call__(
        self,
        model: nn.Module,
        batch: object,
        point: CurriculumPoint,
        generator: torch.Generator,
    ) -> ObjectiveOutput:
        del point
        if not isinstance(batch, TARDISTrainingBatch):
            raise TypeError("TARDISKeyframeObjective requires a TARDISTrainingBatch")
        output = model(
            batch,
            "keyframe_only",
            teacher_forcing_ratio=0.0,
            generator=generator,
            train_mode="keyframe_only",
        )
        if not isinstance(output, TARDISKeyframeTrainOutput):
            raise RuntimeError("keyframe training forward returned an incompatible output")
        latent_loss = residual_reconstruction_loss(
            output.predicted_latent,
            output.target_latent,
        )
        tardis_model = _unwrap_tardis_model(model)
        generated = _decode_video_for_metric(
            tardis_model.priors,
            output.predicted_latent[:, None],
        )
        perceptual_loss = lpips_loss(
            self.perceptual_metric,
            generated[:, 0],
            batch.video[:, 0],
            frame_chunk_size=self.lpips_frame_chunk_size,
        )
        normalized_latent = self.normalizers["keyframe"].normalize(latent_loss)
        normalized_lpips = self.normalizers["lpips"].normalize(perceptual_loss)
        return ObjectiveOutput(
            total=(
                self.keyframe_weight * normalized_latent
                + self.lpips_weight * normalized_lpips
            ),
            losses={"keyframe": latent_loss, "lpips": perceptual_loss},
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "keyframe_weight": self.keyframe_weight,
            "lpips_weight": self.lpips_weight,
            "lpips_frame_chunk_size": self.lpips_frame_chunk_size,
            "normalizers": {
                name: normalizer.state_dict() for name, normalizer in self.normalizers.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {
            "version",
            "keyframe_weight",
            "lpips_weight",
            "lpips_frame_chunk_size",
            "normalizers",
        }
        if set(state) != required or state["version"] != 1:
            raise ValueError("keyframe objective state has an incompatible schema")
        if (
            float(cast(float, state["keyframe_weight"])) != self.keyframe_weight
            or float(cast(float, state["lpips_weight"])) != self.lpips_weight
            or int(cast(int, state["lpips_frame_chunk_size"]))
            != self.lpips_frame_chunk_size
        ):
            raise ValueError("keyframe objective settings do not match")
        raw_normalizers = state["normalizers"]
        if not isinstance(raw_normalizers, Mapping) or set(raw_normalizers) != set(
            self.normalizers
        ):
            raise ValueError("keyframe objective normalizers do not match")
        for name, normalizer in self.normalizers.items():
            raw = raw_normalizers[name]
            if not isinstance(raw, Mapping):
                raise ValueError(f"keyframe normalizer {name!r} must be a mapping")
            normalizer.load_state_dict(
                {str(key): float(cast(float, value)) for key, value in raw.items()}
            )

    def rank_state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "normalizers": {
                name: normalizer.state_dict() for name, normalizer in self.normalizers.items()
            },
        }

    def load_rank_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"version", "normalizers"} or state["version"] != 1:
            raise ValueError("keyframe objective rank state has an incompatible schema")
        raw_normalizers = state["normalizers"]
        if not isinstance(raw_normalizers, Mapping) or set(raw_normalizers) != set(
            self.normalizers
        ):
            raise ValueError("keyframe objective rank normalizers do not match")
        for name, normalizer in self.normalizers.items():
            raw = raw_normalizers[name]
            if not isinstance(raw, Mapping):
                raise ValueError(f"keyframe rank normalizer {name!r} must be a mapping")
            normalizer.load_state_dict(
                {str(key): float(cast(float, value)) for key, value in raw.items()}
            )


def _shared_space_text_loss(
    model: TARDISModel,
    rollout: TARDISTrainOutput,
    prompts: list[str],
) -> torch.Tensor:
    text, mask = model.priors.encode_text(prompts)
    text_tokens = model.residual_dit.text_projection(text)
    weights = mask.unsqueeze(-1).to(text_tokens.dtype)
    pooled_text = (text_tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
    latent_patches, _ = patchify(
        rollout.predicted_latents[:, -1],
        patch_size=model.residual_dit.patch_size,
    )
    latent_tokens = model.residual_dit.input_projection(
        torch.cat((latent_patches, latent_patches), dim=-1)
    )
    return text_alignment_loss(latent_tokens.mean(1), pooled_text)


def _decode_video_for_metric(
    priors: _VideoDecoder,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Use the same bounded pixel range as the deployed generation path."""

    return priors.decode_video(latents).clamp(-1, 1)


def _quotient_diagnostics(rollout: TARDISTrainOutput) -> dict[str, torch.Tensor]:
    raw_energy: list[torch.Tensor] = []
    tar_energy: list[torch.Tensor] = []
    quotient_energy: list[torch.Tensor] = []
    tangent_energy: list[torch.Tensor] = []
    tangent_rank: list[torch.Tensor] = []
    irf_mass: list[torch.Tensor] = []
    proper_time_mass: list[torch.Tensor] = []
    unsettled_hazard: list[torch.Tensor] = []
    active_ratio: list[torch.Tensor] = []
    for index, transition in enumerate(rollout.transitions):
        current = rollout.target_latents[:, index + 1].detach()
        previous = rollout.target_latents[:, index].detach()
        raw = (current - previous).abs().mean()
        tar = (current - transition.transport.prior.detach()).abs().mean()
        quotient = (
            tar.new_zeros(())
            if transition.residual_target is None
            else transition.residual_target.detach().abs().mean()
        )
        tangent = (
            tar.new_zeros(())
            if transition.tangent_target is None
            else transition.tangent_target.detach().abs().mean()
        )
        token_count = transition.router.selection.active_mask[0].numel()
        raw_energy.append(raw)
        tar_energy.append(tar)
        quotient_energy.append(quotient)
        tangent_energy.append(tangent)
        tangent_rank.append(transition.quotient_tangent_rank.detach().float().mean())
        irf_mass.append(transition.router.pixel_probability.detach().float().mean())
        proper_time_mass.append(
            transition.predicted_clock.event_probability.detach().float().mean()
        )
        unsettled_hazard.append(transition.predicted_clock.settled_hazard.detach().float().mean())
        active_ratio.append(
            transition.router.selection.active_counts.detach().float().mean() / token_count
        )
    raw_mean = _mean(raw_energy)
    tar_mean = _mean(tar_energy)
    quotient_mean = _mean(quotient_energy)
    tangent_mean = _mean(tangent_energy)
    epsilon = torch.finfo(raw_mean.dtype).eps
    return {
        "raw_residual_l1": raw_mean,
        "tar_residual_l1": tar_mean,
        "quotient_residual_l1": quotient_mean,
        "tar_to_raw_ratio": tar_mean / raw_mean.clamp_min(epsilon),
        "quotient_to_tar_ratio": quotient_mean / tar_mean.clamp_min(epsilon),
        "tangent_explained_ratio": tangent_mean / tar_mean.clamp_min(epsilon),
        "mean_tangent_rank": _mean(tangent_rank),
        "irf_mass": _mean(irf_mass),
        "proper_time_mass": _mean(proper_time_mass),
        "unsettled_hazard": _mean(unsettled_hazard),
        "active_ratio": _mean(active_ratio),
    }


def _mean(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot average an empty loss sequence")
    return torch.stack(values).mean()


def _unwrap_tardis_model(model: nn.Module) -> TARDISModel:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TARDISModel):
            return current
        wrapped = getattr(current, "module", None)
        if isinstance(wrapped, nn.Module):
            current = wrapped
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module):
            current = original
            continue
        break
    raise TypeError("TARDISObjective requires a TARDISModel or its distributed wrapper")
