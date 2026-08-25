from __future__ import annotations

import inspect
import math
from dataclasses import replace

import torch

from tardis.models.tardis import (
    AblationVariant,
    TARDISAblationFlags,
    TARDISModel,
    TARDISTrainingBatch,
    TransitionConditions,
)
from tests.helpers.tardis_model import build_tiny_tardis


def transition_conditions(
    model: TARDISModel,
    *,
    target: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
    scene_cut: bool = False,
    diffusion_time: torch.Tensor | None = None,
    diffusion_noise: torch.Tensor | None = None,
) -> TransitionConditions:
    text, mask = model.priors.encode_text(["moving light"])
    latent_height = model.config.height // model.priors.spatial_scale
    latent_width = model.config.width // model.priors.spatial_scale
    return TransitionConditions(
        text_embeddings=text,
        text_mask=mask,
        time=torch.tensor([0.5]),
        motion_noise=torch.zeros(
            1,
            model.config.motion_noise_channels,
            latent_height,
            latent_width,
        ),
        diffusion_noise=(
            diffusion_noise
            if diffusion_noise is not None
            else torch.zeros(
                1,
                model.priors.latent_channels,
                latent_height,
                latent_width,
            )
        ),
        diffusion_time=diffusion_time,
        target_latent=target,
        teacher_backward_flow=torch.zeros(1, 2, latent_height, latent_width),
        teacher_visibility=(
            visibility if visibility is not None else torch.ones(1, 1, latent_height, latent_width)
        ),
        scene_cut_mask=torch.tensor([scene_cut]),
        use_oracle_routing=target is not None,
        detach_state=False,
    )


def test_generate_is_strictly_prompt_only_and_uses_semantic_first_frame() -> None:
    assembly = build_tiny_tardis()
    model = assembly.model.eval()
    parameters = inspect.signature(TARDISModel.generate).parameters

    output = model.generate(
        ["a luminous train"],
        num_frames=3,
        fps=8,
        generator=torch.Generator().manual_seed(11),
    )
    text, mask = model.priors.encode_text(["a luminous train"])
    expected_first = model.priors.generate_first_latent(
        text,
        mask,
        generator=torch.Generator().manual_seed(11),
        height=16,
        width=16,
    )

    assert list(parameters) == ["self", "prompts", "num_frames", "fps", "generator"]
    assert not {"video", "source_video", "reference_video"}.intersection(parameters)
    assert output.latents.shape == (1, 3, 4, 8, 8)
    assert output.video.shape == (1, 3, 3, 16, 16)
    assert torch.equal(output.latents[:, 0], expected_first)
    assert assembly.motion_teacher.calls == 0


def test_source_conditioned_generation_binds_innovation_to_source_latents() -> None:
    assembly = build_tiny_tardis()
    model = assembly.model.eval()
    source = torch.randn(1, 3, 3, 16, 16).clamp(-1, 1)
    source_latents = model.priors.encode_video(source)

    identity = model.generate_source_conditioned(
        ["a luminous train"],
        source,
        fps=8,
        generator=torch.Generator().manual_seed(11),
        innovation_strength=0.0,
    )
    quarter = model.generate_source_conditioned(
        ["a luminous train"],
        source,
        fps=8,
        generator=torch.Generator().manual_seed(11),
        innovation_strength=0.25,
    )
    full = model.generate_source_conditioned(
        ["a luminous train"],
        source,
        fps=8,
        generator=torch.Generator().manual_seed(11),
        innovation_strength=1.0,
    )

    assert identity.latents.shape == source_latents.shape
    assert identity.video.shape == source.shape
    assert torch.equal(identity.latents, source_latents)
    assert torch.allclose(
        quarter.latents,
        torch.lerp(source_latents, full.latents, 0.25),
    )
    assert assembly.motion_teacher.calls == 3


def test_source_conditioned_generation_validates_source_contract() -> None:
    model = build_tiny_tardis().model.eval()
    source = torch.zeros(1, 3, 3, 16, 16)

    try:
        model.generate_source_conditioned(
            ["prompt"],
            source,
            fps=8,
            generator=torch.Generator().manual_seed(1),
            innovation_strength=-0.1,
        )
    except ValueError as error:
        assert "innovation_strength" in str(error)
    else:
        raise AssertionError("negative innovation strength must fail")

    try:
        model.generate_source_conditioned(
            ["one", "two"],
            source,
            fps=8,
            generator=torch.Generator().manual_seed(1),
        )
    except ValueError as error:
        assert "prompts" in str(error)
    else:
        raise AssertionError("prompt/source batch mismatch must fail")


def test_keyframe_lite_alignment_corrects_prompt_prior_before_causal_rollout() -> None:
    model = build_tiny_tardis(keyframe_lite_alignment=True).model.eval()
    with torch.no_grad():
        model.lite_corrector.output_projection.bias.fill_(0.5)
    text, mask = model.priors.encode_text(["a luminous train"])
    prior = model.priors.generate_first_latent(
        text,
        mask,
        generator=torch.Generator().manual_seed(11),
        height=16,
        width=16,
    )

    output = model.generate(
        ["a luminous train"],
        num_frames=1,
        fps=8,
        generator=torch.Generator().manual_seed(11),
    )

    assert not torch.equal(output.latents[:, 0], prior)
    assert torch.allclose(
        output.latents[:, 0] - prior,
        torch.full_like(prior, torch.tanh(torch.tensor(0.5)).item() * 0.1),
    )


def test_keyframe_lite_alignment_passes_prompt_condition_to_corrector() -> None:
    model = build_tiny_tardis(keyframe_lite_alignment=True).model.eval()
    calls: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def capture(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor | None],
    ) -> None:
        del inputs
        calls.append((kwargs.get("text_embeddings"), kwargs.get("text_mask")))

    handle = model.lite_corrector.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model.generate(
            ["a luminous train"],
            num_frames=1,
            fps=8,
            generator=torch.Generator().manual_seed(11),
        )
    finally:
        handle.remove()

    assert len(calls) == 1
    assert calls[0][0] is not None
    assert calls[0][1] is not None


def test_keyframe_alignment_exposes_supervised_lite_residual_during_training() -> None:
    model = build_tiny_tardis(keyframe_lite_alignment=True).model.train()
    batch = TARDISTrainingBatch(
        prompts=["a moving prism"],
        video=torch.randn(1, 3, 3, 16, 16),
    )

    output = model.forward_train(
        batch,
        stage="transport_warmup",
        teacher_forcing_ratio=1.0,
        generator=torch.Generator().manual_seed(31),
    )

    assert output.keyframe_prior is not None
    assert output.keyframe_lite_residual is not None
    assert output.keyframe_prior.shape == output.target_latents[:, 0].shape
    assert output.keyframe_lite_residual.shape == output.target_latents[:, 0].shape


def test_keyframe_only_training_skips_motion_and_temporal_rollout() -> None:
    assembly = build_tiny_tardis(
        keyframe_lite_alignment=True,
        keyframe_residual_generation=True,
        diffusion_steps=2,
        diffusion_time_sampling="endpoint",
        sampler_trajectory_alignment=True,
    )
    model = assembly.model.train()
    batch = TARDISTrainingBatch(
        prompts=["a moving prism"],
        video=torch.randn(1, 1, 3, 16, 16),
    )

    output = model(
        batch,
        stage="keyframe_only",
        train_mode="keyframe_only",
        generator=torch.Generator().manual_seed(31),
    )

    assert output.predicted_latent.shape == output.target_latent.shape
    assert output.keyframe_prior.shape == output.target_latent.shape
    assert output.keyframe_lite_residual.shape == output.target_latent.shape
    assert output.keyframe_residual.shape == output.target_latent.shape
    assert assembly.motion_teacher.calls == 0


def test_keyframe_lite_alignment_does_not_reapply_lite_updates_on_transitions() -> None:
    model = build_tiny_tardis(keyframe_lite_alignment=True).model.eval()
    calls: list[torch.Tensor] = []
    handle = model.lite_corrector.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output)
    )

    try:
        output = model.generate(
            ["a luminous train"],
            num_frames=3,
            fps=8,
            generator=torch.Generator().manual_seed(11),
        )
    finally:
        handle.remove()

    assert len(calls) == 1
    assert len(output.transitions) == 2
    assert all(
        torch.count_nonzero(transition.lite_residual).item() == 0
        for transition in output.transitions
    )


def test_keyframe_residual_generation_is_unbounded_by_lite_corrector() -> None:
    model = build_tiny_tardis(keyframe_residual_generation=True).model.eval()
    with torch.no_grad():
        model.keyframe_residual_dit.output_projection.bias.fill_(1.5)
    text, mask = model.priors.encode_text(["a luminous train"])
    prior = model.priors.generate_first_latent(
        text,
        mask,
        generator=torch.Generator().manual_seed(11),
        height=16,
        width=16,
    )

    output = model.generate(
        ["a luminous train"],
        num_frames=1,
        fps=8,
        generator=torch.Generator().manual_seed(11),
    )

    assert (output.latents[:, 0] - prior).abs().max() > model.lite_corrector.max_magnitude


def test_keyframe_generation_keeps_independent_transition_lite_updates() -> None:
    model = build_tiny_tardis(
        keyframe_lite_alignment=True,
        keyframe_residual_generation=True,
    ).model.eval()
    with torch.no_grad():
        model.transition_lite_corrector.output_projection.bias.fill_(0.5)
    model.config = replace(model.config, transport_quotient=False)

    output = model.generate(
        ["a luminous train"],
        num_frames=3,
        fps=8,
        generator=torch.Generator().manual_seed(11),
    )

    assert all(torch.count_nonzero(item.lite_residual) > 0 for item in output.transitions)


def test_diffusion_steps_drive_keyframe_and_transition_denoising_calls() -> None:
    model = build_tiny_tardis(
        keyframe_residual_generation=True,
        diffusion_steps=3,
    ).model.eval()
    keyframe_calls = 0
    transition_calls = 0

    def count_keyframe(*_args: object, **_kwargs: object) -> None:
        nonlocal keyframe_calls
        keyframe_calls += 1

    def count_transition(*_args: object, **_kwargs: object) -> None:
        nonlocal transition_calls
        transition_calls += 1

    keyframe_handle = model.keyframe_residual_dit.register_forward_pre_hook(count_keyframe)
    transition_handle = model.residual_dit.register_forward_pre_hook(count_transition)
    try:
        model.generate(
            ["a luminous train"],
            num_frames=2,
            fps=8,
            generator=torch.Generator().manual_seed(11),
        )
    finally:
        keyframe_handle.remove()
        transition_handle.remove()

    assert keyframe_calls == 3
    assert transition_calls == 3


def test_endpoint_training_matches_deployment_noise_endpoint_for_both_denoisers() -> None:
    model = build_tiny_tardis(
        keyframe_residual_generation=True,
        diffusion_time_sampling="endpoint",
    ).model.train()
    keyframe_times: list[torch.Tensor] = []
    transition_times: list[torch.Tensor] = []

    def capture_keyframe(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        keyframe_times.append(kwargs["diffusion_time"].detach().clone())

    def capture_transition(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        transition_times.append(kwargs["diffusion_time"].detach().clone())

    handles = (
        model.keyframe_residual_dit.register_forward_pre_hook(
            capture_keyframe,
            with_kwargs=True,
        ),
        model.residual_dit.register_forward_pre_hook(
            capture_transition,
            with_kwargs=True,
        ),
    )
    try:
        model.forward_train(
            TARDISTrainingBatch(
                prompts=["a moving prism", "a rotating cube"],
                video=torch.randn(2, 3, 3, 16, 16),
            ),
            stage="transport_warmup",
            generator=torch.Generator().manual_seed(73),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert len(keyframe_times) == 1
    assert len(transition_times) == 2
    assert torch.equal(keyframe_times[0], torch.ones_like(keyframe_times[0]))
    assert all(torch.equal(value, torch.ones_like(value)) for value in transition_times)


def test_sampler_trajectory_alignment_runs_deployment_steps_during_training() -> None:
    model = build_tiny_tardis(
        keyframe_residual_generation=True,
        diffusion_steps=2,
        diffusion_time_sampling="endpoint",
        sampler_trajectory_alignment=True,
    ).model.train()
    keyframe_times: list[torch.Tensor] = []
    transition_times: list[torch.Tensor] = []

    def capture_keyframe(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        keyframe_times.append(kwargs["diffusion_time"].detach().clone())

    def capture_transition(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        transition_times.append(kwargs["diffusion_time"].detach().clone())

    handles = (
        model.keyframe_residual_dit.register_forward_pre_hook(
            capture_keyframe,
            with_kwargs=True,
        ),
        model.residual_dit.register_forward_pre_hook(
            capture_transition,
            with_kwargs=True,
        ),
    )
    try:
        model.forward_train(
            TARDISTrainingBatch(
                prompts=["a moving prism"],
                video=torch.randn(1, 2, 3, 16, 16),
            ),
            stage="metric_alignment",
            teacher_forcing_ratio=0.0,
            generator=torch.Generator().manual_seed(89),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert [float(value.item()) for value in keyframe_times] == [1.0, 0.0]
    assert [float(value.item()) for value in transition_times] == [1.0, 0.0]


def test_sampler_trajectory_alignment_backpropagates_through_all_deployment_steps() -> None:
    model = build_tiny_tardis(
        keyframe_residual_generation=True,
        diffusion_steps=2,
        diffusion_time_sampling="endpoint",
        sampler_trajectory_alignment=True,
    ).model.train()
    observed: list[torch.Tensor] = []

    def capture_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: object,
    ) -> None:
        residual = output.residual  # type: ignore[attr-defined]
        residual.retain_grad()
        observed.append(residual)

    handle = model.residual_dit.register_forward_hook(capture_output)
    try:
        rollout = model.forward_train(
            TARDISTrainingBatch(
                prompts=["a moving prism"],
                video=torch.randn(1, 2, 3, 16, 16),
            ),
            stage="metric_alignment",
            teacher_forcing_ratio=0.0,
            generator=torch.Generator().manual_seed(97),
        )
        rollout.predicted_latents.square().mean().backward()
    finally:
        handle.remove()

    assert len(observed) == 2
    assert all(value.grad is not None for value in observed)
    assert all(
        torch.count_nonzero(value.grad).item() > 0
        for value in observed
        if value.grad is not None
    )


def test_uniform_training_keeps_stochastic_diffusion_times() -> None:
    model = build_tiny_tardis(
        keyframe_residual_generation=True,
        diffusion_time_sampling="uniform",
    ).model.train()
    observed: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        observed.append(kwargs["diffusion_time"].detach().clone())

    handles = (
        model.keyframe_residual_dit.register_forward_pre_hook(capture, with_kwargs=True),
        model.residual_dit.register_forward_pre_hook(capture, with_kwargs=True),
    )
    try:
        model.forward_train(
            TARDISTrainingBatch(
                prompts=["a moving prism", "a rotating cube"],
                video=torch.randn(2, 3, 3, 16, 16),
            ),
            stage="transport_warmup",
            generator=torch.Generator().manual_seed(79),
        )
    finally:
        for handle in handles:
            handle.remove()

    times = torch.cat(observed)
    assert bool(((times > 0) & (times < 1)).all())
    assert torch.unique(times).numel() > 1


def test_high_noise_training_samples_only_deployment_near_times() -> None:
    model = build_tiny_tardis(
        keyframe_residual_generation=True,
        diffusion_time_sampling="high_noise",
    ).model.train()
    observed: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        observed.append(kwargs["diffusion_time"].detach().clone())

    handles = (
        model.keyframe_residual_dit.register_forward_pre_hook(capture, with_kwargs=True),
        model.residual_dit.register_forward_pre_hook(capture, with_kwargs=True),
    )
    try:
        model.forward_train(
            TARDISTrainingBatch(
                prompts=["a moving prism", "a rotating cube"],
                video=torch.randn(2, 3, 3, 16, 16),
            ),
            stage="transport_warmup",
            generator=torch.Generator().manual_seed(83),
        )
    finally:
        for handle in handles:
            handle.remove()

    times = torch.cat(observed)
    assert bool(((times >= 0.5) & (times <= 1)).all())
    assert torch.unique(times).numel() > 1


def test_generation_is_deterministic_for_the_same_generator_seed() -> None:
    model = build_tiny_tardis().model.eval()

    first = model.generate(
        ["one prompt"],
        num_frames=4,
        fps=8,
        generator=torch.Generator().manual_seed(29),
    )
    second = model.generate(
        ["one prompt"],
        num_frames=4,
        fps=8,
        generator=torch.Generator().manual_seed(29),
    )

    assert torch.equal(first.latents, second.latents)
    assert torch.equal(first.video, second.video)


def test_generation_clamps_public_video_to_metric_range() -> None:
    model = build_tiny_tardis().model.eval()
    decoder = model.priors.codec_module.decoder  # type: ignore[attr-defined]
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.fill_(3.0)

    output = model.generate(
        ["one bright frame"],
        num_frames=1,
        fps=8,
        generator=torch.Generator().manual_seed(13),
    )

    assert output.video.min() >= -1
    assert output.video.max() <= 1


def test_transition_is_exact_transport_plus_lite_plus_sparse_residual() -> None:
    model = build_tiny_tardis().model
    previous_latent = torch.randn(1, 4, 8, 8)
    previous = model.state_updater.initialize(previous_latent, detach=False)

    output = model.transition(previous, transition_conditions(model))

    expected = output.transport.prior + output.lite_residual + output.sparse_residual
    assert torch.allclose(output.latent, expected)
    assert torch.equal(output.state.latent, output.latent)
    assert output.router.selection.active_counts.max() <= math.ceil(0.25 * 16)
    assert output.clock.event_probability.shape == (1, 1, 8, 8)
    assert output.state.innovation_hazard.shape == (1, 1, 8, 8)


def test_transition_blends_teacher_motion_into_the_deployable_scaffold() -> None:
    model = build_tiny_tardis().model.eval()
    previous_latent = torch.randn(1, 4, 8, 8)
    previous = model.state_updater.initialize(previous_latent, detach=False)
    teacher_flow = torch.ones(1, 2, 8, 8)

    output = model.transition(
        previous,
        replace(
            transition_conditions(model),
            teacher_backward_flow=teacher_flow,
            teacher_visibility=torch.ones(1, 1, 8, 8),
            teacher_motion_ratio=0.5,
        ),
    )

    assert torch.allclose(output.transport.corrected_flow, teacher_flow * 0.5)


def test_transition_accepts_float_teacher_motion_under_bfloat16_autocast() -> None:
    model = build_tiny_tardis().model.eval()
    previous_latent = torch.randn(1, 4, 8, 8)
    previous = model.state_updater.initialize(previous_latent, detach=False)
    conditions = replace(
        transition_conditions(model),
        teacher_backward_flow=torch.ones(1, 2, 8, 8, dtype=torch.float32),
        teacher_visibility=torch.ones(1, 1, 8, 8, dtype=torch.float32),
        teacher_motion_ratio=0.5,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model.transition(previous, conditions)

    assert output.transport.corrected_flow.dtype == output.motion.backward_flow.dtype


def test_training_target_is_split_into_transport_tangent_and_quotient_innovation() -> None:
    model = build_tiny_tardis().model
    previous_latent = torch.randn(1, 4, 8, 8)
    target = torch.randn_like(previous_latent)
    previous = model.state_updater.initialize(previous_latent, detach=False)

    output = model.transition(previous, transition_conditions(model, target=target))

    assert output.tangent_target is not None
    assert output.residual_target is not None
    expected = target - output.transport.prior.detach()
    assert torch.allclose(
        output.tangent_target + output.residual_target,
        expected,
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_oracle_routing_skips_reusable_regions_and_caps_full_failure() -> None:
    model = build_tiny_tardis(active_ratio=0.25).model
    previous_latent = torch.randn(1, 4, 8, 8)
    previous = model.state_updater.initialize(previous_latent, detach=False)

    reusable = model.transition(
        previous,
        transition_conditions(model, target=previous_latent),
    )
    failure = model.transition(
        previous,
        transition_conditions(
            model,
            target=torch.ones_like(previous_latent),
            visibility=torch.zeros(1, 1, 8, 8),
        ),
    )

    assert reusable.oracle_probability is not None
    assert torch.count_nonzero(reusable.oracle_probability) == 0
    assert reusable.router.selection.active_counts.tolist() == [0]
    assert failure.oracle_probability is not None
    assert torch.allclose(failure.oracle_probability, torch.ones_like(failure.oracle_probability))
    assert failure.router.selection.active_counts.tolist() == [4]


def test_scene_cut_resets_each_selected_causal_state() -> None:
    model = build_tiny_tardis().model
    latent = torch.randn(1, 4, 8, 8)
    state = model.state_updater.initialize(latent, detach=False)
    state = model.state_updater.update(
        state,
        latent,
        innovation_probability=torch.zeros(1, 1, 8, 8),
        reset_mask=torch.zeros(1, dtype=torch.bool),
        detach=False,
    )

    output = model.transition(state, transition_conditions(model, scene_cut=True))

    assert state.frame_index.tolist() == [1]
    assert output.scene_cut_mask.tolist() == [True]
    assert output.state.frame_index.tolist() == [0]


def test_a0_to_a10_add_exactly_one_claimed_mechanism_at_each_step() -> None:
    feature_names = (
        "previous_frame_conditioning",
        "temporal_residual",
        "source_motion_transport",
        "analytical_visibility",
        "learned_vcir",
        "dual_frequency_residual",
        "fixed_budget_routing",
        "innovation_proper_time",
        "crcd",
        "metric_alignment",
    )

    assert [variant.value for variant in AblationVariant] == [f"A{index}" for index in range(11)]
    for index, variant in enumerate(AblationVariant):
        flags = TARDISAblationFlags.for_variant(variant)
        enabled = [name for name in feature_names if getattr(flags, name)]
        assert len(enabled) == index
        assert enabled == list(feature_names[:index])


def test_closed_loop_training_reuses_predicted_causal_state_without_ground_truth_reset() -> None:
    model = build_tiny_tardis().model.train()
    batch = TARDISTrainingBatch(
        prompts=["a moving prism"],
        video=torch.randn(1, 4, 3, 16, 16),
    )

    output = model.forward_train(
        batch,
        stage="closed_loop",
        teacher_forcing_ratio=0.0,
        generator=torch.Generator().manual_seed(31),
    )

    assert [transition.state.frame_index.item() for transition in output.transitions] == [1, 2, 3]
    assert output.teacher_forcing_mask.shape == (1, 3)
    assert not bool(output.teacher_forcing_mask.any())


def test_closed_loop_training_starts_from_the_prompt_prior_used_by_generation() -> None:
    model = build_tiny_tardis().model.train()
    batch = TARDISTrainingBatch(
        prompts=["a moving prism"],
        video=torch.randn(1, 4, 3, 16, 16),
    )
    text, mask = model.priors.encode_text(batch.prompts)
    expected = model.priors.generate_first_latent(
        text,
        mask,
        generator=torch.Generator().manual_seed(31),
        height=16,
        width=16,
    )

    output = model.forward_train(
        batch,
        stage="closed_loop",
        teacher_forcing_ratio=0.0,
        generator=torch.Generator().manual_seed(31),
    )

    assert torch.equal(output.predicted_latents[:, 0], expected)
    assert not torch.equal(output.predicted_latents[:, 0], output.target_latents[:, 0])


def test_teacher_forced_training_keeps_the_next_state_anchored_to_reference() -> None:
    model = build_tiny_tardis().model.train()
    batch = TARDISTrainingBatch(
        prompts=["a moving prism"],
        video=torch.randn(1, 3, 3, 16, 16),
    )

    output = model.forward_train(
        batch,
        stage="transport_warmup",
        teacher_forcing_ratio=1.0,
        generator=torch.Generator().manual_seed(31),
    )

    assert [transition.state.frame_index.item() for transition in output.transitions] == [1, 1]
    assert bool(output.teacher_forcing_mask.all())


def test_quotient_diffusion_training_and_generation_share_the_same_noise_endpoint() -> None:
    model = build_tiny_tardis().model
    previous_latent = torch.randn(1, 4, 8, 8)
    target = torch.randn_like(previous_latent)
    noise = torch.randn_like(previous_latent)
    previous = model.state_updater.initialize(previous_latent, detach=False)

    clean_endpoint = model.transition(
        previous,
        transition_conditions(
            model,
            target=target,
            diffusion_time=torch.zeros(1),
            diffusion_noise=noise,
        ),
    )
    noise_endpoint = model.transition(
        previous,
        transition_conditions(
            model,
            target=target,
            diffusion_time=torch.ones(1),
            diffusion_noise=noise,
        ),
    )

    assert clean_endpoint.residual_target is not None
    assert torch.allclose(clean_endpoint.noisy_residual, clean_endpoint.residual_target)
    basis = model.quotient.build_basis(
        noise_endpoint.transport.prior,
        noise_endpoint.transport.effective_visibility,
    )
    expected_noise = model.quotient.decompose(noise, basis).innovation
    assert torch.allclose(noise_endpoint.noisy_residual, expected_noise)
    assert torch.equal(
        noise_endpoint.noisy_residual,
        noise_endpoint.residual_context.diffusion_noise,
    )
