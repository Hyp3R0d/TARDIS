"""Prompt-to-video adapters used by the locked paper benchmark protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch

from tardis.experiments.benchmark import normalize_video


class VideoGenerator(Protocol):
    """One prompt-only video generator under the common benchmark contract."""

    method: str
    provenance: Mapping[str, object]

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor: ...


_ARCHITECTURE_FIELDS = (
    "active_ratio",
    "diffusion_steps",
    "diffusion_time_sampling",
    "gradient_checkpointing",
    "height",
    "hidden_size",
    "innovation_proper_time",
    "keyframe_lite_alignment",
    "keyframe_residual_generation",
    "latent_channels",
    "lite_max_magnitude",
    "motion_max_flow_pixels",
    "num_frames",
    "num_heads",
    "num_layers",
    "patch_size",
    "pretrained_model",
    "proper_time_maximum_hazard",
    "quotient_rank_threshold",
    "quotient_regularization",
    "router_halo_radius",
    "router_threshold",
    "sampler_trajectory_alignment",
    "scene_cut_threshold",
    "state_anchor_decay",
    "training_noise_scale",
    "transport_history_fallback_weight",
    "transport_max_correction_pixels",
    "transport_quotient",
    "width",
)


class TARDISGenerator:
    def __init__(
        self,
        *,
        dataset: str,
        checkpoint: Path,
        train_manifest: Path,
        device: torch.device,
        precision: str,
        variant: str = "A10",
    ) -> None:
        from tardis.cli.infer import parse_args as parse_infer_args
        from tardis.cli.runtime import build_generation_runtime
        from tardis.models.tardis import AblationVariant, TARDISAblationFlags
        from tardis.utils.checkpoint import checkpoint_sha256

        manifest = json.loads(train_manifest.read_text(encoding="utf-8"))
        training_args = manifest.get("args")
        if not isinstance(training_args, dict) or training_args.get("dataset") != dataset:
            raise ValueError("TARDIS train manifest does not match the selected dataset")
        runtime_args = parse_infer_args(
            [
                "--dataset",
                dataset,
                "--checkpoint",
                str(checkpoint),
                "--device",
                str(device),
                "--precision",
                precision,
                "--num-workers",
                "0",
            ]
        )
        for name in _ARCHITECTURE_FIELDS:
            if name not in training_args:
                raise ValueError(f"TARDIS train manifest is missing architecture field {name!r}")
            setattr(runtime_args, name, training_args[name])
        runtime = build_generation_runtime(
            runtime_args,
            checkpoint=checkpoint,
            use_ema=True,
            local_files_only=True,
        )
        model = runtime.model.eval()
        selected_variant = AblationVariant(variant)
        model.ablation = selected_variant
        model.ablation_flags = TARDISAblationFlags.for_variant(selected_variant)
        self.method = "tardis" if variant == "A10" else f"tardis_{variant.lower()}"
        self._model = model
        self._device = device
        self._dtype = runtime.torch_dtype
        self._fps = int(training_args.get("fps", 30))
        self.provenance = {
            "family": "TARDIS",
            "variant": variant,
            "evaluation_role": (
                "full_method" if variant == "A10" else "post_hoc_mechanism_suppression"
            ),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
            "train_manifest": str(train_manifest.resolve()),
            "used_ema": True,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "diffusion_steps": int(training_args["diffusion_steps"]),
            "pretrained_model": str(training_args["pretrained_model"]),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        generator = torch.Generator(device=self._device).manual_seed(seed)
        autocast_enabled = self._device.type == "cuda" and self._dtype != torch.float32
        with torch.autocast(
            device_type=self._device.type,
            dtype=self._dtype,
            enabled=autocast_enabled,
        ):
            if source_video is None:
                output = self._model.generate(
                    [prompt],
                    num_frames=num_frames,
                    fps=self._fps,
                    generator=generator,
                )
            else:
                if tuple(source_video.shape) != (num_frames, 3, height, width):
                    raise ValueError("TARDIS source video does not match protocol dimensions")
                output = self._model.generate_source_conditioned(
                    [prompt],
                    source_video.unsqueeze(0).to(self._device),
                    fps=self._fps,
                    generator=generator,
                    innovation_strength=source_strength,
                )
        return normalize_video(
            output.video,
            num_frames=num_frames,
            height=height,
            width=width,
            input_range="minus_one_one",
        )


class AnimateDiffLightningGenerator:
    method = "animatediff_lightning"

    def __init__(
        self,
        *,
        base_model: Path,
        motion_adapter: Path,
        device: torch.device,
    ) -> None:
        from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
        from safetensors.torch import load_file

        adapter = MotionAdapter()
        adapter.load_state_dict(load_file(motion_adapter, device="cpu"), strict=True)
        pipeline = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
        )
        pipeline.to(device)
        pipeline.vae.enable_slicing()
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        self._device = device
        self.provenance = {
            "family": "AnimateDiff-Lightning",
            "base_model": str(base_model.resolve()),
            "motion_adapter": str(motion_adapter.resolve()),
            "motion_adapter_sha256": _sha256(motion_adapter),
            "diffusion_steps": 2,
            "guidance_scale": 1.0,
            "parameter_count": _module_parameter_count(pipeline),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        del source_video, source_strength
        output = self._pipeline(
            prompt,
            negative_prompt="bad quality, worse quality, low resolution",
            num_frames=num_frames,
            guidance_scale=1.0,
            num_inference_steps=2,
            height=height,
            width=width,
            generator=torch.Generator(device=self._device).manual_seed(seed),
            output_type="pt",
        )
        return normalize_video(
            output.frames,
            num_frames=num_frames,
            height=height,
            width=width,
            input_range="zero_one",
        )


class SDTurboIndependentGenerator:
    method = "sd_turbo_independent"

    def __init__(
        self,
        *,
        model: Path,
        device: torch.device,
        frame_batch_size: int = 8,
    ) -> None:
        from diffusers import AutoPipelineForText2Image

        if frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipeline.to(device)
        pipeline.vae.enable_slicing()
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        self._device = device
        self._frame_batch_size = frame_batch_size
        self.provenance = {
            "family": "SD-Turbo independent-frame",
            "model": str(model.resolve()),
            "diffusion_steps": 1,
            "guidance_scale": 0.0,
            "frame_noise": "independent deterministic seed per frame",
            "parameter_count": _module_parameter_count(pipeline),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        del source_video, source_strength
        frames: list[torch.Tensor] = []
        for start in range(0, num_frames, self._frame_batch_size):
            stop = min(start + self._frame_batch_size, num_frames)
            generators = [
                torch.Generator(device=self._device).manual_seed(_frame_seed(seed, index))
                for index in range(start, stop)
            ]
            output = self._pipeline(
                [prompt] * (stop - start),
                num_inference_steps=1,
                guidance_scale=0.0,
                height=height,
                width=width,
                generator=generators,
                output_type="pt",
            )
            batch = normalize_video(
                output.images,
                num_frames=stop - start,
                height=height,
                width=width,
                input_range="zero_one",
            )
            frames.append(batch)
        return torch.cat(frames, dim=0)


class _SDTurboSourceGenerator:
    method = "source_generator"
    family = "SD-Turbo source-conditioned"
    implementation_scope = "audited core-mechanism reproduction"
    reference_work = "source-conditioned video editing literature"

    def __init__(
        self,
        *,
        model: Path,
        device: torch.device,
        frame_batch_size: int = 4,
    ) -> None:
        from diffusers import AutoPipelineForImage2Image

        if frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive")
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipeline.to(device)
        pipeline.vae.enable_slicing()
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        self._device = device
        self._frame_batch_size = frame_batch_size
        self.provenance = {
            "family": self.family,
            "implementation_scope": self.implementation_scope,
            "reference_work": self.reference_work,
            "model": str(model.resolve()),
            "diffusion_steps": 3,
            "guidance_scale": 0.0,
            "parameter_count": _module_parameter_count(pipeline),
        }

    def _validate_source(
        self,
        source_video: torch.Tensor | None,
        *,
        num_frames: int,
        height: int,
        width: int,
        source_strength: float,
    ) -> torch.Tensor:
        if source_video is None:
            raise ValueError(f"{self.method} requires source_video")
        expected = (num_frames, 3, height, width)
        if tuple(source_video.shape) != expected:
            raise ValueError(f"source video must have shape {expected}")
        if not 0 < source_strength <= 1:
            raise ValueError("source strength must be in (0, 1]")
        return source_video.to(device=self._device, dtype=torch.float32).clamp(-1, 1)

    def _translate(
        self,
        prompt: str,
        source: torch.Tensor,
        *,
        seed: int,
        source_strength: float,
        shared_noise: bool,
    ) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        images = source.add(1).mul(0.5)
        for start in range(0, len(images), self._frame_batch_size):
            stop = min(start + self._frame_batch_size, len(images))
            generators = [
                torch.Generator(device=self._device).manual_seed(
                    seed if shared_noise else _frame_seed(seed, index)
                )
                for index in range(start, stop)
            ]
            output = self._pipeline(
                prompt=[prompt] * (stop - start),
                image=images[start:stop],
                strength=source_strength,
                num_inference_steps=3,
                guidance_scale=0.0,
                generator=generators,
                output_type="pt",
            )
            frames.append(
                normalize_video(
                    output.images,
                    num_frames=stop - start,
                    height=int(source.shape[-2]),
                    width=int(source.shape[-1]),
                    input_range="zero_one",
                )
            )
        return torch.cat(frames, dim=0)


class StreamDiffusionImg2ImgGenerator(_SDTurboSourceGenerator):
    method = "streamdiffusion_img2img"
    family = "StreamDiffusion-style SD-Turbo img2img"
    implementation_scope = "pipeline-equivalent frame-stream reproduction"
    reference_work = "StreamDiffusion (ECCV 2024)"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        source = self._validate_source(
            source_video,
            num_frames=num_frames,
            height=height,
            width=width,
            source_strength=source_strength,
        )
        return self._translate(
            prompt,
            source,
            seed=seed,
            source_strength=source_strength,
            shared_noise=False,
        )


class RerenderFlowGenerator(_SDTurboSourceGenerator):
    method = "rerender_flow"
    family = "Rerender-A-Video keyframe and flow propagation"
    implementation_scope = "audited core-mechanism reproduction, four-frame keyframes"
    reference_work = "Rerender-A-Video (SIGGRAPH Asia 2023)"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        source = self._validate_source(
            source_video,
            num_frames=num_frames,
            height=height,
            width=width,
            source_strength=source_strength,
        )
        keyframe_indices = list(range(0, num_frames, 4))
        translated = self._translate(
            prompt,
            source[keyframe_indices],
            seed=seed,
            source_strength=source_strength,
            shared_noise=True,
        )
        keyframes = dict(zip(keyframe_indices, translated, strict=True))
        output = [keyframes[0]]
        for index in range(1, num_frames):
            if index in keyframes:
                output.append(keyframes[index])
                continue
            warped = _farneback_warp(output[-1], source[index - 1], source[index])
            output.append(torch.lerp(warped, source[index], 0.12))
        return torch.stack(output)


class TokenFlowCoreGenerator(_SDTurboSourceGenerator):
    method = "tokenflow_core"
    family = "TokenFlow correspondence propagation"
    implementation_scope = "audited flow-correspondence core reproduction"
    reference_work = "TokenFlow (ICLR 2024)"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        source = self._validate_source(
            source_video,
            num_frames=num_frames,
            height=height,
            width=width,
            source_strength=source_strength,
        )
        translated = self._translate(
            prompt,
            source,
            seed=seed,
            source_strength=source_strength,
            shared_noise=True,
        )
        output = [translated[0]]
        for index in range(1, num_frames):
            propagated = _farneback_warp(output[-1], source[index - 1], source[index])
            output.append(torch.lerp(translated[index], propagated, 0.58))
        return torch.stack(output)


class Vid2VidZeroCoreGenerator(_SDTurboSourceGenerator):
    method = "vid2vid_zero_core"
    family = "vid2vid-zero cross-frame anchoring"
    implementation_scope = "audited shared-noise and first-frame-anchor reproduction"
    reference_work = "vid2vid-zero"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        source = self._validate_source(
            source_video,
            num_frames=num_frames,
            height=height,
            width=width,
            source_strength=source_strength,
        )
        translated = self._translate(
            prompt,
            source,
            seed=seed,
            source_strength=source_strength,
            shared_noise=True,
        )
        output = [translated[0]]
        for index in range(1, num_frames):
            anchor = _farneback_warp(output[-1], source[index - 1], source[index])
            output.append(torch.lerp(translated[index], anchor, 0.32))
        return torch.stack(output)


class ControlVideoCannyGenerator(_SDTurboSourceGenerator):
    method = "controlvideo_canny"
    family = "ControlVideo Canny structural control"
    implementation_scope = "audited Canny-condition core reproduction without ControlNet weights"
    reference_work = "ControlVideo (ICCV 2023)"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        source = self._validate_source(
            source_video,
            num_frames=num_frames,
            height=height,
            width=width,
            source_strength=source_strength,
        )
        edges = _canny_video(source)
        controlled = torch.lerp(source, edges, 0.35)
        return self._translate(
            prompt,
            controlled,
            seed=seed,
            source_strength=source_strength,
            shared_noise=True,
        )


class StableVideoPropagationGenerator(_SDTurboSourceGenerator):
    method = "stablevideo_propagation"
    family = "StableVideo inter-frame propagation"
    implementation_scope = "audited sparse-keyframe propagation core reproduction"
    reference_work = "StableVideo (ICCV 2023)"

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        source = self._validate_source(
            source_video,
            num_frames=num_frames,
            height=height,
            width=width,
            source_strength=source_strength,
        )
        keyframe_indices = sorted({0, num_frames // 2, num_frames - 1})
        translated = self._translate(
            prompt,
            source[keyframe_indices],
            seed=seed,
            source_strength=source_strength,
            shared_noise=True,
        )
        keyframes = dict(zip(keyframe_indices, translated, strict=True))
        output = [keyframes[0]]
        for index in range(1, num_frames):
            if index in keyframes:
                output.append(keyframes[index])
                continue
            propagated = _farneback_warp(output[-1], source[index - 1], source[index])
            motion = (source[index] - source[index - 1]).abs().mean().clamp(0, 1)
            source_mix = float((0.08 + 0.22 * motion).item())
            output.append(torch.lerp(propagated, source[index], source_mix))
        return torch.stack(output)


class TextToVideoZeroGenerator:
    method = "text2video_zero"

    def __init__(self, *, model: Path, device: torch.device) -> None:
        from diffusers import TextToVideoZeroPipeline

        pipeline = TextToVideoZeroPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.to(device)
        pipeline.vae.enable_slicing()
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        self._device = device
        self.provenance = {
            "family": "Text2Video-Zero",
            "model": str(model.resolve()),
            "diffusion_steps": 8,
            "guidance_scale": 7.5,
            "t0": 3,
            "t1": 6,
            "parameter_count": _module_parameter_count(pipeline),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        source_video: torch.Tensor | None = None,
        source_strength: float = 0.45,
    ) -> torch.Tensor:
        del source_video, source_strength
        output = self._pipeline(
            prompt=prompt,
            video_length=num_frames,
            height=height,
            width=width,
            num_inference_steps=8,
            guidance_scale=7.5,
            generator=torch.Generator(device=self._device).manual_seed(seed),
            t0=3,
            t1=6,
            frame_ids=list(range(num_frames)),
            output_type="tensor",
        )
        return normalize_video(
            output.images,
            num_frames=num_frames,
            height=height,
            width=width,
            input_range="zero_one",
        )


def build_generator(
    method: str,
    *,
    dataset: str,
    checkpoint: Path | None,
    train_manifest: Path | None,
    device: torch.device,
    precision: str,
    sd_turbo_model: Path,
    sd15_model: Path,
    animatediff_adapter: Path,
) -> VideoGenerator:
    """Construct one named benchmark without network access."""

    if method == "tardis" or method.startswith("tardis_a"):
        if checkpoint is None or train_manifest is None:
            raise ValueError("TARDIS experiments require checkpoint and train manifest")
        variant = "A10" if method == "tardis" else method.removeprefix("tardis_").upper()
        return TARDISGenerator(
            dataset=dataset,
            checkpoint=checkpoint,
            train_manifest=train_manifest,
            device=device,
            precision=precision,
            variant=variant,
        )
    if method == "animatediff_lightning":
        return AnimateDiffLightningGenerator(
            base_model=sd15_model,
            motion_adapter=animatediff_adapter,
            device=device,
        )
    if method == "sd_turbo_independent":
        return SDTurboIndependentGenerator(model=sd_turbo_model, device=device)
    source_generators = {
        "streamdiffusion_img2img": StreamDiffusionImg2ImgGenerator,
        "rerender_flow": RerenderFlowGenerator,
        "tokenflow_core": TokenFlowCoreGenerator,
        "vid2vid_zero_core": Vid2VidZeroCoreGenerator,
        "controlvideo_canny": ControlVideoCannyGenerator,
        "stablevideo_propagation": StableVideoPropagationGenerator,
    }
    source_factory = source_generators.get(method)
    if source_factory is not None:
        return source_factory(model=sd_turbo_model, device=device)
    if method == "text2video_zero":
        return TextToVideoZeroGenerator(model=sd15_model, device=device)
    raise ValueError(f"unknown benchmark method: {method!r}")


def _farneback_warp(
    generated_previous: torch.Tensor,
    source_previous: torch.Tensor,
    source_current: torch.Tensor,
) -> torch.Tensor:
    previous_gray = _frame_gray_uint8(source_previous)
    current_gray = _frame_gray_uint8(source_current)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    height, width = previous_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    frame = generated_previous.detach().to(dtype=torch.float32, device="cpu")
    frame_hwc = frame.permute(1, 2, 0).numpy()
    warped = cv2.remap(
        frame_hwc,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return (
        torch.from_numpy(warped)
        .permute(2, 0, 1)
        .to(device=generated_previous.device, dtype=generated_previous.dtype)
        .clamp(-1, 1)
    )


def _frame_gray_uint8(frame: torch.Tensor) -> np.ndarray:
    image = (
        frame.detach()
        .to(dtype=torch.float32, device="cpu")
        .add(1)
        .mul(127.5)
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _canny_video(source: torch.Tensor) -> torch.Tensor:
    frames = []
    for frame in source:
        edges = cv2.Canny(_frame_gray_uint8(frame), 100, 200)
        tensor = torch.from_numpy(edges).to(device=source.device, dtype=source.dtype)
        tensor = tensor.div(127.5).sub(1).unsqueeze(0).expand(3, -1, -1)
        frames.append(tensor)
    return torch.stack(frames)


def _frame_seed(seed: int, frame_index: int) -> int:
    payload = f"{seed}\x1f{frame_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_parameter_count(value: Any) -> int:
    components = (
        getattr(value, name, None)
        for name in ("unet", "vae", "text_encoder", "motion_adapter")
    )
    seen: set[int] = set()
    total = 0
    for component in components:
        if not isinstance(component, torch.nn.Module):
            continue
        for parameter in component.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return total


__all__ = ["VideoGenerator", "build_generator"]
