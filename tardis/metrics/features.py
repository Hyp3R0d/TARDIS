"""Lazy frozen production feature adapters with explicit model provenance."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import nn


class _LazyFeatureAdapter:
    model_id: str
    weights_id: str
    provenance_id: str
    feature_dim: int

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device or "cpu")
        self._model: nn.Module | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _freeze(self, model: nn.Module) -> nn.Module:
        model.to(self.device)
        model.requires_grad_(False)
        model.eval()
        return model


class AlexNetLPIPS(_LazyFeatureAdapter):
    """LPIPS 0.1 with the calibrated AlexNet backbone."""

    model_id = "alex"
    weights_id = "lpips-v0.1"
    provenance_id = "lpips:alex:lpips-v0.1"
    feature_dim = 1

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        model = self._load()
        result = model(
            generated.to(device=self.device, dtype=torch.float32),
            reference.to(device=self.device, dtype=torch.float32),
        )
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("LPIPS AlexNet returned a non-tensor result")
        return result

    def _load(self) -> nn.Module:
        if self._model is not None:
            return self._model
        try:
            module = importlib.import_module("lpips")
        except ImportError as error:
            raise RuntimeError(
                "lpips is required for the AlexNet LPIPS production metric"
            ) from error
        factory = cast(Callable[..., object], module.LPIPS)
        model = factory(net="alex", version="0.1", pretrained=True)
        if not isinstance(model, nn.Module):
            raise RuntimeError("lpips.LPIPS did not construct a torch module")
        self._model = self._freeze(model)
        return self._model


class InceptionV3PoolFeatures(_LazyFeatureAdapter):
    """Torchvision Inception-v3 global-pool features for FID statistics."""

    model_id = "torchvision/inception_v3"
    weights_id = "IMAGENET1K_V1"
    provenance_id = "torchvision/inception_v3:IMAGENET1K_V1:avgpool"
    feature_dim = 2048

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        model = self._load()
        frames = _prepare_image_frames(
            video,
            size=299,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            device=self.device,
        )
        result = model(frames)
        if not isinstance(result, torch.Tensor) or result.shape != (
            video.shape[0],
            self.feature_dim,
        ):
            raise RuntimeError("Inception-v3 pool extractor returned an incompatible shape")
        return result.detach()

    def _load(self) -> nn.Module:
        if self._model is not None:
            return self._model
        try:
            module = importlib.import_module("torchvision.models")
        except ImportError as error:
            raise RuntimeError("torchvision is required for Inception-v3 FID features") from error
        weights_type = module.Inception_V3_Weights
        weights = getattr(weights_type, self.weights_id)
        factory = cast(Callable[..., object], module.inception_v3)
        model = factory(weights=weights, transform_input=False)
        if not isinstance(model, nn.Module):
            raise RuntimeError("torchvision.inception_v3 did not construct a torch module")
        model.fc = nn.Identity()
        self._model = self._freeze(model)
        return self._model


class I3DKineticsFeatures(_LazyFeatureAdapter):
    """Kinetics-400 I3D-R50 penultimate pooled video features for FVD."""

    model_id = "pytorchvideo/i3d_r50"
    weights_id = "KINETICS400"
    provenance_id = "pytorchvideo/i3d_r50:KINETICS400:pre-logits"
    feature_dim = 2048

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        super().__init__(device=device)
        self._projection: nn.Module | None = None

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        model, projection = self._load()
        frames = _prepare_image_frames(
            video,
            size=224,
            mean=(0.45, 0.45, 0.45),
            std=(0.225, 0.225, 0.225),
            device=self.device,
        )
        if frames.shape[0] < 8:
            indices = (
                torch.linspace(
                    0,
                    frames.shape[0] - 1,
                    steps=8,
                    device=frames.device,
                )
                .round()
                .to(torch.int64)
            )
            frames = frames.index_select(0, indices)
        clip = frames.permute(1, 0, 2, 3).unsqueeze(0)
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, inputs: tuple[object, ...]) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured.append(inputs[0].detach())

        handle = cast(Any, projection).register_forward_pre_hook(capture)
        try:
            model(clip)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("I3D projection hook did not capture one feature tensor")
        return _pool_feature_axis(captured[0], self.feature_dim)

    def _load(self) -> tuple[nn.Module, nn.Module]:
        if self._model is not None and self._projection is not None:
            return self._model, self._projection
        try:
            module = importlib.import_module("pytorchvideo.models.hub")
        except ImportError as error:
            raise RuntimeError("pytorchvideo is required for Kinetics I3D FVD features") from error
        factory = cast(Callable[..., object], module.i3d_r50)
        model = factory(pretrained=True)
        if not isinstance(model, nn.Module):
            raise RuntimeError("pytorchvideo i3d_r50 did not construct a torch module")
        blocks = cast(Any, model).blocks
        projection = cast(object, blocks[-1].proj)
        if not isinstance(projection, nn.Module):
            raise RuntimeError("pytorchvideo I3D head has no torch projection module")
        self._model = self._freeze(model)
        self._projection = projection
        return self._model, self._projection


class OpenCLIPFeatures(_LazyFeatureAdapter):
    """OpenCLIP ViT-B/32 frame and prompt features for cosine scoring."""

    model_id = "ViT-B-32"
    weights_id = "openai"
    provenance_id = "open_clip:ViT-B-32:openai"
    feature_dim = 512

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        super().__init__(device=device)
        self._tokenizer: Callable[[list[str]], torch.Tensor] | None = None

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        model, _ = self._load()
        frames = _prepare_image_frames(
            video,
            size=224,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            device=self.device,
        )
        result = cast(Any, model).encode_image(frames)
        if not isinstance(result, torch.Tensor) or result.shape != (
            video.shape[0],
            self.feature_dim,
        ):
            raise RuntimeError("OpenCLIP image encoder returned an incompatible shape")
        return result.detach()

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        model, tokenizer = self._load()
        if not prompt.strip():
            raise ValueError("OpenCLIP prompt must be non-empty")
        tokens = tokenizer([prompt]).to(self.device)
        result = cast(Any, model).encode_text(tokens)
        if not isinstance(result, torch.Tensor) or result.shape != (1, self.feature_dim):
            raise RuntimeError("OpenCLIP text encoder returned an incompatible shape")
        return result.detach()

    def _load(self) -> tuple[nn.Module, Callable[[list[str]], torch.Tensor]]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            module = importlib.import_module("open_clip")
        except ImportError as error:
            raise RuntimeError("open_clip is required for OpenCLIP ViT-B/32 features") from error
        create = cast(Callable[..., object], module.create_model_and_transforms)
        created = create(self.model_id, pretrained=self.weights_id)
        if not isinstance(created, tuple) or not created or not isinstance(created[0], nn.Module):
            raise RuntimeError("open_clip did not construct a torch model")
        tokenizer_factory = cast(Callable[[str], object], module.get_tokenizer)
        tokenizer = tokenizer_factory(self.model_id)
        if not callable(tokenizer):
            raise RuntimeError("open_clip did not construct a tokenizer")
        self._model = self._freeze(created[0])
        self._tokenizer = cast(Callable[[list[str]], torch.Tensor], tokenizer)
        return self._model, self._tokenizer


LPIPSFeatureExtractor = AlexNetLPIPS
InceptionFeatureExtractor = InceptionV3PoolFeatures
I3DFeatureExtractor = I3DKineticsFeatures
OpenCLIPFeatureExtractor = OpenCLIPFeatures


def _prepare_image_frames(
    video: torch.Tensor,
    *,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    frames = video.detach().to(device=device, dtype=torch.float32).add(1).mul(0.5)
    frames = functional.interpolate(
        frames,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    mean_tensor = torch.tensor(mean, device=device).reshape(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).reshape(1, 3, 1, 1)
    return cast(torch.Tensor, (frames - mean_tensor) / std_tensor)


def _pool_feature_axis(features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if features.shape[0] != 1:
        raise RuntimeError("I3D feature tensor must contain one video")
    axes = [index for index, size in enumerate(features.shape[1:], start=1) if size == feature_dim]
    if len(axes) != 1:
        raise RuntimeError("I3D feature tensor does not expose one 2048-dimensional axis")
    channel_axis = axes[0]
    moved = features.movedim(channel_axis, -1)
    return moved.reshape(1, -1, feature_dim).mean(dim=1)
