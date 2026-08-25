"""Frozen semantic-prior adapters used outside the claimed temporal network."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tardis.models.contracts import FirstFrameGenerator, LatentCodec, TextConditioner


class FrozenPriorBundle(nn.Module):
    """Enforce a gradient-free boundary around codec, text, and keyframe priors."""

    def __init__(
        self,
        codec: nn.Module,
        text_conditioner: nn.Module,
        first_frame_generator: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(codec, LatentCodec):
            raise TypeError("codec does not satisfy LatentCodec")
        if not isinstance(text_conditioner, TextConditioner):
            raise TypeError("text_conditioner does not satisfy TextConditioner")
        if not isinstance(first_frame_generator, FirstFrameGenerator):
            raise TypeError("first_frame_generator does not satisfy FirstFrameGenerator")
        self.codec_module = codec
        self.text_module = text_conditioner
        self.first_frame_module = first_frame_generator
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> FrozenPriorBundle:
        """Frozen priors remain in evaluation mode regardless of parent state."""

        del mode
        super().train(False)
        return self

    @property
    def latent_channels(self) -> int:
        return cast(LatentCodec, self.codec_module).latent_channels

    @property
    def spatial_scale(self) -> int:
        return cast(LatentCodec, self.codec_module).spatial_scale

    @property
    def text_dim(self) -> int:
        return cast(TextConditioner, self.text_module).embedding_dim

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("video must have shape [B,T,3,H,W]")
        result = cast(LatentCodec, self.codec_module).encode(video)
        expected = (video.shape[0], video.shape[1], self.latent_channels)
        if result.ndim != 5 or result.shape[:3] != expected:
            raise ValueError("codec encode must return [B,T,C,H_latent,W_latent] with matching B/T")
        return result.detach()

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[2] != self.latent_channels:
            raise ValueError("latents must have shape [B,T,C,H,W]")
        result = cast(LatentCodec, self.codec_module).decode(latents)
        if result.ndim != 5 or result.shape[:2] != latents.shape[:2] or result.shape[2] != 3:
            raise ValueError("codec decode must return [B,T,3,H,W] with matching B/T")
        return result

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain non-empty strings")
        embeddings, mask = cast(TextConditioner, self.text_module).encode_text(prompts)
        if embeddings.ndim != 3 or embeddings.shape[0] != len(prompts):
            raise ValueError("text conditioner must return embeddings [B,L,D]")
        if mask.shape != embeddings.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("text conditioner mask must be bool [B,L]")
        return embeddings.detach(), mask.detach()

    @torch.no_grad()
    def generate_first_latent(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        generator: torch.Generator | Sequence[torch.Generator],
        height: int,
        width: int,
    ) -> torch.Tensor:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        result = cast(FirstFrameGenerator, self.first_frame_module).generate_first_latent(
            text_embeddings,
            text_mask,
            generator=generator,
            height=height,
            width=width,
        )
        expected = (
            text_embeddings.shape[0],
            self.latent_channels,
            height // self.spatial_scale,
            width // self.spatial_scale,
        )
        if result.shape != expected:
            raise ValueError(
                f"first-frame prior returned {tuple(result.shape)}; expected {expected}"
            )
        return result.detach()


class DiffusersVaeCodec(nn.Module):
    """Lazy-compatible adapter around a diffusers AutoencoderKL instance."""

    latent_channels = 4
    spatial_scale = 8

    def __init__(
        self,
        vae: nn.Module,
        *,
        scaling_factor: float,
        training_decode_chunk_size: int = 1,
        inference_decode_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if training_decode_chunk_size <= 0:
            raise ValueError("training_decode_chunk_size must be positive")
        if inference_decode_chunk_size is not None and inference_decode_chunk_size <= 0:
            raise ValueError("inference_decode_chunk_size must be positive when provided")
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.training_decode_chunk_size = training_decode_chunk_size
        self.inference_decode_chunk_size = inference_decode_chunk_size

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        flattened = video.reshape(batch * frames, channels, height, width)
        encoded = cast(Any, self.vae).encode(flattened)
        latent_dist = cast(Any, encoded).latent_dist
        latent = cast(torch.Tensor, latent_dist.mode()) * self.scaling_factor
        return latent.reshape(batch, frames, self.latent_channels, height // 8, width // 8)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = latents.shape
        flattened = latents.reshape(batch * frames, channels, height, width)
        scaled = flattened / self.scaling_factor
        if torch.is_grad_enabled() and scaled.requires_grad:
            decoded_chunks = [
                cast(
                    torch.Tensor,
                    checkpoint(
                        self._decode_chunk,
                        scaled[start : start + self.training_decode_chunk_size],
                        use_reentrant=False,
                    ),
                )
                for start in range(0, scaled.shape[0], self.training_decode_chunk_size)
            ]
            decoded = torch.cat(decoded_chunks)
        elif self.inference_decode_chunk_size is not None:
            decoded = torch.cat(
                [
                    self._decode_chunk(
                        scaled[start : start + self.inference_decode_chunk_size]
                    )
                    for start in range(0, scaled.shape[0], self.inference_decode_chunk_size)
                ]
            )
        else:
            decoded = self._decode_chunk(scaled)
        return decoded.reshape(batch, frames, 3, height * 8, width * 8)

    def _decode_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        output = cast(Any, self.vae).decode(latents)
        decoded = cast(torch.Tensor, output.sample)
        if decoded.ndim != 4 or decoded.shape[0] != latents.shape[0] or decoded.shape[1] != 3:
            raise RuntimeError("SD-Turbo VAE decode returned an incompatible pixel tensor")
        return decoded


class ClipTextConditioner(nn.Module):
    """Adapter around a tokenizer and CLIP text encoder without eager imports."""

    def __init__(
        self,
        tokenizer: Any,
        text_encoder: nn.Module,
        *,
        embedding_dim: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim
        self.max_length = max_length

    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.text_encoder.parameters()).device
        tokens: Mapping[str, torch.Tensor] = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state, attention_mask.bool()


class DiffusersFirstFrameGenerator(nn.Module):
    """Use a frozen SD-Turbo-compatible pipeline for a one-step latent keyframe."""

    def __init__(self, pipeline: Any) -> None:
        super().__init__()
        unet = getattr(pipeline, "unet", None)
        if not isinstance(unet, nn.Module):
            raise TypeError("SD-Turbo pipeline must expose a torch UNet module")
        self.unet_module = unet
        object.__setattr__(self, "pipeline", pipeline)

    def generate_first_latent(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        generator: torch.Generator,
        height: int,
        width: int,
    ) -> torch.Tensor:
        masked_embeddings = text_embeddings * text_mask.unsqueeze(-1).to(text_embeddings.dtype)
        output = cast(Callable[..., Any], self.pipeline)(
            prompt_embeds=masked_embeddings,
            num_inference_steps=1,
            guidance_scale=0.0,
            output_type="latent",
            generator=generator,
            height=height,
            width=width,
        )
        latent = cast(torch.Tensor, output.images)
        if latent.ndim != 4:
            raise ValueError("SD-Turbo first-frame pipeline must return latent [B,C,H,W]")
        return latent


def load_sd_turbo_prior_bundle(
    model_id: str,
    *,
    cache_dir: str | None,
    torch_dtype: torch.dtype,
    device: torch.device,
    local_files_only: bool = False,
    pipeline_factory: Callable[..., object] | None = None,
) -> FrozenPriorBundle:
    """Load one shared SD-Turbo pipeline and expose frozen typed prior adapters."""

    if not model_id.strip():
        raise ValueError("model_id must be non-empty")
    if pipeline_factory is None:
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as error:
            raise RuntimeError(
                "diffusers is required to load the SD-Turbo production prior"
            ) from error
        pipeline_factory = cast(Callable[..., object], AutoPipelineForText2Image.from_pretrained)
    pipeline = pipeline_factory(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        safety_checker=None,
        requires_safety_checker=False,
    )
    vae = getattr(pipeline, "vae", None)
    text_encoder = getattr(pipeline, "text_encoder", None)
    tokenizer = getattr(pipeline, "tokenizer", None)
    if not isinstance(vae, nn.Module) or not isinstance(text_encoder, nn.Module):
        raise RuntimeError("SD-Turbo pipeline must expose VAE and text encoder modules")
    if tokenizer is None:
        raise RuntimeError("SD-Turbo pipeline must expose a tokenizer")
    vae_config = getattr(vae, "config", None)
    text_config = getattr(text_encoder, "config", None)
    scaling_factor = getattr(vae_config, "scaling_factor", None)
    embedding_dim = getattr(text_config, "hidden_size", None)
    max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(scaling_factor, int | float) or float(scaling_factor) <= 0:
        raise RuntimeError("SD-Turbo VAE has no positive scaling factor")
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise RuntimeError("SD-Turbo text encoder has no positive hidden size")
    if not isinstance(max_length, int) or max_length <= 0:
        raise RuntimeError("SD-Turbo tokenizer has no positive model_max_length")
    configure_progress = getattr(pipeline, "set_progress_bar_config", None)
    if callable(configure_progress):
        cast(Callable[..., object], configure_progress)(disable=True)
    bundle = FrozenPriorBundle(
        DiffusersVaeCodec(
            vae,
            scaling_factor=float(scaling_factor),
            inference_decode_chunk_size=16,
        ),
        ClipTextConditioner(
            tokenizer,
            text_encoder,
            embedding_dim=embedding_dim,
            max_length=max_length,
        ),
        DiffusersFirstFrameGenerator(pipeline),
    )
    bundle.to(device=device, dtype=torch_dtype)
    bundle.train(False)
    return bundle


def load_sd_turbo_components(
    model_id: str,
    *,
    cache_dir: str | None,
    torch_dtype: torch.dtype,
    local_files_only: bool = False,
) -> tuple[nn.Module, nn.Module, Any]:
    """Load frozen SD-Turbo components lazily so lightweight CLIs stay importable."""

    try:
        from diffusers import AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as error:
        raise RuntimeError(
            "diffusers and transformers are required to load the SD-Turbo production prior"
        ) from error
    common = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    vae_factory = cast(Callable[..., object], AutoencoderKL.from_pretrained)
    text_encoder_factory = cast(Callable[..., object], CLIPTextModel.from_pretrained)
    tokenizer_factory = cast(Callable[..., object], CLIPTokenizer.from_pretrained)
    vae = vae_factory(
        model_id,
        subfolder="vae",
        torch_dtype=torch_dtype,
        **common,
    )
    text_encoder = text_encoder_factory(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        **common,
    )
    tokenizer = tokenizer_factory(model_id, subfolder="tokenizer", **common)
    if not isinstance(vae, nn.Module) or not isinstance(text_encoder, nn.Module):
        raise RuntimeError("SD-Turbo factories must return VAE and text encoder torch modules")
    return vae, text_encoder, tokenizer
