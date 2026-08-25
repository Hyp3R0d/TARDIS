from __future__ import annotations

import torch
from torch import nn

from tardis.models.priors import (
    DiffusersFirstFrameGenerator,
    DiffusersVaeCodec,
    FrozenPriorBundle,
    load_sd_turbo_prior_bundle,
)


class TinyCodec(nn.Module):
    latent_channels = 4
    spatial_scale = 8

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.decoder = nn.Conv2d(4, 3, kernel_size=1)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        flattened = video.reshape(batch * frames, channels, height, width)
        latent = torch.nn.functional.adaptive_avg_pool2d(self.encoder(flattened), (64, 64))
        return latent.reshape(batch, frames, 4, 64, 64)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, _, _ = latents.shape
        flattened = latents.reshape(batch * frames, channels, 64, 64)
        pixels = torch.nn.functional.interpolate(
            self.decoder(flattened), size=(512, 512), mode="bilinear", align_corners=False
        )
        return pixels.reshape(batch, frames, 3, 512, 512)


class TinyTextConditioner(nn.Module):
    embedding_dim = 16
    max_length = 8

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
        self.projection = nn.Linear(16, 4)

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
        return latent.expand(-1, -1, height // 8, width // 8).contiguous()


def build_bundle() -> FrozenPriorBundle:
    return FrozenPriorBundle(TinyCodec(), TinyTextConditioner(), TinyFirstFrameGenerator())


def test_frozen_prior_video_text_and_first_frame_contracts() -> None:
    bundle = build_bundle()
    video = torch.randn(2, 3, 3, 512, 512, requires_grad=True)

    latents = bundle.encode_video(video)
    text, mask = bundle.encode_text(["alpha", "beta"])
    first = bundle.generate_first_latent(
        text,
        mask,
        generator=torch.Generator().manual_seed(7),
        height=512,
        width=512,
    )

    assert latents.shape == (2, 3, 4, 64, 64)
    assert text.shape == (2, 8, 16)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool
    assert first.shape == (2, 4, 64, 64)
    assert latents.requires_grad is False
    assert text.requires_grad is False
    assert first.requires_grad is False


def test_all_prior_parameters_are_frozen_and_stay_in_eval_mode() -> None:
    bundle = build_bundle()

    bundle.train(True)

    assert bundle.training is False
    assert all(module.training is False for module in bundle.modules())
    assert all(parameter.requires_grad is False for parameter in bundle.parameters())


def test_codec_round_trip_preserves_public_tensor_geometry() -> None:
    bundle = build_bundle()
    video = torch.randn(1, 2, 3, 512, 512)

    reconstructed = bundle.decode_video(bundle.encode_video(video))

    assert reconstructed.shape == video.shape
    assert reconstructed.requires_grad is False


def test_frozen_decoder_propagates_gradient_to_generated_latents_only() -> None:
    bundle = build_bundle()
    latents = torch.randn(1, 2, 4, 64, 64, requires_grad=True)

    decoded = bundle.decode_video(latents)
    decoded.square().mean().backward()

    assert decoded.requires_grad
    assert latents.grad is not None
    assert all(parameter.grad is None for parameter in bundle.parameters())


def test_diffusers_vae_decoder_checkpoints_training_frames_individually() -> None:
    class DecodeOutput:
        def __init__(self, sample: torch.Tensor) -> None:
            self.sample = sample

    class FakeVae(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def decode(self, value: torch.Tensor) -> DecodeOutput:
            self.batch_sizes.append(value.shape[0])
            pixels = torch.nn.functional.interpolate(value[:, :3], scale_factor=8)
            return DecodeOutput(pixels)

    vae = FakeVae()
    codec = DiffusersVaeCodec(vae, scaling_factor=1.0, training_decode_chunk_size=1)
    latents = torch.randn(1, 3, 4, 2, 2, requires_grad=True)

    decoded = codec.decode(latents)
    decoded.square().mean().backward()

    assert decoded.shape == (1, 3, 3, 16, 16)
    assert vae.batch_sizes and max(vae.batch_sizes) == 1
    assert latents.grad is not None

    vae.batch_sizes.clear()
    with torch.no_grad():
        codec.decode(latents.detach())
    assert vae.batch_sizes == [3]


def test_prior_bundle_rejects_malformed_video_and_first_frame_shape() -> None:
    bundle = build_bundle()

    try:
        bundle.encode_video(torch.randn(1, 3, 512, 512))
    except ValueError as error:
        assert "[B,T,3,H,W]" in str(error)
    else:
        raise AssertionError("rank-four input must fail")


def test_tiny_prior_doubles_have_no_accidental_gradient_after_use() -> None:
    bundle = build_bundle()
    latents = bundle.encode_video(torch.randn(1, 2, 3, 512, 512))

    assert latents.grad_fn is None
    assert all(parameter.grad is None for parameter in bundle.parameters())


class FakePipelineOutput:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images


class FakeTurboPipeline:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}
        self.vae = FakePriorModule()
        self.text_encoder = FakeTextEncoder()
        self.tokenizer = FakeTokenizer()
        self.unet = FakePriorModule()
        self.progress_bar_disabled = False

    def __call__(self, **kwargs: object) -> FakePipelineOutput:
        self.arguments = kwargs
        prompt_embeddings = kwargs["prompt_embeds"]
        assert isinstance(prompt_embeddings, torch.Tensor)
        return FakePipelineOutput(torch.ones(prompt_embeddings.shape[0], 4, 64, 64))

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_bar_disabled = disable


class FakeConfig:
    scaling_factor = 0.18215
    hidden_size = 16


class FakePriorModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = FakeConfig()

    def encode(self, value: torch.Tensor) -> object:
        del value
        raise NotImplementedError

    def decode(self, value: torch.Tensor) -> object:
        del value
        raise NotImplementedError


class FakeTextEncoder(FakePriorModule):
    def forward(self, **kwargs: torch.Tensor) -> object:
        input_ids = kwargs["input_ids"]
        output = type("TextOutput", (), {})()
        output.last_hidden_state = torch.ones(input_ids.shape[0], input_ids.shape[1], 16)
        return output


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, prompts: list[str], **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.ones(len(prompts), 8, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), 8, dtype=torch.long),
        }


def test_diffusers_first_frame_adapter_runs_one_step_latent_generation() -> None:
    pipeline = FakeTurboPipeline()
    adapter = DiffusersFirstFrameGenerator(pipeline)
    embeddings = torch.randn(2, 8, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]).bool()
    generator = torch.Generator().manual_seed(9)

    latent = adapter.generate_first_latent(
        embeddings,
        mask,
        generator=generator,
        height=512,
        width=512,
    )

    assert latent.shape == (2, 4, 64, 64)
    assert pipeline.arguments["num_inference_steps"] == 1
    assert pipeline.arguments["guidance_scale"] == 0.0
    assert pipeline.arguments["output_type"] == "latent"
    masked = pipeline.arguments["prompt_embeds"]
    assert isinstance(masked, torch.Tensor)
    assert torch.count_nonzero(masked[0, 3:]) == 0
    assert adapter.unet_module is pipeline.unet


def test_sd_turbo_bundle_loads_one_shared_pipeline_and_registers_all_modules() -> None:
    pipeline = FakeTurboPipeline()
    calls: list[tuple[str, dict[str, object]]] = []

    def factory(model_id: str, **kwargs: object) -> FakeTurboPipeline:
        calls.append((model_id, kwargs))
        return pipeline

    bundle = load_sd_turbo_prior_bundle(
        "mirror/sd-turbo",
        cache_dir="/tmp/model-cache",
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        local_files_only=True,
        pipeline_factory=factory,
    )

    assert calls == [
        (
            "mirror/sd-turbo",
            {
                "cache_dir": "/tmp/model-cache",
                "torch_dtype": torch.float32,
                "local_files_only": True,
                "safety_checker": None,
                "requires_safety_checker": False,
            },
        )
    ]
    assert pipeline.progress_bar_disabled
    assert bundle.codec_module.vae is pipeline.vae
    assert bundle.text_module.text_encoder is pipeline.text_encoder
    assert bundle.first_frame_module.unet_module is pipeline.unet
    assert all(not parameter.requires_grad for parameter in bundle.parameters())
    assert all(not module.training for module in bundle.modules())
