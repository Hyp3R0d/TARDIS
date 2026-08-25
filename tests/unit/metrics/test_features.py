from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


def _features() -> ModuleType:
    return importlib.import_module("tardis.metrics.features")


def test_production_feature_adapters_are_lazy_and_record_exact_provenance() -> None:
    features = _features()
    adapters = [
        features.AlexNetLPIPS(),
        features.InceptionV3PoolFeatures(),
        features.I3DKineticsFeatures(),
        features.OpenCLIPFeatures(),
    ]

    assert [(adapter.model_id, adapter.weights_id) for adapter in adapters] == [
        ("alex", "lpips-v0.1"),
        ("torchvision/inception_v3", "IMAGENET1K_V1"),
        ("pytorchvideo/i3d_r50", "KINETICS400"),
        ("ViT-B-32", "openai"),
    ]
    assert all(not adapter.loaded for adapter in adapters)
    assert all(adapter.provenance_id for adapter in adapters)


def test_i3d_fvd_runtime_dependency_is_declared_for_clean_installs() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.startswith("pytorchvideo") for dependency in dependencies)


class _FakeLPIPSModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3), keepdim=True) * self.scale


def test_alexnet_lpips_loads_once_with_exact_weights_and_stays_frozen_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features()
    model = _FakeLPIPSModel()
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> nn.Module:
        calls.append(kwargs)
        return model

    module = SimpleNamespace(LPIPS=factory)
    monkeypatch.setattr(features.importlib, "import_module", lambda name: module)
    adapter = features.AlexNetLPIPS(device="cpu")
    generated = torch.zeros(2, 3, 4, 4)
    reference = torch.ones_like(generated)

    first = adapter(generated, reference)
    second = adapter(generated, reference)

    assert calls == [{"net": "alex", "version": "0.1", "pretrained": True}]
    assert first.shape == (2, 1, 1, 1)
    assert torch.equal(first, second)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_alexnet_lpips_backpropagates_to_generated_pixels_but_not_metric_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features()
    model = _FakeLPIPSModel()
    monkeypatch.setattr(
        features.importlib,
        "import_module",
        lambda name: SimpleNamespace(LPIPS=lambda **kwargs: model),
    )
    adapter = features.AlexNetLPIPS(device="cpu")
    generated = torch.zeros(2, 3, 4, 4, requires_grad=True)
    reference = torch.ones_like(generated)

    adapter(generated, reference).mean().backward()

    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.count_nonzero(generated.grad).item() == generated.numel()
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


class _MinimumTemporalI3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Identity()
        self.clip: torch.Tensor | None = None

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        self.clip = clip
        if clip.shape[2] < 8:
            raise RuntimeError("I3D requires at least eight frames")
        features = torch.ones(1, 2048, 1, 1, 1, device=clip.device)
        return self.projection(features)


def test_i3d_uniformly_repeats_short_clips_to_its_eight_frame_minimum() -> None:
    features = _features()
    model = _MinimumTemporalI3D()
    adapter = features.I3DKineticsFeatures(device="cpu")
    adapter._model = model
    adapter._projection = model.projection

    result = adapter(torch.zeros(3, 3, 8, 8))

    assert model.clip is not None
    assert model.clip.shape == (1, 3, 8, 224, 224)
    assert result.shape == (1, 2048)


@pytest.mark.parametrize(
    ("adapter_name", "dependency"),
    [("I3DKineticsFeatures", "pytorchvideo"), ("OpenCLIPFeatures", "open_clip")],
)
def test_optional_feature_dependencies_raise_clear_runtime_errors(
    adapter_name: str,
    dependency: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features()

    def unavailable(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(features.importlib, "import_module", unavailable)
    adapter = getattr(features, adapter_name)(device="cpu")

    with pytest.raises(RuntimeError, match=rf"{dependency}.*required"):
        if adapter_name == "OpenCLIPFeatures":
            adapter.encode_text("a prompt")
        else:
            adapter(torch.zeros(2, 3, 8, 8))
