"""Reusable source-video optical-flow cache for paper diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import torch

from tardis.metrics.base import validate_video
from tardis.utils.checkpoint import atomic_torch_save


class FlowEstimator(Protocol):
    provenance_id: str

    def __call__(self, video: torch.Tensor) -> torch.Tensor: ...


class BackwardFlowCache:
    def __init__(self, root: Path, *, estimator: FlowEstimator) -> None:
        if not estimator.provenance_id.strip():
            raise ValueError("flow estimator provenance must be non-empty")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.estimator = estimator

    def get_or_compute(
        self,
        dataset: str,
        record_id: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        validate_video(reference, name="reference", min_frames=2)
        path, metadata = self._entry(dataset, record_id, reference)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict) and payload.get("metadata") == metadata:
                flow = payload.get("backward_flow")
                if isinstance(flow, torch.Tensor):
                    return _validate_flow(flow.to(torch.float32), reference)

        flow = _validate_flow(self.estimator(reference).detach().cpu(), reference)
        atomic_torch_save(
            {
                "schema_version": 1,
                "metadata": metadata,
                "backward_flow": flow.to(torch.float16),
            },
            path,
        )
        return flow.to(torch.float32)

    def _entry(
        self,
        dataset: str,
        record_id: str,
        reference: torch.Tensor,
    ) -> tuple[Path, dict[str, object]]:
        if not re.fullmatch(r"[a-z0-9_-]+", dataset):
            raise ValueError("dataset cache namespace must be lowercase alphanumeric")
        if not record_id.strip():
            raise ValueError("record ID must be non-empty")
        metadata: dict[str, object] = {
            "schema_version": 1,
            "dataset": dataset,
            "record_id": record_id,
            "video_shape": list(reference.shape),
            "estimator": self.estimator.provenance_id,
        }
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self.root / dataset / f"{digest}.pt", metadata


class TorchvisionRAFTSmallBackwardFlow:
    """Backward flow from target frame t+1 to source frame t using RAFT-Small."""

    provenance_id = "torchvision/raft_small:Raft_Small_Weights.C_T_V2:backward-flow"

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda:0",
        batch_size: int = 2,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("RAFT batch size must be positive")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model: torch.nn.Module | None = None

    @torch.inference_mode()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        validate_video(video, name="video", min_frames=2)
        height, width = video.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError("RAFT input height and width must be divisible by eight")
        if float(video.min().item()) < -1.001 or float(video.max().item()) > 1.001:
            raise ValueError("RAFT source video must be normalized to [-1, 1]")
        model = self._load()
        values: list[torch.Tensor] = []
        for start in range(0, video.shape[0] - 1, self.batch_size):
            stop = min(start + self.batch_size, video.shape[0] - 1)
            target = video[start + 1 : stop + 1].to(
                device=self.device,
                dtype=torch.float32,
            )
            source = video[start:stop].to(device=self.device, dtype=torch.float32)
            predictions = model(target.contiguous(), source.contiguous())
            if not isinstance(predictions, list) or not predictions:
                raise RuntimeError("torchvision RAFT returned no flow predictions")
            values.append(predictions[-1].detach().to(device="cpu", dtype=torch.float32))
        return torch.cat(values, dim=0)

    def _load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        model = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=True)
        model.to(self.device)
        model.requires_grad_(False)
        model.eval()
        self._model = model
        return model


def _validate_flow(flow: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    expected = (
        reference.shape[0] - 1,
        2,
        reference.shape[-2],
        reference.shape[-1],
    )
    if tuple(flow.shape) != expected:
        raise ValueError(f"flow estimator returned {tuple(flow.shape)}; expected {expected}")
    if not bool(torch.isfinite(flow).all().item()):
        raise ValueError("flow estimator returned non-finite values")
    return flow.to(device="cpu", dtype=torch.float32).contiguous()
