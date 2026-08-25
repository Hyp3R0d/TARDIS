from __future__ import annotations

from pathlib import Path

import torch

from tardis.experiments.flow_cache import BackwardFlowCache


class CountingEstimator:
    provenance_id = "test/counting-flow-v1"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full(
            (video.shape[0] - 1, 2, video.shape[-2], video.shape[-1]),
            float(self.calls),
        )


def test_backward_flow_cache_reuses_matching_record(tmp_path: Path) -> None:
    estimator = CountingEstimator()
    cache = BackwardFlowCache(tmp_path, estimator=estimator)
    video = torch.zeros(4, 3, 8, 8)

    first = cache.get_or_compute("dataverse", "video/one.mp4", video)
    second = cache.get_or_compute("dataverse", "video/one.mp4", video)

    assert estimator.calls == 1
    assert torch.equal(first, second)
    assert first.dtype == torch.float32
    assert len(list(tmp_path.rglob("*.pt"))) == 1


def test_backward_flow_cache_separates_video_geometry(tmp_path: Path) -> None:
    estimator = CountingEstimator()
    cache = BackwardFlowCache(tmp_path, estimator=estimator)

    cache.get_or_compute("dataverse", "same-id", torch.zeros(4, 3, 8, 8))
    cache.get_or_compute("dataverse", "same-id", torch.zeros(5, 3, 8, 8))

    assert estimator.calls == 2
    assert len(list(tmp_path.rglob("*.pt"))) == 2
