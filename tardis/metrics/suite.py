"""Serializable six-metric streaming suite for one video pair at a time."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

import torch

from tardis.metrics.frechet import FIDMetric, FVDMetric
from tardis.metrics.paired import (
    CLIPScoreMetric,
    LPIPSMetric,
    SSIMMetric,
    TemporalConsistencyMetric,
)

SuiteResult = dict[str, dict[str, float]]


class MetricSuite:
    """Stream six metrics without retaining generated or reference videos."""

    def __init__(
        self,
        *,
        tc: TemporalConsistencyMetric | None = None,
        lpips: LPIPSMetric | None = None,
        fvd: FVDMetric | None = None,
        fid: FIDMetric | None = None,
        clipscore: CLIPScoreMetric | None = None,
        ssim: SSIMMetric | None = None,
    ) -> None:
        self.tc = TemporalConsistencyMetric() if tc is None else tc
        self.lpips = LPIPSMetric() if lpips is None else lpips
        self.fvd = FVDMetric() if fvd is None else fvd
        self.fid = FIDMetric() if fid is None else fid
        self.clipscore = CLIPScoreMetric() if clipscore is None else clipscore
        self.ssim = SSIMMetric() if ssim is None else ssim

    @torch.no_grad()
    def update(self, generated: torch.Tensor, reference: torch.Tensor, prompt: str) -> None:
        self.tc.update(generated, reference)
        self.lpips.update(generated, reference)
        self.fvd.update(generated, reference)
        self.fid.update(generated, reference)
        self.clipscore.update(generated, prompt)
        self.ssim.update(generated, reference)

    def compute(self) -> SuiteResult:
        fid = self.fid.compute()
        fvd = self.fvd.compute()
        return {
            "macro": {
                "tc": self.tc.compute("macro"),
                "lpips": self.lpips.compute("macro"),
                "fvd": fvd,
                "fid": fid,
                "clipscore": self.clipscore.compute("macro"),
                "ssim": self.ssim.compute("macro"),
            },
            "micro": {
                "tc": self.tc.compute("micro"),
                "lpips": self.lpips.compute("micro"),
                "fvd": fvd,
                "fid": fid,
                "clipscore": self.clipscore.compute("micro"),
                "ssim": self.ssim.compute("micro"),
            },
        }

    @property
    def provenance_ids(self) -> dict[str, str]:
        return {
            "tc": self.tc.provenance_id,
            "lpips": self.lpips.provenance_id,
            "fvd": self.fvd.provenance_id,
            "fid": self.fid.provenance_id,
            "clipscore": self.clipscore.provenance_id,
            "ssim": self.ssim.provenance_id,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "provenance_ids": self.provenance_ids,
            "metrics": {
                "tc": self.tc.state_dict(),
                "lpips": self.lpips.state_dict(),
                "fvd": self.fvd.state_dict(),
                "fid": self.fid.state_dict(),
                "clipscore": self.clipscore.state_dict(),
                "ssim": self.ssim.state_dict(),
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"version", "provenance_ids", "metrics"} or state["version"] != 1:
            raise ValueError("metric suite state has an incompatible schema or version")
        provenance = state["provenance_ids"]
        metrics = state["metrics"]
        if not isinstance(provenance, Mapping) or dict(provenance) != self.provenance_ids:
            raise ValueError("metric suite state provenance does not match")
        if not isinstance(metrics, Mapping) or set(metrics) != set(self.provenance_ids):
            raise ValueError("metric suite state metric entries are incompatible")
        self.tc.load_state_dict(_metric_state(metrics, "tc"))
        self.lpips.load_state_dict(_metric_state(metrics, "lpips"))
        self.fvd.load_state_dict(_metric_state(metrics, "fvd"))
        self.fid.load_state_dict(_metric_state(metrics, "fid"))
        self.clipscore.load_state_dict(_metric_state(metrics, "clipscore"))
        self.ssim.load_state_dict(_metric_state(metrics, "ssim"))

    def merge(self, other: Self) -> None:
        if self.provenance_ids != other.provenance_ids:
            raise ValueError("metric suites must have matching provenance to merge")
        self.tc.merge(other.tc)
        self.lpips.merge(other.lpips)
        self.fvd.merge(other.fvd)
        self.fid.merge(other.fid)
        self.clipscore.merge(other.clipscore)
        self.ssim.merge(other.ssim)

    def all_reduce(self) -> None:
        self.tc.all_reduce()
        self.lpips.all_reduce()
        self.fvd.all_reduce()
        self.fid.all_reduce()
        self.clipscore.all_reduce()
        self.ssim.all_reduce()

    def reset(self) -> None:
        self.tc.reset()
        self.lpips.reset()
        self.fvd.reset()
        self.fid.reset()
        self.clipscore.reset()
        self.ssim.reset()


StreamingMetricSuite = MetricSuite


def _metric_state(metrics: Mapping[object, object], name: str) -> Mapping[str, object]:
    value = metrics[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"metric suite {name} state must be a mapping")
    return cast(Mapping[str, object], value)
