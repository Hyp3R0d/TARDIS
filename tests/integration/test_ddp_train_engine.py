from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tardis.models.tardis import TARDISTrainingBatch
from tardis.training.curriculum import CurriculumSchedule
from tardis.training.engine import TrainEngine, TrainEngineOptions
from tardis.training.objective import TARDISObjective
from tests.helpers.tardis_model import build_tiny_tardis


class TinyPerceptualMetric:
    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (generated - reference).abs().mean(dim=(1, 2, 3), keepdim=True)


@pytest.mark.integration
def test_ddp_executes_tardis_objective_but_checkpoints_unwrapped_temporal_model(
    tmp_path: Path,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    rendezvous = (tmp_path / "gloo-init").resolve()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        model = build_tiny_tardis().model
        distributed_model = DistributedDataParallel(model, find_unused_parameters=True)
        engine = TrainEngine(
            distributed_model,
            objective=TARDISObjective(
                perceptual_metric=TinyPerceptualMetric(),
                temporal_levels=1,
            ),
            options=TrainEngineOptions(
                learning_rate=1.0e-3,
                weight_decay=0,
                gradient_accumulation_steps=1,
                gradient_clip_norm=1,
                warmup_steps=0,
                total_optimizer_steps=6,
                precision="fp32",
            ),
            curriculum=CurriculumSchedule(durations=(1, 1, 1, 1, 1, 1)),
            generator=torch.Generator().manual_seed(13),
        )
        training_batch = TARDISTrainingBatch(
            prompts=["a moving crystal"],
            video=torch.randn(1, 3, 3, 16, 16),
        )

        result = engine.train_microbatch(training_batch)
        checkpoint = engine.state_dict(epoch=1, next_batch_index=0)
        model_state = cast(dict[str, torch.Tensor], checkpoint["model"])

        assert result.optimizer_updated
        assert engine.unwrapped_model is model
        assert model_state
        assert all(not name.startswith(("module.", "priors.")) for name in model_state)
    finally:
        dist.destroy_process_group()
