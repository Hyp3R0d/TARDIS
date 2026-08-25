"""Minimal torchrun-aware distributed runtime context."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import ParamSpec, TypeVar

import torch
import torch.distributed as dist

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls, device_type: str | None = None) -> DistributedContext:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        resolved_type = device_type or ("cuda" if torch.cuda.is_available() else "cpu")
        device = (
            torch.device(resolved_type, local_rank)
            if resolved_type == "cuda"
            else torch.device(resolved_type)
        )
        return cls(rank=rank, local_rank=local_rank, world_size=world_size, device=device)

    def initialize(self, backend: str | None = None, timeout_seconds: int = 1800) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.world_size > 1 and not dist.is_initialized():
            resolved_backend = backend or ("nccl" if self.device.type == "cuda" else "gloo")
            dist.init_process_group(
                backend=resolved_backend,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def rank_zero_only(
    context: DistributedContext,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R | None]]:
    """Return a decorator that suppresses side effects on non-zero ranks."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R | None]:
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
            if not context.is_main:
                return None
            return function(*args, **kwargs)

        return wrapped

    return decorate


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across ranks, returning the input unchanged in one process."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value
