"""Deterministic process, rank, and worker seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

_RANK_STRIDE = 100_003
_WORKER_STRIDE = 1_009
_UINT32_MODULUS = 2**32


def effective_seed(base_seed: int, rank: int = 0, worker_id: int = 0) -> int:
    """Derive a stable uint32 seed for a distributed worker."""
    return (base_seed + rank * _RANK_STRIDE + worker_id * _WORKER_STRIDE) % _UINT32_MODULUS


def seed_everything(
    base_seed: int, rank: int = 0, worker_id: int = 0, *, deterministic: bool = False
) -> int:
    """Seed Python, NumPy, and PyTorch and return the effective seed."""
    seed = effective_seed(base_seed, rank, worker_id)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return seed


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create an explicitly seeded generator without changing global RNG state."""
    return torch.Generator(device=device).manual_seed(seed)
