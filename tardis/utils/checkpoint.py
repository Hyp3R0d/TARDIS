"""Atomic, schema-versioned checkpoint persistence."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^(?P<timestamp>\d{8}_\d{6}_\d{6})(?:[_-].*)?$")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint violates the persisted schema contract."""


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Write a checkpoint through a same-filesystem temporary and atomic rename."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a trusted local checkpoint and enforce the current schema version."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a mapping")
    version = payload.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema {version!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    return payload


def checkpoint_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the content digest used by run manifests and apply sidecars."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest_checkpoint(root: Path, filename: str = "best.pt") -> Path | None:
    """Find the newest checkpoint using parsed UTC run timestamps."""
    candidates: list[tuple[datetime, str, Path]] = []
    for path in Path(root).glob(f"*/{filename}"):
        match = _RUN_ID.fullmatch(path.parent.name)
        if match is None:
            continue
        timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d_%H%M%S_%f")
        candidates.append((timestamp, path.parent.name, path))
    return max(candidates)[2] if candidates else None
