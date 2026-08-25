"""Atomic JSON manifests and collision-safe run directories."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    output_dir: Path
    checkpoint_dir: Path


def _timestamp(now: datetime) -> str:
    aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y%m%d_%H%M%S_%f")


def create_run_paths(
    output_root: Path, checkpoint_root: Path, kind: str, now: datetime | None = None
) -> RunPaths:
    """Create matching timestamped paths, adding a suffix on collisions."""
    base = _timestamp(now or datetime.now(UTC))
    output_parent = Path(output_root) / kind
    checkpoint_parent = Path(checkpoint_root)
    output_parent.mkdir(parents=True, exist_ok=True)
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    for collision in range(10_000):
        run_id = base if collision == 0 else f"{base}_{collision:02d}"
        output_dir = output_parent / run_id
        checkpoint_dir = checkpoint_parent / run_id
        if output_dir.exists() or checkpoint_dir.exists():
            continue
        output_dir.mkdir()
        try:
            checkpoint_dir.mkdir()
        except BaseException:
            output_dir.rmdir()
            raise
        return RunPaths(run_id, output_dir, checkpoint_dir)
    raise RuntimeError(f"unable to allocate a unique run directory for timestamp {base}")


def create_output_run_dir(
    output_root: Path,
    kind: str,
    now: datetime | None = None,
) -> Path:
    """Create one timestamped output directory without allocating checkpoint storage."""

    base = _timestamp(now or datetime.now(UTC))
    output_parent = Path(output_root) / kind
    output_parent.mkdir(parents=True, exist_ok=True)
    for collision in range(10_000):
        run_id = base if collision == 0 else f"{base}_{collision:02d}"
        output_dir = output_parent / run_id
        try:
            output_dir.mkdir()
        except FileExistsError:
            continue
        return output_dir
    raise RuntimeError(f"unable to allocate a unique run directory for timestamp {base}")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def write_json_manifest(destination: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a UTF-8 JSON manifest in the destination directory."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
