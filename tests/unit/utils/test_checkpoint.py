from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tardis.utils.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    atomic_torch_save,
    checkpoint_sha256,
    find_latest_checkpoint,
    load_checkpoint,
)


@pytest.mark.unit
def test_atomic_checkpoint_round_trip_and_digest(tmp_path: Path) -> None:
    destination = tmp_path / "best.pt"
    payload = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "epoch": 7, "weights": {"x": 1}}

    atomic_torch_save(payload, destination)

    assert load_checkpoint(destination)["epoch"] == 7
    assert len(checkpoint_sha256(destination)) == 64


@pytest.mark.unit
def test_failed_atomic_save_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "latest.pt"
    original = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "epoch": 1}
    atomic_torch_save(original, destination)

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="synthetic save failure"):
        atomic_torch_save({"schema_version": CHECKPOINT_SCHEMA_VERSION, "epoch": 2}, destination)

    assert load_checkpoint(destination)["epoch"] == 1
    assert not list(tmp_path.glob(".latest.pt.*.tmp"))


@pytest.mark.unit
def test_latest_checkpoint_uses_parsed_timestamp_not_lexical_noise(tmp_path: Path) -> None:
    older = tmp_path / "20260801_235959_000000_a" / "best.pt"
    newer = tmp_path / "20260802_000001_000000_z" / "best.pt"
    malformed = tmp_path / "zzzz" / "best.pt"
    for path in (older, newer, malformed):
        path.parent.mkdir()
        atomic_torch_save({"schema_version": CHECKPOINT_SCHEMA_VERSION}, path)

    assert find_latest_checkpoint(tmp_path, "best.pt") == newer


@pytest.mark.unit
def test_checkpoint_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    destination = tmp_path / "bad.pt"
    atomic_torch_save({"schema_version": CHECKPOINT_SCHEMA_VERSION + 1}, destination)

    with pytest.raises(CheckpointError, match="schema"):
        load_checkpoint(destination)
