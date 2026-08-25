from __future__ import annotations

import importlib.util
import os
from pathlib import Path

EXPERIMENTS = Path(__file__).parents[3] / "TARDIS_SOTA" / "work" / "experiments"
SPEC = importlib.util.spec_from_file_location(
    "watch_validation_candidate",
    EXPERIMENTS / "watch_validation_candidate.py",
)
assert SPEC is not None and SPEC.loader is not None
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def test_explicit_training_pid_does_not_require_run_id_in_command_line() -> None:
    pid = os.getpid()

    assert watcher._train_pids("run-id-not-present", [pid]) == [pid]


def test_explicit_training_pid_ignores_dead_process() -> None:
    assert watcher._train_pids("run-id-not-present", [999999999]) == []
