from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = ROOT / "TARDIS_SOTA" / "work" / "experiments"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_prompt_only_validation",
    EXPERIMENTS / "evaluate_prompt_only_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_frozen_split_evaluator_can_disable_ema(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_prompt_only_validation.py",
            "--dataset",
            "dataverse",
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--train-manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "validation.json"),
            "--no-use-ema",
        ],
    )

    assert evaluator.parse_args().use_ema is False


def test_frozen_split_evaluator_restores_sampler_trajectory_alignment() -> None:
    assert "sampler_trajectory_alignment" in evaluator.ARCHITECTURE_FIELDS
