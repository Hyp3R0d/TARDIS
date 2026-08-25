from __future__ import annotations

from pathlib import Path

from tardis.experiments.queue import _benchmark_command, parse_args


def test_queue_forwards_temporal_diagnostic_options(tmp_path: Path) -> None:
    cache = tmp_path / "flow-cache"
    args = parse_args(
        [
            "--output-root",
            str(tmp_path / "runs"),
            "--protocol",
            "source50_diagnostics",
            "--methods",
            "tardis",
            "--datasets",
            "dataverse",
            "--seeds",
            "3407",
            "--temporal-diagnostics",
            "--flow-cache-root",
            str(cache),
            "--flow-batch-size",
            "3",
        ]
    )

    command = _benchmark_command(
        "torchrun",
        args,
        method="tardis",
        dataset="dataverse",
        seed=3407,
        output=tmp_path / "unit",
    )

    assert "--temporal-diagnostics" in command
    assert command[command.index("--flow-cache-root") + 1] == str(cache)
    assert command[command.index("--flow-batch-size") + 1] == "3"


def test_queue_forwards_validation_manifest_options(tmp_path: Path) -> None:
    manifest = tmp_path / "validation50.json"
    args = parse_args(
        [
            "--output-root",
            str(tmp_path / "runs"),
            "--protocol",
            "source50",
            "--methods",
            "tardis",
            "--datasets",
            "seedance",
            "--seeds",
            "3407",
            "--data-split",
            "validation",
            "--record-ids-file",
            str(manifest),
            "--metrics",
            "primary",
        ]
    )

    command = _benchmark_command(
        "torchrun",
        args,
        method="tardis",
        dataset="seedance",
        seed=3407,
        output=tmp_path / "unit",
    )

    assert command[command.index("--data-split") + 1] == "validation"
    assert command[command.index("--record-ids-file") + 1] == str(manifest)
    assert command[command.index("--metrics") + 1] == "primary"
