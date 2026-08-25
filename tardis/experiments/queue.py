"""Sequential, resumable execution queue for the locked paper benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from tardis.experiments.benchmark import DATASETS, METHODS
from tardis.utils.manifest import write_json_manifest

DEFAULT_METHODS = (
    "sd_turbo_independent",
    "animatediff_lightning",
    "text2video_zero",
    "tardis",
)
DEFAULT_SEEDS = (3407, 3413, 3433, 3469, 3491)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/TARDIS/TARDIS_PAPER_EXPERIMENTS/main"),
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--protocol",
        choices=("paper50", "source50", "source50_diagnostics"),
        default="paper50",
    )
    parser.add_argument("--data-split", choices=("test", "validation"), default="test")
    parser.add_argument("--record-ids-file", type=Path, default=None)
    parser.add_argument("--metrics", choices=("primary", "full"), default="full")
    parser.add_argument("--source-strength", type=float, default=0.45)
    parser.add_argument(
        "--temporal-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--flow-cache-root",
        type=Path,
        default=Path("/root/autodl-tmp/TARDIS/metric_cache/raft_small_backward"),
    )
    parser.add_argument("--flow-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--showcase-count", type=int, default=0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--refresh-package",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    unknown_methods = sorted(set(args.methods) - set(METHODS))
    unknown_datasets = sorted(set(args.datasets) - set(DATASETS))
    if unknown_methods:
        raise ValueError(f"unsupported queue methods: {unknown_methods}")
    if unknown_datasets:
        raise ValueError(f"unsupported queue datasets: {unknown_datasets}")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("queue seeds must be non-negative")
    if args.num_workers < 0 or args.checkpoint_every <= 0 or args.showcase_count < 0:
        raise ValueError("queue worker and checkpoint settings are invalid")
    if args.flow_batch_size <= 0:
        raise ValueError("queue flow batch size must be positive")
    if not 0 <= args.source_strength <= 1:
        raise ValueError("source strength must be in [0, 1]")
    if args.protocol == "source50_diagnostics" and not args.temporal_diagnostics:
        raise ValueError("source50_diagnostics requires temporal diagnostics")
    if args.temporal_diagnostics and args.protocol != "source50_diagnostics":
        raise ValueError("temporal diagnostics require source50_diagnostics protocol")
    return args


def _benchmark_command(
    torchrun: str,
    args: argparse.Namespace,
    *,
    method: str,
    dataset: str,
    seed: int,
    output: Path,
) -> list[str]:
    command = [
        torchrun,
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "tardis.experiments.benchmark",
        "--method",
        method,
        "--dataset",
        dataset,
        "--protocol",
        str(args.protocol),
        "--data-split",
        str(args.data_split),
        "--metrics",
        str(args.metrics),
        "--seed",
        str(seed),
        "--source-strength",
        str(args.source_strength),
        "--num-workers",
        str(args.num_workers),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--showcase-count",
        str(args.showcase_count),
        "--output",
        str(output),
    ]
    if args.record_ids_file is not None:
        command.extend(("--record-ids-file", str(args.record_ids_file)))
    if args.temporal_diagnostics:
        command.extend(
            (
                "--temporal-diagnostics",
                "--flow-cache-root",
                str(args.flow_cache_root),
                "--flow-batch-size",
                str(args.flow_batch_size),
            )
        )
    return command


def run_queue(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    queue_manifest = output_root.parent / "queue_manifest.json"
    units = [
        (method, dataset, int(seed))
        for method in args.methods
        for dataset in args.datasets
        for seed in args.seeds
    ]
    results: list[dict[str, object]] = []
    started = time.time()
    _write_queue_manifest(
        queue_manifest,
        status="running",
        started=started,
        units=units,
        results=results,
    )
    environment = _offline_environment()
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun is not available on PATH")

    for index, (method, dataset, seed) in enumerate(units, start=1):
        output = output_root / method / dataset / f"seed_{seed}"
        metrics_path = output / "metrics.json"
        if _completed(metrics_path):
            result = _unit_result(method, dataset, seed, output, "already_completed", 0)
            results.append(result)
            _write_queue_manifest(
                queue_manifest,
                status="running",
                started=started,
                units=units,
                results=results,
            )
            continue

        print(
            f"\n[{index}/{len(units)}] method={method} dataset={dataset} seed={seed}",
            flush=True,
        )
        command = _benchmark_command(
            torchrun,
            args,
            method=method,
            dataset=dataset,
            seed=seed,
            output=output,
        )
        unit_started = time.time()
        completed = subprocess.run(command, cwd="/home/TARDIS", env=environment, check=False)
        status = "completed" if completed.returncode == 0 and _completed(metrics_path) else "failed"
        result = _unit_result(
            method,
            dataset,
            seed,
            output,
            status,
            completed.returncode,
            elapsed_seconds=time.time() - unit_started,
        )
        results.append(result)
        _write_queue_manifest(
            queue_manifest,
            status="running",
            started=started,
            units=units,
            results=results,
        )
        if args.refresh_package:
            _refresh_package(environment)
        if status == "failed" and args.stop_on_error:
            break

    failures = [item for item in results if item["status"] == "failed"]
    final_status = (
        "completed"
        if len(results) == len(units) and not failures
        else "completed_with_errors"
    )
    _write_queue_manifest(
        queue_manifest,
        status=final_status,
        started=started,
        units=units,
        results=results,
    )
    if args.refresh_package:
        _refresh_package(environment)
    return 1 if failures else 0


def _offline_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HOME": "/root/autodl-tmp/TARDIS/cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "TORCH_HOME": "/root/autodl-tmp/TARDIS/cache/torch",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _completed(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    coverage = payload.get("coverage", {})
    return (
        payload.get("status") == "completed"
        and coverage.get("completed") == coverage.get("expected")
        and coverage.get("failed") == 0
    )


def _unit_result(
    method: str,
    dataset: str,
    seed: int,
    output: Path,
    status: str,
    returncode: int,
    *,
    elapsed_seconds: float = 0.0,
) -> dict[str, object]:
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "output": str(output),
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
    }


def _write_queue_manifest(
    path: Path,
    *,
    status: str,
    started: float,
    units: list[tuple[str, str, int]],
    results: list[dict[str, object]],
) -> None:
    write_json_manifest(
        path,
        {
            "schema_version": 1,
            "status": status,
            "started_unix": started,
            "updated_unix": time.time(),
            "unit_count": len(units),
            "completed_result_count": len(results),
            "units": [
                {"method": method, "dataset": dataset, "seed": seed}
                for method, dataset, seed in units
            ],
            "results": results,
        },
    )


def _refresh_package(environment: dict[str, str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tardis.experiments.package", "--refresh"],
        cwd="/home/TARDIS",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        print("warning: package refresh failed; queue will continue", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_queue(parse_args(argv)))


if __name__ == "__main__":
    main()
