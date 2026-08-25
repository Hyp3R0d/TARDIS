"""Audit experiment queue coverage and completed artifact integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED = {
    "prompt": 60,
    "source": 105,
    "source_prompt_baselines": 45,
    "source_diagnostics": 150,
    "ablation": 11,
    "source_ablation": 11,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/TARDIS"))
    return parser


def audit(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    locations = {
        "prompt": root / "TARDIS_PAPER_EXPERIMENTS/queue_manifest.json",
        "source": root / "TARDIS_SOURCE_EXPERIMENTS/queue_manifest.json",
        "source_prompt_baselines": (
            root / "TARDIS_SOURCE_EXPERIMENTS/supplement/queue_manifest.json"
        ),
        "source_diagnostics": (
            root / "TARDIS_SOURCE_EXPERIMENTS/diagnostics/queue_manifest.json"
        ),
        "ablation": root / "TARDIS_ABLATION_EXPERIMENTS/queue_manifest.json",
        "source_ablation": root / "TARDIS_SOURCE_ABLATION_EXPERIMENTS/queue_manifest.json",
    }
    report: dict[str, Any] = {"status": "incomplete", "queues": {}}
    all_complete = True
    for name, path in locations.items():
        if not path.is_file():
            report["queues"][name] = {"status": "missing", "completed": 0}
            all_complete = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        completed = sum(
            item.get("status") in {"completed", "already_completed"} for item in results
        )
        failed = sum(item.get("status") == "failed" for item in results)
        expected = EXPECTED[name]
        valid = (
            payload.get("status") == "completed"
            and int(payload.get("unit_count", -1)) == expected
            and completed == expected
            and failed == 0
        )
        report["queues"][name] = {
            "status": payload.get("status"),
            "expected": expected,
            "completed": completed,
            "failed": failed,
            "valid": valid,
        }
        all_complete = all_complete and valid
    report["status"] = "complete" if all_complete else "incomplete"
    return report


def main() -> None:
    report = audit(build_parser().parse_args().root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
