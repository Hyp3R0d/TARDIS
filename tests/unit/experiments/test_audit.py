from __future__ import annotations

import json
from pathlib import Path

from tardis.experiments.audit import audit


def _write_queue(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "unit_count": count,
                "results": [{"status": "completed"} for _ in range(count)],
            }
        ),
        encoding="utf-8",
    )


def test_audit_requires_source_prompt_baseline_supplement(tmp_path: Path) -> None:
    _write_queue(tmp_path / "TARDIS_PAPER_EXPERIMENTS/queue_manifest.json", 60)
    _write_queue(tmp_path / "TARDIS_SOURCE_EXPERIMENTS/queue_manifest.json", 105)
    _write_queue(tmp_path / "TARDIS_ABLATION_EXPERIMENTS/queue_manifest.json", 11)
    _write_queue(tmp_path / "TARDIS_SOURCE_ABLATION_EXPERIMENTS/queue_manifest.json", 11)

    missing = audit(tmp_path)

    assert missing["status"] == "incomplete"
    assert missing["queues"]["source_prompt_baselines"]["status"] == "missing"

    _write_queue(
        tmp_path
        / "TARDIS_SOURCE_EXPERIMENTS/supplement/queue_manifest.json",
        45,
    )

    complete = audit(tmp_path)

    assert complete["status"] == "incomplete"
    assert complete["queues"]["source_prompt_baselines"]["valid"] is True

    _write_queue(
        tmp_path / "TARDIS_SOURCE_EXPERIMENTS/diagnostics/queue_manifest.json",
        150,
    )

    complete = audit(tmp_path)

    assert complete["status"] == "complete"
    assert complete["queues"]["source_diagnostics"]["valid"] is True
