"""Select a source-conditioned TARDIS innovation strength on validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

TC_TARGET = 0.10
LPIPS_TARGET = 0.60
TC_WEIGHT = 0.625
LPIPS_WEIGHT = 0.375


def weighted_score(*, tc: float, lpips: float) -> float:
    """Return the locked, target-normalized lower-is-better primary score."""

    if not math.isfinite(tc) or not math.isfinite(lpips) or tc < 0 or lpips < 0:
        raise ValueError("TC and LPIPS must be finite and non-negative")
    return TC_WEIGHT * (tc / TC_TARGET) + LPIPS_WEIGHT * (lpips / LPIPS_TARGET)


def select_candidate(candidates: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    """Apply the pre-registered primary score and deterministic tie breakers."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    required = {"source_strength", "tc", "lpips"}
    if any(not required.issubset(candidate) for candidate in candidates):
        raise ValueError(f"candidate rows must contain {sorted(required)}")
    return min(
        candidates,
        key=lambda candidate: (
            weighted_score(tc=float(candidate["tc"]), lpips=float(candidate["lpips"])),
            float(candidate["tc"]),
            float(candidate["lpips"]),
            float(candidate["source_strength"]),
        ),
    )


def _strength_tag(strength: float) -> str:
    return f"s{int(round(strength * 100)):03d}"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not load JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON payload must be an object: {path}")
    return payload


def _read_record_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        record_id = str(item.get("record_id", ""))
        if not record_id or record_id in identifiers:
            raise RuntimeError(f"invalid benchmark ledger record IDs in {path}")
        identifiers.add(record_id)
    return identifiers


def _candidate_run(
    *,
    root: Path,
    dataset: str,
    strength: float,
    seed: int,
    expected_record_ids: set[str],
) -> dict[str, float | int | str]:
    output = root / f"runs_{_strength_tag(strength)}" / "tardis" / dataset / f"seed_{seed}"
    metrics_path = output / "metrics.json"
    payload = _load_json(metrics_path)
    coverage = payload.get("coverage")
    settings = payload.get("settings")
    metric_values = payload.get("metrics")
    if not isinstance(coverage, Mapping) or not isinstance(settings, Mapping):
        raise RuntimeError(f"incomplete benchmark schema: {metrics_path}")
    if payload.get("status") != "completed" or coverage.get("failed") != 0:
        raise RuntimeError(f"benchmark unit is not complete: {metrics_path}")
    if coverage.get("completed") != coverage.get("expected") or coverage.get("expected") != len(
        expected_record_ids
    ):
        raise RuntimeError(
            f"benchmark coverage does not match the frozen validation subset: {metrics_path}"
        )
    if (
        settings.get("dataset") != dataset
        or settings.get("data_split") != "validation"
        or settings.get("protocol") != "source50"
        or settings.get("metric_mode") != "primary"
        or not math.isclose(float(settings.get("source_strength", -1)), strength, abs_tol=1e-8)
    ):
        raise RuntimeError(
            f"benchmark settings do not match the validation selection protocol: {metrics_path}"
        )
    if not isinstance(metric_values, Mapping) or not isinstance(
        metric_values.get("macro"), Mapping
    ):
        raise RuntimeError(f"benchmark primary metrics are missing: {metrics_path}")
    macro = metric_values["macro"]
    tc = float(macro["tc"])
    lpips = float(macro["lpips"])
    weighted_score(tc=tc, lpips=lpips)
    if _read_record_ids(output / "per_video.jsonl") != expected_record_ids:
        raise RuntimeError(
            f"benchmark ledger does not match the frozen validation subset: {output}"
        )
    return {"seed": seed, "tc": tc, "lpips": lpips, "output": str(output)}


def analyze(root: Path, protocol_path: Path) -> dict[str, Any]:
    """Aggregate fully completed validation units and lock one strength."""

    protocol = _load_json(protocol_path)
    if protocol.get("development_split") != "validation":
        raise ValueError("source-strength selection must use a validation split")
    dataset = str(protocol["experiment"]).split()[0].lower()
    if dataset not in {"dataverse", "seedance", "openvid"}:
        raise ValueError("protocol experiment name must start with a supported dataset")
    manifest_path = protocol_path.parent / str(protocol["development_record_manifest"])
    manifest = _load_json(manifest_path)
    record_ids = {str(item["record_id"]) for item in manifest.get("records", [])}
    if len(record_ids) != int(protocol["development_records"]):
        raise RuntimeError("validation manifest record count does not match the protocol")
    strengths = [float(value) for value in protocol["candidate_source_strengths"]]
    seeds = [int(value) for value in protocol["seeds"]]
    summaries: list[dict[str, Any]] = []
    for strength in strengths:
        per_seed = [
            _candidate_run(
                root=root,
                dataset=dataset,
                strength=strength,
                seed=seed,
                expected_record_ids=record_ids,
            )
            for seed in seeds
        ]
        tc_values = [float(item["tc"]) for item in per_seed]
        lpips_values = [float(item["lpips"]) for item in per_seed]
        tc = fmean(tc_values)
        lpips = fmean(lpips_values)
        summaries.append(
            {
                "source_strength": strength,
                "tc": tc,
                "tc_std": pstdev(tc_values),
                "lpips": lpips,
                "lpips_std": pstdev(lpips_values),
                "score": weighted_score(tc=tc, lpips=lpips),
                "per_seed": per_seed,
            }
        )
    selected = dict(select_candidate(summaries))
    return {
        "schema_version": 1,
        "status": "completed",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "validation_manifest_path": str(manifest_path.resolve()),
        "validation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset": dataset,
        "selection_metric": {
            "tc_target": TC_TARGET,
            "lpips_target": LPIPS_TARGET,
            "tc_weight": TC_WEIGHT,
            "lpips_weight": LPIPS_WEIGHT,
        },
        "candidates": summaries,
        "selected": selected,
        "test_set_consulted": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"selection output already exists: {output}")
    payload = analyze(args.root.expanduser().resolve(), args.protocol.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    main()
