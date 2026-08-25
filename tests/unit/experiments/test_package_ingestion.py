from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tardis.experiments.package import (
    _benchmark_frame_level_frame,
    _benchmark_per_video_frame,
    _benchmark_run_frame,
    _load_benchmark_runs,
    _load_source_selection,
    _motion_bin_frame,
    _paired_statistics_frame,
    _preferred_source_protocol,
    _release_metadata,
    _temporal_lag_frame,
    _validate_selected_source_runs,
    _write_manifest,
    _write_scripts,
)
from tardis.experiments.queue import _completed


def _write_completed_run(root: Path) -> Path:
    run = root / "pilots" / "sd_turbo_independent" / "dataverse" / "seed_3407"
    run.mkdir(parents=True)
    record = {
        "experiment_id": "exp01_sd_turbo_independent_dataverse_seed3407",
        "record_id": "video-1",
        "prompt": "a test prompt",
        "seed": 99,
        "generation_seconds": 1.5,
        "tc": 0.2,
        "lpips": 0.4,
        "tc_per_transition": [0.1, 0.3],
        "lpips_per_frame": [0.3, 0.4, 0.5],
        "status": "completed",
    }
    (run / "per_video.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    metrics = {
        "status": "completed",
        "settings": {
            "method": "sd_turbo_independent",
            "dataset": "dataverse",
            "protocol": "pilot",
            "metric_mode": "full",
            "global_seed": 3407,
        },
        "coverage": {"expected": 1, "completed": 1, "failed": 0},
        "metrics": {
            "macro": {
                "tc": 0.2,
                "lpips": 0.4,
                "fvd": 10.0,
                "fid": 20.0,
                "clipscore": 0.3,
                "ssim": 0.6,
            }
        },
        "latency": {
            "mean_video_seconds": 1.5,
            "p50_video_seconds": 1.5,
            "p95_video_seconds": 1.5,
            "mean_frame_milliseconds": 500.0,
        },
        "resources": {
            "peak_reserved_mb": 1024.0,
            "mean_gpu_utilization_percent": 80.0,
        },
        "elapsed_wall_seconds": 3.0,
    }
    metrics_path = run / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return metrics_path


def test_package_discovers_and_flattens_completed_benchmark(tmp_path: Path) -> None:
    metrics_path = _write_completed_run(tmp_path)

    runs = _load_benchmark_runs(tmp_path)
    run_frame = _benchmark_run_frame(runs)
    per_video = _benchmark_per_video_frame(runs)
    frame_level = _benchmark_frame_level_frame(runs)

    assert len(runs) == 1
    assert run_frame.iloc[0]["FVD"] == 10.0
    assert per_video.iloc[0]["TC"] == 0.2
    assert frame_level["TC"].isna().sum() == 1
    assert frame_level["LPIPS"].tolist() == [0.3, 0.4, 0.5]
    assert _completed(metrics_path)


def test_queue_rejects_incomplete_metrics(tmp_path: Path) -> None:
    metrics_path = _write_completed_run(tmp_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["coverage"]["completed"] = 0
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not _completed(metrics_path)


def test_package_flattens_extended_temporal_diagnostics(tmp_path: Path) -> None:
    metrics_path = _write_completed_run(tmp_path)
    run_root = metrics_path.parent
    record = json.loads((run_root / "per_video.jsonl").read_text(encoding="utf-8"))
    record.update(
        {
            "flow_warp_error": 0.12,
            "tlpips": 0.34,
            "flicker_rate": 0.5,
            "drift_slope": -0.01,
            "motion_magnitude": 1.25,
            "tc_by_lag": {"1": 0.2, "2": 0.3},
            "flow_warp_error_per_transition": [0.1, 0.14],
            "tlpips_per_transition": [0.3, 0.38],
            "motion_magnitude_per_transition": [1.0, 1.5],
            "brightness_per_frame": [0.2, 0.3, 0.4],
            "brightness_delta_per_transition": [0.1, 0.1],
            "flicker_flags": [False, True],
        }
    )
    (run_root / "per_video.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    runs = _load_benchmark_runs(tmp_path)
    per_video = _benchmark_per_video_frame(runs)
    frame_level = _benchmark_frame_level_frame(runs)

    assert per_video.iloc[0]["flow_warp_error"] == 0.12
    assert per_video.iloc[0]["tLPIPS"] == 0.34
    assert per_video.iloc[0]["flicker_rate"] == 0.5
    assert frame_level["flow_warp_err"].dropna().tolist() == [0.1, 0.14]
    assert frame_level["tLPIPS"].dropna().tolist() == [0.3, 0.38]
    assert frame_level["motion_magnitude"].dropna().tolist() == [1.0, 1.5]


def test_paired_statistics_requires_complete_five_seed_coverage() -> None:
    rows = []
    for method, offset in (("tardis", 0.0), ("baseline", 0.2)):
        for seed in range(5):
            for index in range(50):
                rows.append(
                    {
                        "dataset": "dataverse",
                        "method": method,
                        "protocol": "paper50",
                        "global_seed": seed,
                        "video_id": f"video-{index}",
                        "TC": 0.1 + offset + index * 1.0e-4,
                        "LPIPS": 0.4 + offset + index * 1.0e-4,
                    }
                )

    result = _paired_statistics_frame(pd.DataFrame(rows))

    assert set(result["metric"]) == {"TC", "LPIPS"}
    assert result["tardis_better"].all()
    assert (result["records"] == 50).all()
    assert (result["p_value_holm"] < 0.05).all()


def test_package_prefers_only_complete_source_diagnostic_protocol() -> None:
    methods = (
        "streamdiffusion_img2img",
        "rerender_flow",
        "tokenflow_core",
        "vid2vid_zero_core",
        "controlvideo_canny",
        "stablevideo_propagation",
        "animatediff_lightning",
        "text2video_zero",
        "sd_turbo_independent",
        "tardis",
    )
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "protocol": "source50_diagnostics",
        }
        for dataset in ("dataverse", "seedance", "openvid")
        for method in methods
        for seed in (3407, 3413, 3433, 3469, 3491)
    ]
    legacy = pd.DataFrame(
        [{"dataset": "dataverse", "method": "tardis", "seed": 3407, "protocol": "source50"}]
    )

    assert _preferred_source_protocol(pd.concat((legacy, pd.DataFrame(rows[:-1])))) == "source50"
    assert _preferred_source_protocol(pd.concat((legacy, pd.DataFrame(rows)))) == (
        "source50_diagnostics"
    )


def test_package_builds_motion_bins_and_lag_tables() -> None:
    frame_level = pd.DataFrame(
        {
            "dataset": ["dataverse"] * 4,
            "method": ["tardis"] * 4,
            "protocol": ["source50_diagnostics"] * 4,
            "motion_magnitude": [0.2, 1.0, 3.0, 6.0],
            "TC": [0.01, 0.02, 0.03, 0.04],
            "flow_warp_err": [0.1, 0.2, 0.3, 0.4],
            "tLPIPS": [0.2, 0.3, 0.4, 0.5],
        }
    )
    per_video = pd.DataFrame(
        {
            "dataset": ["dataverse", "dataverse"],
            "method": ["tardis", "tardis"],
            "protocol": ["source50_diagnostics", "source50_diagnostics"],
            "TC_lag_1": [0.1, 0.2],
            "TC_lag_2": [0.2, 0.4],
            "TC_lag_4": [0.4, 0.8],
            "TC_lag_8": [0.8, 1.6],
        }
    )

    motion = _motion_bin_frame(frame_level, "source50_diagnostics")
    lag = _temporal_lag_frame(per_video, "source50_diagnostics")

    assert motion["motion_bin"].tolist() == ["0-0.5", "0.5-2", "2-5", ">5"]
    assert motion["observations"].tolist() == [1, 1, 1, 1]
    assert lag["lag"].tolist() == [1, 2, 4, 8]
    assert lag["TC_mean"].tolist() == pytest.approx([0.15, 0.3, 0.6, 1.2])


def test_package_script_readme_documents_runnable_source_protocol(tmp_path: Path) -> None:
    output = tmp_path / "package"
    (output / "05_scripts").mkdir(parents=True)

    _write_scripts(output, repo=Path("/home/TARDIS"))

    readme = (output / "05_scripts/README_scripts.md").read_text(encoding="utf-8")
    assert "`streamdiffusion_img2img`" in readme
    assert "`rerender_flow`" in readme
    assert "audited core-mechanism reproduction" in readme
    assert "run_source_prompt_baselines.sh" in readme
    assert "run_source_ablations.sh" in readme
    assert "stay `N/A`" not in readme


def test_release_metadata_becomes_final_only_after_complete_audit() -> None:
    assert _release_metadata({"status": "incomplete"}) == {
        "version": "1.0-initial",
        "status": "initial_in_progress",
    }
    assert _release_metadata({"status": "complete"}) == {
        "version": "1.0",
        "status": "final_complete",
    }
    assert _release_metadata({"status": "complete"}, version="1.1") == {
        "version": "1.1",
        "status": "final_complete",
    }


def test_manifest_uses_release_metadata_from_version_file(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("1.0\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("delivery\n", encoding="utf-8")

    _write_manifest(tmp_path)

    payload = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["version"] == "1.0"
    assert payload["status"] == "final_complete"


def test_manifest_accepts_a_final_incremental_release_version(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("1.1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("delivery\n", encoding="utf-8")

    _write_manifest(tmp_path)

    payload = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["version"] == "1.1"
    assert payload["status"] == "final_complete"


def test_package_loads_validation_only_source_selection(tmp_path: Path) -> None:
    validation_manifest = tmp_path / "validation50.json"
    validation_manifest.write_text('{"records": []}\n', encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset": "seedance",
                "test_set_consulted": False,
                "validation_manifest_path": str(validation_manifest),
                "selected": {"source_strength": 0.30},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selection = _load_source_selection(selection_path)

    assert selection["dataset"] == "seedance"
    assert selection["source_strength"] == pytest.approx(0.30)
    assert selection["test_set_consulted"] is False


def test_package_requires_matching_selected_source_test_runs() -> None:
    selection = {"dataset": "seedance", "source_strength": 0.30}
    runs = [
        {
            "payload": {
                "settings": {
                    "method": "tardis",
                    "dataset": "seedance",
                    "protocol": "source50",
                    "data_split": "test",
                    "source_strength": 0.30,
                    "global_seed": seed,
                },
                "coverage": {"expected": 50, "completed": 50},
            }
        }
        for seed in (3407, 3413, 3433, 3469, 3491)
    ]

    _validate_selected_source_runs(runs, selection)
