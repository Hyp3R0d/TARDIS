"""Assemble the auditable RTVD-TC paper data package from experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_NAME = "RTVD-TC-DataPackage-v1.0"
DATASETS = ("dataverse", "seedance", "openvid")
METRICS = ("tc", "lpips", "fvd", "fid", "clipscore", "ssim")
FIELD_DICTIONARY = (
    ("experiment_id", "实验编号", "str"),
    ("dataset", "数据集名称", "str"),
    ("method", "方法名", "str"),
    ("source_strength", "source-conditioned 残差创新强度", "float [0,1]"),
    ("variant", "消融变体", "str"),
    ("video_id", "视频记录 ID", "str"),
    ("frame_idx", "帧序号，从 0 起", "int"),
    ("seed", "确定性样本种子", "uint64"),
    ("prompt_id", "提示词条目 ID", "str"),
    ("LPIPS", "AlexNet LPIPS，越低越好", "float"),
    ("TC", "赛题帧差 L1，越低越好", "float"),
    ("flow_warp_error", "RAFT 反向光流补偿后的生成帧 MSE，越低越好", "float"),
    ("tLPIPS", "生成相邻帧经源运动对齐后的 AlexNet LPIPS，越低越好", "float"),
    ("flicker_rate", "亮度相邻变化大于 0.1 的 transition 比例，越低越好", "float"),
    ("drift_slope", "逐 transition TC 的线性回归斜率，绝对值越小越好", "float"),
    ("motion_magnitude", "RAFT 反向光流平均模长", "px"),
    ("FVD", "I3D Frechet Video Distance，越低越好", "float"),
    ("FID", "Inception Frechet Distance，越低越好", "float"),
    ("CLIP_score", "OpenCLIP 图文余弦相似度，越高越好", "float"),
    ("SSIM", "结构相似度，越高越好", "float"),
    ("latency_total_ms", "单段视频端到端生成时延", "ms"),
    ("gpu_mem_mb", "峰值 CUDA reserved memory", "MiB"),
    ("status", "completed/failed/planned", "str"),
    ("provenance", "数据来源与限制", "str"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RTVD-TC experiment data package")
    parser.add_argument("--repo", type=Path, default=Path("/home/TARDIS"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/TARDIS") / PACKAGE_NAME,
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/home/TARDIS/TARDIS_PAPER_EXPERIMENTS"),
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("/home/TARDIS/TARDIS_ABLATION_EXPERIMENTS"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/TARDIS/TARDIS_SOURCE_EXPERIMENTS"),
    )
    parser.add_argument(
        "--source-ablation-root",
        type=Path,
        default=Path("/home/TARDIS/TARDIS_SOURCE_ABLATION_EXPERIMENTS"),
    )
    parser.add_argument(
        "--source-selection",
        type=Path,
        default=None,
        help="completed validation-only source-strength selection to include in the package",
    )
    parser.add_argument(
        "--release-version",
        default="1.0",
        help="semantic version written to VERSION and MANIFEST.json",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def initialize_package(args: argparse.Namespace) -> Path:
    repo = args.repo.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.refresh:
        raise FileExistsError(f"package already exists; pass --refresh to update it: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for relative in (
        "00_docs",
        "01_configs/prompts",
        "01_configs/video_splits",
        "02_raw_data",
        "03_figures",
        "04_tables",
        "05_scripts",
        "06_logs",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)

    snapshot = _load_tardis_snapshot(repo)
    experiment_root = args.experiment_root.expanduser().resolve()
    benchmark_runs = _load_benchmark_runs(experiment_root)
    ablation_root = args.ablation_root.expanduser().resolve()
    ablation_runs = _load_benchmark_runs(ablation_root)
    source_root = args.source_root.expanduser().resolve()
    source_runs = _load_benchmark_runs(source_root)
    source_ablation_root = args.source_ablation_root.expanduser().resolve()
    source_ablation_runs = _load_benchmark_runs(source_ablation_root)
    source_selection = (
        _load_source_selection(args.source_selection)
        if args.source_selection is not None
        else None
    )
    if source_selection is not None:
        _validate_selected_source_runs(source_runs, source_selection)
    selected = _selected_records(snapshot["ledgers"], count=50)
    from tardis.experiments.audit import audit

    release = _release_metadata(audit(repo), version=str(args.release_version))
    _write_text(output / "VERSION", f"{release['version']}\n")
    _write_docs(
        output,
        repo=repo,
        snapshot=snapshot,
        benchmark_runs=benchmark_runs,
        source_runs=source_runs,
        source_ablation_runs=source_ablation_runs,
        source_selection=source_selection,
    )
    _write_configs(
        output,
        repo=repo,
        selected=selected,
        source_selection=source_selection,
    )
    _write_raw_workbooks(
        output,
        snapshot=snapshot,
        benchmark_runs=benchmark_runs,
        ablation_runs=ablation_runs,
        source_runs=source_runs,
        source_ablation_runs=source_ablation_runs,
        source_selection=source_selection,
    )
    _write_tables(
        output,
        snapshot=snapshot,
        benchmark_runs=benchmark_runs,
        source_runs=source_runs,
        source_ablation_runs=source_ablation_runs,
        source_selection=source_selection,
    )
    _write_figures(
        output,
        snapshot=snapshot,
        benchmark_runs=benchmark_runs,
        source_runs=source_runs,
        source_ablation_runs=source_ablation_runs,
        source_selection=source_selection,
    )
    _write_scripts(output, repo=repo)
    _copy_benchmark_logs(
        output,
        experiment_root=experiment_root,
        runs=benchmark_runs,
        namespace="main",
        reset=True,
    )
    _copy_benchmark_logs(
        output,
        experiment_root=ablation_root,
        runs=ablation_runs,
        namespace="ablation",
        reset=False,
    )
    _copy_benchmark_logs(
        output,
        experiment_root=source_ablation_root,
        runs=source_ablation_runs,
        namespace="source_ablation",
        reset=False,
    )
    _copy_benchmark_logs(
        output,
        experiment_root=source_root,
        runs=source_runs,
        namespace="source",
        reset=False,
    )
    _write_text(
        output / "06_logs/package_initialization.log",
        json.dumps(
            {
                "timestamp_utc": _utc_now(),
                "status": release["status"],
                "version": release["version"],
                "source": str(repo / "TARDIS_SOTA"),
                "records_imported": sum(len(items) for items in snapshot["ledgers"].values()),
                "benchmark_runs_imported": len(benchmark_runs),
                "ablation_runs_imported": len(ablation_runs),
                "source_runs_imported": len(source_runs),
                "source_ablation_runs_imported": len(source_ablation_runs),
                "source_selection": (
                    None
                    if source_selection is None
                    else {
                        "dataset": source_selection["dataset"],
                        "source_strength": source_selection["source_strength"],
                        "sha256": source_selection["sha256"],
                    }
                ),
            },
            indent=2,
        )
        + "\n",
    )
    _write_manifest(output)
    return output


def _load_tardis_snapshot(repo: Path) -> dict[str, Any]:
    root = repo / "TARDIS_SOTA"
    delivery = json.loads((root / "delivery_manifest.json").read_text(encoding="utf-8"))
    metrics: dict[str, dict[str, float]] = {}
    latency: dict[str, dict[str, float]] = {}
    resources: dict[str, dict[str, float]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        infer = (root / "infer_outputs" / dataset).resolve(strict=True)
        with (infer / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        metrics[dataset] = {name: float(row[name]) for name in METRICS}
        latency[dataset] = json.loads((infer / "latency.json").read_text(encoding="utf-8"))
        resources[dataset] = json.loads((infer / "resources.json").read_text(encoding="utf-8"))
        ledgers[dataset] = _read_jsonl(infer / "per_video_details.jsonl")
        if len(ledgers[dataset]) != 512:
            raise RuntimeError(f"{dataset} formal infer ledger must contain exactly 512 records")
    return {
        "delivery": delivery,
        "metrics": metrics,
        "latency": latency,
        "resources": resources,
        "ledgers": ledgers,
    }


def _load_benchmark_runs(experiment_root: Path) -> list[dict[str, Any]]:
    if not experiment_root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(experiment_root.rglob("metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        settings = payload.get("settings")
        coverage = payload.get("coverage")
        metrics = payload.get("metrics")
        if not isinstance(settings, dict) or not isinstance(coverage, dict):
            raise ValueError(f"benchmark artifact has no settings or coverage: {metrics_path}")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("macro"), dict):
            raise ValueError(f"benchmark artifact has no macro metrics: {metrics_path}")
        expected = int(coverage.get("expected", -1))
        completed = int(coverage.get("completed", -1))
        failed = int(coverage.get("failed", -1))
        if expected <= 0 or completed != expected or failed != 0:
            raise ValueError(f"completed benchmark has invalid coverage: {metrics_path}")
        per_video_path = metrics_path.parent / "per_video.jsonl"
        records = _read_jsonl(per_video_path)
        if len(records) != expected:
            raise ValueError(f"benchmark ledger coverage mismatch: {per_video_path}")
        runs.append(
            {
                "metrics_path": metrics_path,
                "run_root": metrics_path.parent,
                "payload": payload,
                "records": records,
            }
        )
    return runs


def _load_source_selection(path: Path) -> dict[str, Any]:
    """Validate a completed validation-only source-strength selection artifact."""

    selection_path = path.expanduser().resolve(strict=True)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source selection must be a JSON object: {selection_path}")
    if payload.get("status") != "completed" or payload.get("test_set_consulted") is not False:
        raise ValueError("source selection must be completed without consulting the test set")
    dataset = str(payload.get("dataset", ""))
    if dataset not in DATASETS:
        raise ValueError(f"source selection has an unsupported dataset: {dataset!r}")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("source selection is missing the selected configuration")
    strength = float(selected.get("source_strength", -1))
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("source selection has an invalid selected source strength")
    manifest_path = Path(str(payload.get("validation_manifest_path", ""))).expanduser()
    manifest_path = manifest_path.resolve(strict=True)
    return {
        "path": selection_path,
        "sha256": _sha256(selection_path),
        "payload": payload,
        "dataset": dataset,
        "source_strength": strength,
        "test_set_consulted": False,
        "validation_manifest_path": manifest_path,
    }


def _validate_selected_source_runs(
    source_runs: list[dict[str, Any]], source_selection: dict[str, Any]
) -> None:
    """Ensure a selection artifact is paired with exactly one frozen test evaluation set."""

    dataset = str(source_selection["dataset"])
    strength = float(source_selection["source_strength"])
    matches = []
    for run in source_runs:
        settings = run["payload"]["settings"]
        if (
            settings.get("method") == "tardis"
            and settings.get("dataset") == dataset
            and settings.get("protocol") == "source50"
            and settings.get("data_split") == "test"
            and math.isclose(float(settings.get("source_strength", -1)), strength, abs_tol=1e-8)
        ):
            matches.append(run)
    expected_seeds = {3407, 3413, 3433, 3469, 3491}
    observed_seeds = {int(run["payload"]["settings"]["global_seed"]) for run in matches}
    if observed_seeds != expected_seeds or len(matches) != len(expected_seeds):
        raise ValueError(
            "source selection requires exactly five matching TARDIS source50 test runs "
            f"for {dataset} at source_strength={strength:.2f}"
        )
    for run in matches:
        coverage = run["payload"]["coverage"]
        if coverage.get("expected") != 50 or coverage.get("completed") != 50:
            raise ValueError("selected source test runs must cover the complete 50-record protocol")


def _write_docs(
    output: Path,
    *,
    repo: Path,
    snapshot: dict[str, Any],
    benchmark_runs: list[dict[str, Any]],
    source_runs: list[dict[str, Any]],
    source_ablation_runs: list[dict[str, Any]],
    source_selection: dict[str, Any] | None,
) -> None:
    release = _read_release_metadata(output)
    benchmark_run_frame = _benchmark_run_frame(benchmark_runs)
    benchmark_per_video = _benchmark_per_video_frame(benchmark_runs)
    prompt_summary = _protocol_summary_frame(benchmark_run_frame, "paper50")
    prompt_statistics = _paired_statistics_frame(benchmark_per_video)
    prompt_claims = int(prompt_statistics["tardis_better"].sum())
    source_run_frame = _benchmark_run_frame(source_runs)
    source_protocol = "source50" if source_selection is not None else _preferred_source_protocol(
        source_run_frame
    )
    source_summary = _protocol_summary_frame(source_run_frame, source_protocol)
    benchmark_records = sum(len(run["records"]) for run in benchmark_runs)
    source_records = sum(len(run["records"]) for run in source_runs)
    readme = f"""# RTVD-TC Data Package v{release['version']}

本目录是 TARDIS 论文实验的唯一数据入口。当前版本为 **{release['status']}**：已经冻结并导入
TARDIS 在 DataVerse、Seedance、OpenVid 三个完整 test split 上的正式结果、1,536 条推理
ledger、六项聚合指标、延时和资源记录，并导入 {len(benchmark_runs)} 个 prompt-only 运行
（{benchmark_records} 条视频级记录）与 {len(source_runs)} 个 source-conditioned 运行
（{source_records} 条视频级记录）。
source50 消融另有 {len(source_ablation_runs)} 个单元，归档在 `exp04`。

## 快速校验

```bash
cd {output}
python 05_scripts/verify_package.py
```

校验器逐文件计算 SHA256，并拒绝缺失、篡改或未登记文件。`MANIFEST.json` 本身不参与自身
哈希。

## 当前真实数据

| 数据集 | Test 视频 | TC ↓ | LPIPS ↓ | FVD ↓ | FID ↓ | CLIP ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
{_markdown_metric_rows(snapshot['metrics'])}

上述结果来自三份 validation-only 选择的 EMA checkpoint；test 未参与选权重。权重 SHA256、
模型配置和原始输出位置见 `01_configs/methods_config.yaml` 与 `TARDIS_SOTA/delivery_manifest.json`
的快照说明。

## 当前同协议 benchmark 实测

{_summary_markdown_frame(prompt_summary)}

`pilot` 只证明适配器和统一指标链路可运行，不进入正式主表；`paper50` 才进入方法比较与显著性
检验。每次运行的 `metrics.json`、`per_video.jsonl` 和 `run_manifest.json` 均已归档到
`06_logs/benchmark_runs/`。

当前 prompt-only 主表在三个数据集、三个 benchmark、TC/LPIPS 两项主指标上共有
`{prompt_claims}/{len(prompt_statistics)}` 项通过配对 bootstrap CI、单侧 Wilcoxon 和 Holm 校正。
该结论只适用于 TC/LPIPS；FID、CLIPScore 等次指标必须按表中实测值分别讨论。

## Source-conditioned 独立协议

{_summary_markdown_frame(source_summary)}

`{source_protocol}` 与 prompt-only 主表严格分开：source video 同时作为生成条件和指标参考，
{_source_strength_note(source_selection)} 非官方原仓直接执行的方法必须在 manifest 中标记
`implementation_scope=audited core-mechanism reproduction`，不能写成官方数值。

## 协议边界

补全后的《实验方案》沿用 DAVIS/DeepFashion/WebVid、RTX 4090 和 source-conditioned 编辑协议，
而当前正式工程使用 DataVerse/Seedance/OpenVid、RTX 4080 SUPER 和 prompt-only 生成。因此：

1. 本包以当前正式三数据集 locked test split 为主协议；不把名称不同的数据集伪装成 DAVIS 等。
2. Text2Video-Zero、AnimateDiff-Lightning、SD-Turbo 逐帧可在 prompt-only 协议直接实测。
3. Rerender-A-Video、TokenFlow、vid2vid-zero、ControlVideo、StableVideo 依赖 source video；若未建立
   等价 prompt-only 适配，主表记为 `N/A (protocol incompatible)`，不得填造数值。
   它们的 source50 核心机制复现单独进入 `exp07_generalization.xlsx`，不与 prompt-only 主表混合。
4. 论文报告值仅在 `provenance=paper_reported_dagger` 时以 `†` 展示，不与当前协议做显著性检验。
5. “TARDIS 全面领先”只在同记录、同 seed、同分辨率、同帧数、同指标实现的实测结果支持时成立。

## XLSX 使用

- `02_raw_data/exp01_main_comparison.xlsx`：当前 TARDIS 主表、1,536 条生成 ledger、benchmark 状态。
- `02_raw_data/exp02_latency.xlsx`：视频级生成时延和聚合资源数据。
- `exp03`：逐帧 TC/LPIPS 时间演化；`exp04`：A0-A10 消融；`exp05`：5-seed 稳定性。
- `exp06`：参数、延时、显存和收益开销；`exp07`：独立 source-conditioned 协议。
- `exp08`：真实用户研究模板，未执行条目明确标记 `planned`，空值不是零。
- 每个工作簿均含 `_meta` 和 `summary`；字段字典见 `00_docs/usage_guide_cn.md`。

## 已导出图表

- `04_tables/table01` 至 `table08`：主比较、延时、配对统计、source 比较、seed 稳定性、效率、
  时序诊断和 source A0-A10 消融（XLSX 与 LaTeX 双格式）。
- `03_figures/fig01` 至 `fig15`：只导出已有测量直接支持的图；没有受控分辨率扫描、人工场景标签、
  跨基座或真实用户研究的数据时，对应图不会生成，不能视为零值或负结果。

## 重跑实验

```bash
cd {repo}
torchrun --standalone --nproc_per_node=1 -m tardis.experiments.benchmark \
  --method sd_turbo_independent --dataset dataverse \
  --output {output}/06_logs/sd_turbo_independent/dataverse
```

完整命令和方法状态见 `05_scripts/README_scripts.md`。实验完成后重新运行：

```bash
python -m tardis.experiments.package --refresh
```

## 写论文前的硬门槛

- 主表 benchmark 必须覆盖相同的 50 个 test record × 5 seeds，且失败数为 0。
- TC/LPIPS 必须保留 per-video 配对原始值，bootstrap CI 和 Wilcoxon 只从这些原始值计算。
- 任何 `planned`、`N/A` 或 `†` 行不能写成“已实测领先”。
- `exp08_user_study.xlsx` 只能写真实匿名用户输入，禁止合成偏好票。
"""
    _write_text(output / "README.md", readme)
    _write_text(output / "00_docs/usage_guide_cn.md", _usage_guide_cn())
    _write_text(output / "00_docs/usage_guide_en.md", _usage_guide_en())
    _write_text(output / "00_docs/method_descriptions.md", _method_descriptions())
    _write_text(output / "00_docs/hardware_environment.md", _hardware_environment(repo))
    _write_text(
        output / "00_docs/changelog.md",
        f"# Changelog\n\n## {release['version']} ({_utc_now()})\n\n"
        "- Imported three complete TARDIS test summaries and 1,536 inference ledger rows.\n"
        f"- Imported {len(benchmark_runs)} completed benchmark runs with raw ledgers.\n"
        f"- Imported {len(source_runs)} completed source-conditioned runs.\n"
        "- Locked the current DataVerse/Seedance/OpenVid prompt-only protocol.\n"
        "- Added explicit protocol-compatibility status for all requested benchmarks.\n",
    )


def _source_strength_note(source_selection: dict[str, Any] | None) -> str:
    if source_selection is None:
        return "所有 source-conditioned 方法固定 `source_strength=0.45`。"
    return (
        "比较器保持冻结的 `source_strength=0.45`；"
        f"TARDIS 的 {source_selection['dataset']} 配置使用 "
        f"`source_strength={source_selection['source_strength']:.2f}`，"
        "该值仅由独立 validation-50 的锁定规则选择，选择过程未读取 test 集，"
        "完整选择证据和 validation 清单分别见 "
        "`01_configs/source_strength_selection.json` 与 "
        "`01_configs/video_splits/*_validation_selection50.json`。"
    )


def _write_configs(
    output: Path,
    *,
    repo: Path,
    selected: dict[str, list[dict[str, Any]]],
    source_selection: dict[str, Any] | None,
) -> None:
    _write_text(
        output / "01_configs/global_config.yaml",
        """schema_version: 1
protocol: tardis-prompt-only-locked-test-v1
datasets: [dataverse, seedance, openvid]
resolution: [512, 512]
frames: 16
fps_metadata: 30
split_seed: 3407
validation_size: 256
test_size: 512
paper_subset_size: 50
seeds: [3407, 3413, 3433, 3469, 3491]
precision: bf16
hardware: NVIDIA GeForce RTX 4080 SUPER
primary_metrics: [TC, LPIPS]
secondary_metrics: [FVD, FID, CLIP_score, SSIM]
""",
    )
    delivery = json.loads(
        (repo / "TARDIS_SOTA/delivery_manifest.json").read_text(encoding="utf-8")
    )
    if source_selection is not None:
        shutil.copy2(
            source_selection["path"],
            output / "01_configs/source_strength_selection.json",
        )
        shutil.copy2(
            source_selection["validation_manifest_path"],
            output
            / "01_configs/video_splits"
            / f"{source_selection['dataset']}_validation_selection50.json",
        )
    method_lines = [
        "schema_version: 1",
        "methods:",
        "  tardis:",
        "    status: measured_full_test",
        "    role: ours",
        "    weights:",
    ]
    for dataset in DATASETS:
        item = delivery["datasets"][dataset]
        method_lines.extend(
            (
                f"      {dataset}:",
                f"        path: {repo / 'TARDIS_SOTA' / item['checkpoint']}",
                f"        sha256: {item['checkpoint_sha256']}",
            )
        )
    statuses = {
        "streamdiffusion": "source50_core_reproduction",
        "rerender_a_video": "source50_core_reproduction",
        "tokenflow": "source50_core_reproduction",
        "vid2vid_zero": "source50_core_reproduction",
        "text2video_zero": "runnable_local",
        "controlvideo": "source50_core_reproduction",
        "stablevideo": "source50_core_reproduction",
        "animatediff_lcm": "runnable_as_animatediff_lightning_2step",
        "sd_turbo_independent": "runnable_local",
    }
    for method, status in statuses.items():
        method_lines.extend((f"  {method}:", f"    status: {status}"))
    _write_text(output / "01_configs/methods_config.yaml", "\n".join(method_lines) + "\n")
    _write_json(output / "01_configs/seeds.json", {"seeds": [3407, 3413, 3433, 3469, 3491]})
    for dataset, records in selected.items():
        _write_json(
            output / f"01_configs/video_splits/{dataset}_test.json",
            {
                "dataset": dataset,
                "split": "test",
                "selection": "lowest sha256(split_seed, record_id), method-independent",
                "records": [
                    {
                        "record_id": item["record_id"],
                        "prompt": item["prompt"],
                        "locked_seed": item["seed"],
                    }
                    for item in records
                ],
            },
        )
    _write_prompt_sets(output, selected)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _write_text(
        output / "01_configs/environment_lock.yaml",
        "generated_at_utc: " + _utc_now() + "\npackages: |\n" + _indent(freeze, 2),
    )


def _write_prompt_sets(
    output: Path,
    selected: dict[str, list[dict[str, Any]]],
) -> None:
    simple: list[dict[str, str]] = []
    sentence: list[dict[str, str]] = []
    style: list[dict[str, str]] = []
    for dataset, records in selected.items():
        for index, item in enumerate(records[:20]):
            prompt = str(item["prompt"]).strip()
            prompt_id = f"{dataset}_{index:02d}"
            words = prompt.replace("\n", " ").split()
            simple.append(
                {
                    "prompt_id": prompt_id,
                    "dataset": dataset,
                    "prompt": " ".join(words[:12]),
                }
            )
            sentence.append({"prompt_id": prompt_id, "dataset": dataset, "prompt": prompt})
            style.append(
                {
                    "prompt_id": prompt_id,
                    "dataset": dataset,
                    "prompt": prompt + ", cinematic lighting, stable temporal detail",
                }
            )
    _write_json(output / "01_configs/prompts/prompt_set_a_simple.json", simple)
    _write_json(output / "01_configs/prompts/prompt_set_b_sentence.json", sentence)
    _write_json(output / "01_configs/prompts/prompt_set_c_style.json", style)


def _write_raw_workbooks(
    output: Path,
    *,
    snapshot: dict[str, Any],
    benchmark_runs: list[dict[str, Any]],
    ablation_runs: list[dict[str, Any]],
    source_runs: list[dict[str, Any]],
    source_ablation_runs: list[dict[str, Any]],
    source_selection: dict[str, Any] | None,
) -> None:
    summary = _metric_summary_frame(snapshot)
    benchmark_run_frame = _benchmark_run_frame(benchmark_runs)
    benchmark_per_video = _benchmark_per_video_frame(benchmark_runs)
    benchmark_frame_level = _benchmark_frame_level_frame(benchmark_runs)
    paper50_summary = _paper50_summary_frame(benchmark_run_frame)
    paired_statistics = _paired_statistics_frame(benchmark_per_video)
    source_run_frame = _benchmark_run_frame(source_runs)
    source_per_video = _benchmark_per_video_frame(source_runs)
    source_protocol = "source50" if source_selection is not None else _preferred_source_protocol(
        source_run_frame
    )
    source_summary = _protocol_summary_frame(source_run_frame, source_protocol)
    source_statistics = _paired_statistics_frame(source_per_video, protocol=source_protocol)
    source_diagnostic_summary = _group_metric_summary(
        _protocol_frame(source_per_video, source_protocol),
        group_columns=("dataset", "method"),
        metrics=(
            "flow_warp_error",
            "tLPIPS",
            "flicker_rate",
            "drift_slope",
            "motion_magnitude",
            "TC_lag_1",
            "TC_lag_2",
            "TC_lag_4",
            "TC_lag_8",
        ),
    )
    ledger_rows: list[dict[str, Any]] = []
    for dataset, records in snapshot["ledgers"].items():
        resource = snapshot["resources"][dataset]
        for item in records:
            ledger_rows.append(
                {
                    "experiment_id": "formal_tardis_test_20260816",
                    "dataset": dataset,
                    "method": "TARDIS",
                    "variant": "A10",
                    "video_id": item["record_id"],
                    "seed": item["seed"],
                    "prompt_id": _prompt_id(dataset, str(item["record_id"])),
                    "prompt": item["prompt"],
                    "LPIPS": pd.NA,
                    "TC": pd.NA,
                    "FVD": pd.NA,
                    "FID": pd.NA,
                    "CLIP_score": pd.NA,
                    "SSIM": pd.NA,
                    "latency_total_ms": float(item["generation_seconds"]) * 1000,
                    "gpu_mem_mb": float(resource.get("peak_reserved_mb", 0.0)),
                    "status": item["status"],
                    "provenance": "formal inference ledger; per-video metric values not retained",
                }
            )
    per_video = pd.concat(
        (pd.DataFrame(ledger_rows), benchmark_per_video),
        ignore_index=True,
        sort=False,
    )
    measured_methods = {
        str(item["payload"]["settings"]["method"])
        for item in benchmark_runs
    }
    methods_status = pd.DataFrame(
        [
            ("TARDIS", "measured_full_test", "prompt-only"),
            ("StreamDiffusion", "adapter_required", "streaming image diffusion"),
            ("Rerender-A-Video", "N/A", "requires source video"),
            ("TokenFlow", "N/A", "requires source video"),
            ("vid2vid-zero", "N/A", "requires source video"),
            (
                "Text2Video-Zero",
                "measured_or_queued" if "text2video_zero" in measured_methods else "queued",
                "prompt-only compatible",
            ),
            ("ControlVideo", "N/A", "requires source control"),
            ("StableVideo", "N/A", "requires source video"),
            (
                "AnimateDiff-LCM",
                "measured_or_queued"
                if "animatediff_lightning" in measured_methods
                else "queued",
                "prompt-only compatible",
            ),
            (
                "SD-Turbo independent",
                "measured_or_queued"
                if "sd_turbo_independent" in measured_methods
                else "queued",
                "prompt-only compatible",
            ),
        ],
        columns=("method", "status", "protocol_note"),
    )
    empty_frame = pd.DataFrame(
        columns=(
            "experiment_id",
            "dataset",
            "method",
            "video_id",
            "frame_idx",
            "seed",
            "LPIPS",
            "TC",
            "flow_warp_err",
            "tLPIPS",
            "CLIP_score",
        )
    )
    _write_workbook(
        output / "02_raw_data/exp01_main_comparison.xlsx",
        description="P0 main comparison; initial TARDIS snapshot plus benchmark queue status",
        sheets={
            "per_video": per_video,
            "frame_level_dataverse": _dataset_frame(
                benchmark_frame_level, "dataverse", empty_frame
            ),
            "frame_level_seedance": _dataset_frame(
                benchmark_frame_level, "seedance", empty_frame
            ),
            "frame_level_openvid": _dataset_frame(
                benchmark_frame_level, "openvid", empty_frame
            ),
            "measured_runs": benchmark_run_frame,
            "pilot_runs": _protocol_frame(benchmark_run_frame, "pilot"),
            "paper50_runs": _protocol_frame(benchmark_run_frame, "paper50"),
            "paired_statistics": paired_statistics,
            "source50_runs": source_run_frame,
            "source50_per_video": source_per_video,
            "source50_summary": source_summary,
            "source50_statistics": source_statistics,
            "source_diagnostic_summary": source_diagnostic_summary,
            "seed_runs": pd.concat(
                (summary.assign(seed=3407, protocol="formal512_legacy"), benchmark_run_frame),
                ignore_index=True,
                sort=False,
            ),
            "fid_fvd": pd.concat(
                (
                    summary[["dataset", "method", "FVD", "FID"]],
                    benchmark_run_frame[["dataset", "method", "FVD", "FID"]],
                ),
                ignore_index=True,
            ),
            "method_status": methods_status,
            "summary": pd.concat((summary, paper50_summary), ignore_index=True, sort=False),
        },
    )

    latency_rows = per_video[
        [
            "experiment_id",
            "dataset",
            "method",
            "video_id",
            "seed",
            "latency_total_ms",
            "gpu_mem_mb",
            "status",
            "provenance",
        ]
    ].copy()
    latency_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "method": "TARDIS",
                "video_count": int(snapshot["latency"][dataset]["successful_video_count"]),
                "mean_video_ms": float(snapshot["latency"][dataset]["mean_generation_seconds"])
                * 1000,
                "p50_video_ms": float(snapshot["latency"][dataset]["p50_generation_seconds"])
                * 1000,
                "p95_video_ms": float(snapshot["latency"][dataset]["p95_generation_seconds"])
                * 1000,
                "mean_frame_ms": float(snapshot["latency"][dataset]["mean_seconds_per_frame"])
                * 1000,
                "peak_reserved_mb": float(snapshot["resources"][dataset]["peak_reserved_mb"]),
                "mean_gpu_utilization_percent": float(
                    snapshot["resources"][dataset]["mean_gpu_utilization_percent"]
                ),
                "measurement_note": "legacy formal infer; no stage-level latency instrumentation",
            }
            for dataset in DATASETS
        ]
    )
    _write_workbook(
        output / "02_raw_data/exp02_latency.xlsx",
        description="P0 latency; initial video-level TARDIS measurements",
        sheets={
            "per_video": latency_rows,
            "benchmark_runs": benchmark_run_frame,
            "frame_level": pd.DataFrame(
                columns=(
                    "method",
                    "dataset",
                    "resolution",
                    "batch",
                    "frame_idx",
                    "latency_vae_enc",
                    "latency_unet",
                    "latency_vae_dec",
                    "latency_post",
                    "latency_total",
                )
            ),
            "percentiles": latency_summary,
            "summary": pd.concat(
                (
                    latency_summary,
                    benchmark_run_frame[
                        [
                            "dataset",
                            "method",
                            "protocol",
                            "seed",
                            "records",
                            "mean_video_ms",
                            "mean_frame_ms",
                            "peak_reserved_mb",
                            "mean_gpu_utilization_percent",
                        ]
                    ],
                ),
                ignore_index=True,
                sort=False,
            ),
        },
    )
    _write_measured_analysis_workbooks(
        output,
        run_frame=benchmark_run_frame,
        per_video=benchmark_per_video,
        frame_level=benchmark_frame_level,
        source_per_video=source_per_video,
        source_frame_level=_benchmark_frame_level_frame(source_runs),
        source_protocol=source_protocol,
    )
    _write_ablation_workbook(
        output,
        runs=ablation_runs,
        source_runs=source_ablation_runs,
    )
    _write_source_workbook(
        output,
        run_frame=source_run_frame,
        per_video=source_per_video,
        frame_level=_benchmark_frame_level_frame(source_runs),
        protocol=source_protocol,
        summary=source_summary,
        statistics=source_statistics,
    )
    skeletons = {
        "exp08_user_study.xlsx": (
            "P1 real human study; synthetic ratings are prohibited",
            ("raw_ratings", "pair_stats", "mos"),
        ),
    }
    for filename, (description, sheet_names) in skeletons.items():
        sheets = {name: pd.DataFrame({"status": ["planned"]}) for name in sheet_names}
        sheets["summary"] = pd.DataFrame(
            {"status": ["planned"], "note": ["No measurements have been fabricated."]}
        )
        _write_workbook(output / "02_raw_data" / filename, description=description, sheets=sheets)


def _write_measured_analysis_workbooks(
    output: Path,
    *,
    run_frame: pd.DataFrame,
    per_video: pd.DataFrame,
    frame_level: pd.DataFrame,
    source_per_video: pd.DataFrame,
    source_frame_level: pd.DataFrame,
    source_protocol: str,
) -> None:
    selected_frames = _protocol_frame(frame_level, "paper50")
    selected_runs = _protocol_frame(run_frame, "paper50")
    selected_videos = _protocol_frame(per_video, "paper50")

    tc_rows = selected_frames.dropna(subset=["TC"])
    tc_evolution = _group_time_metric(tc_rows, "TC")
    lpips_evolution = _group_time_metric(selected_frames, "LPIPS")
    temporal_summary = _group_metric_summary(
        selected_videos,
        group_columns=("dataset", "method"),
        metrics=("TC", "LPIPS"),
    )
    selected_source_frames = _protocol_frame(source_frame_level, source_protocol)
    selected_source_videos = _protocol_frame(source_per_video, source_protocol)
    source_tc_evolution = _group_time_metric(selected_source_frames.dropna(subset=["TC"]), "TC")
    source_tlpips_evolution = _group_time_metric(
        selected_source_frames.dropna(subset=["tLPIPS"]),
        "tLPIPS",
    )
    source_motion_bins = _motion_bin_frame(source_frame_level, source_protocol)
    source_lag = _temporal_lag_frame(source_per_video, source_protocol)
    source_flicker = selected_source_frames.dropna(subset=["flicker_flag"])[
        [
            "experiment_id",
            "dataset",
            "method",
            "video_id",
            "frame_idx",
            "seed",
            "global_seed",
            "brightness",
            "brightness_delta",
            "flicker_flag",
            "motion_magnitude",
        ]
    ].copy()
    source_temporal_summary = _group_metric_summary(
        selected_source_videos,
        group_columns=("dataset", "method"),
        metrics=(
            "TC",
            "LPIPS",
            "flow_warp_error",
            "tLPIPS",
            "flicker_rate",
            "drift_slope",
        ),
    )
    _write_workbook(
        output / "02_raw_data/exp03_tc_analysis.xlsx",
        description="P1 measured TC and LPIPS temporal evolution under paper50",
        sheets={
            "tc_time_evolution": tc_evolution,
            "lpips_time_evolution": lpips_evolution,
            "source_tc_time_evolution": source_tc_evolution,
            "source_tlpips_time_evolution": source_tlpips_evolution,
            "tc_motion_bins": (
                source_motion_bins
                if not source_motion_bins.empty
                else _planned_frame("requires completed source50_diagnostics runs")
            ),
            "tc_temporal_lag": (
                source_lag
                if not source_lag.empty
                else _planned_frame("requires completed source50_diagnostics runs")
            ),
            "scene_categories": _planned_frame("requires audited scene annotations"),
            "flicker": (
                source_flicker
                if not source_flicker.empty
                else _planned_frame("requires completed source50_diagnostics runs")
            ),
            "summary": pd.concat(
                (
                    temporal_summary.assign(protocol="paper50"),
                    source_temporal_summary.assign(protocol=source_protocol),
                ),
                ignore_index=True,
                sort=False,
            ),
        },
    )

    seed_stability = selected_runs[
        [
            "dataset",
            "method",
            "seed",
            "records",
            "TC",
            "LPIPS",
            "FVD",
            "FID",
            "CLIP_score",
            "SSIM",
        ]
    ].copy()
    robustness_summary = _group_metric_summary(
        seed_stability,
        group_columns=("dataset", "method"),
        metrics=("TC", "LPIPS", "FVD", "FID", "CLIP_score", "SSIM"),
    )
    _write_workbook(
        output / "02_raw_data/exp05_robustness.xlsx",
        description="P2 measured five-seed stability; remaining interventions are pending",
        sheets={
            "seed_stability": seed_stability,
            "prompt_styles": _planned_frame("requires three controlled prompt sets to be run"),
            "noise_perturb": _planned_frame("requires source-noise intervention runs"),
            "video_sources": _planned_frame("requires audited source-domain labels"),
            "summary": robustness_summary,
        },
    )

    params = (
        selected_runs[
            ["method", "parameter_count", "diffusion_steps", "peak_reserved_mb"]
        ]
        .drop_duplicates(subset=["method"])
        .sort_values("method")
        .reset_index(drop=True)
    )
    accel = selected_runs[
        [
            "dataset",
            "method",
            "seed",
            "records",
            "mean_video_ms",
            "p50_video_ms",
            "p95_video_ms",
            "mean_frame_ms",
            "peak_reserved_mb",
            "mean_gpu_utilization_percent",
        ]
    ].copy()
    efficiency_summary = _group_metric_summary(
        selected_runs,
        group_columns=("dataset", "method"),
        metrics=(
            "TC",
            "LPIPS",
            "mean_frame_ms",
            "peak_reserved_mb",
            "mean_gpu_utilization_percent",
        ),
    )
    overhead = _overhead_gain_frame(efficiency_summary)
    _write_workbook(
        output / "02_raw_data/exp06_efficiency.xlsx",
        description="P2 measured parameter, latency, memory and primary-metric trade-offs",
        sheets={
            "params_macs_mem": params,
            "accel_configs": accel,
            "overhead_vs_gain": overhead,
            "summary": efficiency_summary,
        },
    )


def _write_ablation_workbook(
    output: Path,
    *,
    runs: list[dict[str, Any]],
    source_runs: list[dict[str, Any]],
) -> None:
    run_frame = _benchmark_run_frame(runs)
    per_video = _benchmark_per_video_frame(runs)
    components = {
        "tardis_a0": "full-frame prompt prior only",
        "tardis_a1": "+ previous-frame conditioning",
        "tardis_a2": "+ temporal residual prediction",
        "tardis_a3": "+ source-motion transport",
        "tardis_a4": "+ analytical visibility",
        "tardis_a5": "+ learned VCIR",
        "tardis_a6": "+ dual-frequency residual",
        "tardis_a7": "+ fixed-budget routing",
        "tardis_a8": "+ innovation proper time",
        "tardis_a9": "+ CRCD",
        "tardis_a10": "+ metric-aligned objective (full)",
    }
    if run_frame.empty:
        ablation_main = pd.DataFrame(
            columns=("method", "component", "TC", "LPIPS", "mean_frame_ms")
        )
    else:
        ablation_main = run_frame.copy()
        ablation_main.insert(
            2,
            "component",
            ablation_main["method"].map(components).fillna("unknown"),
        )
        ablation_main["level"] = ablation_main["method"].map(_ablation_level)
        ablation_main = ablation_main.sort_values("level").reset_index(drop=True)
    incremental = _ablation_incremental_frame(ablation_main)
    paired = _ablation_paired_frame(per_video)
    source_run_frame = _benchmark_run_frame(source_runs)
    source_per_video = _benchmark_per_video_frame(source_runs)
    source_main = _ablation_main_table(source_run_frame)
    source_paired = _ablation_paired_frame(source_per_video)
    _write_workbook(
        output / "02_raw_data/exp04_ablation.xlsx",
        description="P0 cumulative A0-A10 mechanism ablation on DataVerse paper50",
        sheets={
            "ablation_main": ablation_main,
            "incremental_contribution": incremental,
            "paired_to_full": paired,
            "window_sweep": _planned_frame("not represented by cumulative A0-A10 variants"),
            "lambda_sweep": _planned_frame("requires dedicated loss-weight runs"),
            "kf_sweep": _planned_frame("requires dedicated keyframe interval runs"),
            "steps_sweep": _planned_frame("requires dedicated sampler-step runs"),
            "summary": ablation_main,
            "source_ablation_main": source_main,
            "source_paired_to_full": source_paired,
        },
    )


def _ablation_main_table(run_frame: pd.DataFrame) -> pd.DataFrame:
    components = {
        "tardis_a0": "full-frame prompt prior only",
        "tardis_a1": "+ previous-frame conditioning",
        "tardis_a2": "+ temporal residual prediction",
        "tardis_a3": "+ source-motion transport",
        "tardis_a4": "+ analytical visibility",
        "tardis_a5": "+ learned VCIR",
        "tardis_a6": "+ dual-frequency residual",
        "tardis_a7": "+ fixed-budget routing",
        "tardis_a8": "+ innovation proper time",
        "tardis_a9": "+ CRCD",
        "tardis_a10": "+ metric-aligned objective (full)",
    }
    if run_frame.empty:
        return pd.DataFrame()
    result = run_frame.copy()
    result.insert(2, "component", result["method"].map(components).fillna("unknown"))
    result["level"] = result["method"].map(_ablation_level)
    return result.sort_values("level").reset_index(drop=True)


def _write_source_workbook(
    output: Path,
    *,
    run_frame: pd.DataFrame,
    per_video: pd.DataFrame,
    frame_level: pd.DataFrame,
    protocol: str,
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    selected_runs = _protocol_frame(run_frame, protocol)
    selected_videos = _protocol_frame(per_video, protocol)
    selected_frames = _protocol_frame(frame_level, protocol)
    diagnostic_summary = _group_metric_summary(
        selected_videos,
        group_columns=("dataset", "method"),
        metrics=(
            "flow_warp_error",
            "tLPIPS",
            "flicker_rate",
            "drift_slope",
            "motion_magnitude",
            "TC_lag_1",
            "TC_lag_2",
            "TC_lag_4",
            "TC_lag_8",
        ),
    )
    _write_workbook(
        output / "02_raw_data/exp07_generalization.xlsx",
        description=(
            "P2 source-conditioned cross-dataset benchmark under fixed source strength"
        ),
        sheets={
            "all_source_runs": run_frame,
            "source50_runs": selected_runs,
            "source50_per_video": selected_videos,
            "source50_frame_level": selected_frames,
            "paired_statistics": statistics,
            "diagnostic_summary": diagnostic_summary,
            "cross_dataset": summary,
            "cross_backbone": _planned_frame("all current source adapters share SD-Turbo"),
            "cross_task": _planned_frame("requires audited task-category annotations"),
            "summary": summary,
        },
    )


def _ablation_level(method: object) -> int:
    text = str(method)
    return int(text.removeprefix("tardis_a")) if text.startswith("tardis_a") else -1


def _ablation_incremental_frame(ablation: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "from_variant",
        "to_variant",
        "component_added",
        "tc_improvement",
        "lpips_improvement",
        "frame_latency_delta_ms",
    )
    if len(ablation) < 2:
        return pd.DataFrame(columns=columns)
    rows = []
    for index in range(1, len(ablation)):
        previous = ablation.iloc[index - 1]
        current = ablation.iloc[index]
        rows.append(
            {
                "from_variant": previous["method"],
                "to_variant": current["method"],
                "component_added": current["component"],
                "tc_improvement": float(previous["TC"]) - float(current["TC"]),
                "lpips_improvement": float(previous["LPIPS"]) - float(current["LPIPS"]),
                "frame_latency_delta_ms": (
                    float(current["mean_frame_ms"]) - float(previous["mean_frame_ms"])
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _ablation_paired_frame(per_video: pd.DataFrame) -> pd.DataFrame:
    from tardis.experiments.statistics import compare_lower_is_better, holm_adjust

    columns = (
        "variant",
        "metric",
        "records",
        "full_mean",
        "variant_mean",
        "full_improvement",
        "ci_low",
        "ci_high",
        "p_value_one_sided",
        "p_value_holm",
    )
    full = per_video[per_video["method"] == "tardis_a10"]
    if len(full) != 50:
        return pd.DataFrame(columns=columns)
    full = full.set_index("video_id")
    rows: list[dict[str, Any]] = []
    for method in sorted(set(per_video["method"]) - {"tardis_a10"}, key=_ablation_level):
        variant = per_video[per_video["method"] == method].set_index("video_id")
        if len(variant) != 50 or set(variant.index) != set(full.index):
            continue
        ordered = sorted(full.index)
        for metric in ("TC", "LPIPS"):
            comparison = compare_lower_is_better(
                full.loc[ordered, metric].to_numpy(dtype="float64"),
                variant.loc[ordered, metric].to_numpy(dtype="float64"),
            )
            rows.append(
                {
                    "variant": method,
                    "metric": metric,
                    "records": comparison.sample_count,
                    "full_mean": comparison.ours_mean,
                    "variant_mean": comparison.benchmark_mean,
                    "full_improvement": comparison.absolute_improvement,
                    "ci_low": comparison.ci_low,
                    "ci_high": comparison.ci_high,
                    "p_value_one_sided": comparison.p_value_one_sided,
                    "p_value_holm": pd.NA,
                }
            )
    if rows:
        adjusted = holm_adjust([float(row["p_value_one_sided"]) for row in rows])
        for row, value in zip(rows, adjusted, strict=True):
            row["p_value_holm"] = value
    return pd.DataFrame(rows, columns=columns)


def _group_time_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    columns = (
        "dataset",
        "method",
        "frame_idx",
        "observations",
        "mean",
        "std",
        "p50",
        "p95",
    )
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in frame.groupby(["dataset", "method", "frame_idx"], sort=True):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        dataset, method, frame_idx = keys
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "frame_idx": int(frame_idx),
                "observations": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "p50": float(values.quantile(0.50)),
                "p95": float(values.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _motion_bin_frame(frame_level: pd.DataFrame, protocol: str) -> pd.DataFrame:
    columns = (
        "dataset",
        "method",
        "motion_bin",
        "observations",
        "TC_mean",
        "TC_std",
        "flow_warp_error_mean",
        "tLPIPS_mean",
    )
    if frame_level.empty:
        return pd.DataFrame(columns=columns)
    selected = frame_level[frame_level["protocol"] == protocol].copy()
    selected["motion_magnitude"] = pd.to_numeric(
        selected["motion_magnitude"], errors="coerce"
    )
    selected = selected.dropna(subset=["motion_magnitude", "TC"])
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["motion_bin"] = pd.cut(
        selected["motion_magnitude"],
        bins=(0.0, 0.5, 2.0, 5.0, np.inf),
        labels=("0-0.5", "0.5-2", "2-5", ">5"),
        right=False,
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(
        ["dataset", "method", "motion_bin"],
        observed=True,
        sort=True,
    ):
        dataset, method, motion_bin = keys
        tc_values = pd.to_numeric(group["TC"], errors="coerce").dropna()
        flow_values = pd.to_numeric(group["flow_warp_err"], errors="coerce").dropna()
        tlpips_values = pd.to_numeric(group["tLPIPS"], errors="coerce").dropna()
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "motion_bin": str(motion_bin),
                "observations": int(len(group)),
                "TC_mean": float(tc_values.mean()),
                "TC_std": float(tc_values.std(ddof=0)),
                "flow_warp_error_mean": (
                    float(flow_values.mean()) if not flow_values.empty else pd.NA
                ),
                "tLPIPS_mean": (
                    float(tlpips_values.mean()) if not tlpips_values.empty else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _temporal_lag_frame(per_video: pd.DataFrame, protocol: str) -> pd.DataFrame:
    columns = (
        "dataset",
        "method",
        "lag",
        "observations",
        "TC_mean",
        "TC_std",
        "TC_p50",
        "TC_p95",
    )
    if per_video.empty:
        return pd.DataFrame(columns=columns)
    selected = per_video[per_video["protocol"] == protocol]
    rows: list[dict[str, Any]] = []
    for lag in (1, 2, 4, 8):
        name = f"TC_lag_{lag}"
        if name not in selected:
            continue
        for keys, group in selected.groupby(["dataset", "method"], sort=True):
            values = pd.to_numeric(group[name], errors="coerce").dropna()
            if values.empty:
                continue
            dataset, method = keys
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "lag": lag,
                    "observations": int(len(values)),
                    "TC_mean": float(values.mean()),
                    "TC_std": float(values.std(ddof=0)),
                    "TC_p50": float(values.quantile(0.50)),
                    "TC_p95": float(values.quantile(0.95)),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(["dataset", "method", "lag"]).reset_index(drop=True)


def _group_metric_summary(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    metric_columns = tuple(
        f"{name}_{stat}" for name in metrics for stat in ("mean", "std")
    )
    columns = (*group_columns, "observations", *metric_columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for keys, group in frame.groupby(grouper, sort=True):
        values = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = dict(zip(group_columns, values, strict=True))
        row["observations"] = int(len(group))
        for metric in metrics:
            metric_values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = (
                float(metric_values.mean()) if not metric_values.empty else pd.NA
            )
            row[f"{metric}_std"] = (
                float(metric_values.std(ddof=0)) if not metric_values.empty else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _overhead_gain_frame(summary: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "dataset",
        "method",
        "baseline",
        "tc_absolute_gain",
        "lpips_absolute_gain",
        "latency_ratio",
        "memory_ratio",
    )
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for dataset, group in summary.groupby("dataset", sort=True):
        baseline = group[group["method"] == "sd_turbo_independent"]
        if len(baseline) != 1:
            continue
        reference = baseline.iloc[0]
        for _, item in group.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "method": item["method"],
                    "baseline": "sd_turbo_independent",
                    "tc_absolute_gain": reference["TC_mean"] - item["TC_mean"],
                    "lpips_absolute_gain": reference["LPIPS_mean"] - item["LPIPS_mean"],
                    "latency_ratio": _safe_ratio(
                        item["mean_frame_ms_mean"], reference["mean_frame_ms_mean"]
                    ),
                    "memory_ratio": _safe_ratio(
                        item["peak_reserved_mb_mean"], reference["peak_reserved_mb_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _safe_ratio(numerator: object, denominator: object) -> float | object:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return pd.NA
    return float(numerator) / float(denominator)


def _planned_frame(note: str) -> pd.DataFrame:
    return pd.DataFrame({"status": ["planned"], "note": [note]})


def _write_tables(
    output: Path,
    *,
    snapshot: dict[str, Any],
    benchmark_runs: list[dict[str, Any]],
    source_runs: list[dict[str, Any]],
    source_ablation_runs: list[dict[str, Any]],
    source_selection: dict[str, Any] | None,
) -> None:
    benchmark = _benchmark_run_frame(benchmark_runs)
    main = pd.concat(
        (_metric_summary_frame(snapshot), _paper50_summary_frame(benchmark)),
        ignore_index=True,
        sort=False,
    )
    main.to_excel(output / "04_tables/table01_main_comparison.xlsx", index=False)
    _write_text(output / "04_tables/table01_main_comparison.tex", main.to_latex(index=False))
    latency = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "method": "TARDIS",
                "p50_video_ms": snapshot["latency"][dataset]["p50_generation_seconds"] * 1000,
                "p95_video_ms": snapshot["latency"][dataset]["p95_generation_seconds"] * 1000,
                "mean_frame_ms": snapshot["latency"][dataset]["mean_seconds_per_frame"] * 1000,
                "under_33_3ms": False,
            }
            for dataset in DATASETS
        ]
    )
    latency = pd.concat(
        (
            latency,
            benchmark[
                [
                    "dataset",
                    "method",
                    "protocol",
                    "seed",
                    "p50_video_ms",
                    "p95_video_ms",
                    "mean_frame_ms",
                    "peak_reserved_mb",
                ]
            ],
        ),
        ignore_index=True,
        sort=False,
    )
    latency.to_excel(output / "04_tables/table02_latency.xlsx", index=False)
    _write_text(output / "04_tables/table02_latency.tex", latency.to_latex(index=False))
    paired = _paired_statistics_frame(_benchmark_per_video_frame(benchmark_runs))
    paired.to_excel(output / "04_tables/table03_paired_statistics.xlsx", index=False)
    _write_text(
        output / "04_tables/table03_paired_statistics.tex",
        paired.to_latex(index=False),
    )
    source_run_frame = _benchmark_run_frame(source_runs)
    source_protocol = "source50" if source_selection is not None else _preferred_source_protocol(
        source_run_frame
    )
    source_main = _protocol_summary_frame(source_run_frame, source_protocol)
    source_main.to_excel(output / "04_tables/table04_source_comparison.xlsx", index=False)
    _write_text(
        output / "04_tables/table04_source_comparison.tex",
        source_main.to_latex(index=False),
    )
    paper_runs = _protocol_frame(benchmark, "paper50")
    seed_stability = _protocol_summary_frame(benchmark, "paper50")
    _write_table_pair(output, "table05_seed_stability", seed_stability)
    efficiency = _group_metric_summary(
        paper_runs,
        group_columns=("dataset", "method"),
        metrics=(
            "TC",
            "LPIPS",
            "mean_frame_ms",
            "peak_reserved_mb",
            "mean_gpu_utilization_percent",
        ),
    )
    _write_table_pair(output, "table06_efficiency", efficiency)
    source_per_video = _benchmark_per_video_frame(source_runs)
    source_diagnostics = _group_metric_summary(
        _protocol_frame(source_per_video, source_protocol),
        group_columns=("dataset", "method"),
        metrics=(
            "TC",
            "LPIPS",
            "flow_warp_error",
            "tLPIPS",
            "flicker_rate",
            "drift_slope",
        ),
    )
    _write_table_pair(output, "table07_temporal_diagnostics", source_diagnostics)
    source_ablation = _ablation_main_table(_benchmark_run_frame(source_ablation_runs))
    _write_table_pair(output, "table08_source_ablation", source_ablation)


def _write_table_pair(output: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(output / "04_tables" / f"{name}.xlsx", index=False)
    _write_text(output / "04_tables" / f"{name}.tex", frame.to_latex(index=False))


def _write_figures(
    output: Path,
    *,
    snapshot: dict[str, Any],
    benchmark_runs: list[dict[str, Any]],
    source_runs: list[dict[str, Any]],
    source_ablation_runs: list[dict[str, Any]],
    source_selection: dict[str, Any] | None,
) -> None:
    figure_root = output / "03_figures"
    for path in figure_root.glob("fig*.png"):
        path.unlink()
    for path in figure_root.glob("fig*.pdf"):
        path.unlink()

    run_frame = _benchmark_run_frame(benchmark_runs)
    per_video = _benchmark_per_video_frame(benchmark_runs)
    frame_level = _benchmark_frame_level_frame(benchmark_runs)
    paper_runs = _protocol_frame(run_frame, "paper50")
    paper_videos = _protocol_frame(per_video, "paper50")
    paper_frames = _protocol_frame(frame_level, "paper50")
    if paper_runs.empty:
        _write_tardis_snapshot_figure(figure_root, snapshot)
        return

    _write_metric_profile_figure(figure_root, paper_runs)
    _write_dataset_metric_figure(figure_root, paper_runs)
    _write_latency_violin_figure(figure_root, paper_videos)
    _write_tc_evolution_figure(figure_root, paper_frames)
    _write_seed_stability_figure(figure_root, paper_runs)
    _write_latency_memory_figure(figure_root, paper_runs)
    _write_gain_overhead_figure(figure_root, paper_runs)
    source_run_frame = _benchmark_run_frame(source_runs)
    source_protocol = "source50" if source_selection is not None else _preferred_source_protocol(
        source_run_frame
    )
    source_frame = _protocol_frame(source_run_frame, source_protocol)
    source_per_video = _protocol_frame(_benchmark_per_video_frame(source_runs), source_protocol)
    source_frame_level = _protocol_frame(
        _benchmark_frame_level_frame(source_runs), source_protocol
    )
    _write_source_metric_figure(figure_root, source_frame)
    _write_source_latency_figure(figure_root, source_frame)
    _write_motion_bins_figure(figure_root, source_frame_level, source_protocol)
    _write_temporal_lag_figure(figure_root, source_per_video, source_protocol)
    _write_source_flicker_figure(figure_root, source_per_video, source_protocol)
    _write_ablation_figures(
        figure_root,
        _ablation_main_table(_benchmark_run_frame(source_ablation_runs)),
    )


def _write_tardis_snapshot_figure(figure_root: Path, snapshot: dict[str, Any]) -> None:
    labels = ["DataVerse", "Seedance", "OpenVid"]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "tc", "TARDIS Temporal Consistency"),
        (axes[1], "lpips", "TARDIS Perceptual Distance"),
    ):
        values = [snapshot["metrics"][dataset][metric] for dataset in DATASETS]
        axis.bar(positions, values, color=("#237a57", "#d1495b", "#2f6690"))
        axis.set_title(title)
        axis.set_ylabel(f"{metric.upper()} (lower is better)")
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, figure_root / "fig02_per_dataset_bar")


def _write_metric_profile_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    metrics = ("TC", "LPIPS", "FVD", "FID", "CLIP_score", "SSIM")
    numeric = runs.copy()
    for metric in metrics:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    means = numeric.groupby("method")[list(metrics)].mean()
    if len(means) < 2:
        return
    normalized = pd.DataFrame(index=means.index)
    for metric in metrics:
        values = means[metric]
        span = float(values.max() - values.min())
        if span <= 1.0e-12:
            normalized[metric] = 1.0
        elif metric in {"TC", "LPIPS", "FVD", "FID"}:
            normalized[metric] = (values.max() - values) / span
        else:
            normalized[metric] = (values - values.min()) / span
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed_angles = np.concatenate((angles, angles[:1]))
    figure, axis = plt.subplots(
        figsize=(6.4, 5.4),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for method, row in normalized.iterrows():
        values = row.to_numpy(dtype=float)
        closed = np.concatenate((values, values[:1]))
        axis.plot(closed_angles, closed, linewidth=1.8, label=method)
        axis.fill(closed_angles, closed, alpha=0.06)
    axis.set_xticks(angles, ["TC", "LPIPS", "FVD", "FID", "CLIP", "SSIM"])
    axis.set_ylim(0, 1)
    axis.set_title("Normalized Measured Metric Profile", pad=18)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), fontsize=8)
    _save_figure(figure, figure_root / "fig01_radar")


def _write_dataset_metric_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    numeric = runs.copy()
    for metric in ("TC", "LPIPS"):
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    summary = numeric.groupby(["dataset", "method"])[["TC", "LPIPS"]].mean()
    methods = sorted(set(runs["method"]))
    datasets = [dataset for dataset in DATASETS if dataset in set(runs["dataset"])]
    positions = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(len(methods), 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        for axis, metric in zip(axes, ("TC", "LPIPS"), strict=True):
            values = [
                summary.loc[(dataset, method), metric]
                if (dataset, method) in summary.index
                else np.nan
                for dataset in datasets
            ]
            axis.bar(positions + offset, values, width=width, label=method)
    for axis, metric in zip(axes, ("TC", "LPIPS"), strict=True):
        axis.set_title(f"{metric} by Dataset")
        axis.set_ylabel(f"{metric} (lower is better)")
        axis.set_xticks(positions, datasets)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    _save_figure(figure, figure_root / "fig02_per_dataset_bar")


def _write_latency_violin_figure(figure_root: Path, videos: pd.DataFrame) -> None:
    if videos.empty:
        return
    methods = sorted(set(videos["method"]))
    values = [
        pd.to_numeric(
            videos[videos["method"] == method]["latency_total_ms"], errors="coerce"
        ).dropna()
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    axis.violinplot(values, showmeans=True, showmedians=True)
    axis.axhline(16 * 1000 / 30, color="#c1121f", linestyle="--", linewidth=1.2)
    axis.set_xticks(range(1, len(methods) + 1), methods, rotation=20, ha="right")
    axis.set_ylabel("Video generation latency (ms, 16 frames)")
    axis.set_title("Measured End-to-End Latency Distribution")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, figure_root / "fig03_latency_violin")


def _write_tc_evolution_figure(figure_root: Path, frames: pd.DataFrame) -> None:
    tc = frames.dropna(subset=["TC"])
    if tc.empty:
        return
    tc = tc.copy()
    tc["TC"] = pd.to_numeric(tc["TC"], errors="coerce")
    evolution = tc.groupby(["method", "frame_idx"])["TC"].mean()
    figure, axis = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for method in sorted(set(tc["method"])):
        series = evolution.loc[method]
        axis.plot(series.index, series.values, marker="o", markersize=3, label=method)
    axis.set_xlabel("Frame index")
    axis.set_ylabel("TC (lower is better)")
    axis.set_title("Temporal Consistency Evolution")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(figure, figure_root / "fig05_tc_time_evolution")


def _write_seed_stability_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    methods = sorted(set(runs["method"]))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric in zip(axes, ("TC", "LPIPS"), strict=True):
        values = [
            pd.to_numeric(
                runs[runs["method"] == method][metric], errors="coerce"
            ).dropna()
            for method in methods
        ]
        axis.boxplot(values, tick_labels=methods, showmeans=True)
        axis.set_ylabel(f"{metric} (lower is better)")
        axis.set_title(f"{metric} Seed Stability")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, figure_root / "fig12_seed_stability")


def _write_latency_memory_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    numeric = runs.copy()
    for metric in ("mean_frame_ms", "peak_reserved_mb"):
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    summary = numeric.groupby("method")[["mean_frame_ms", "peak_reserved_mb"]].mean()
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for method, row in summary.iterrows():
        axis.scatter(row["mean_frame_ms"], row["peak_reserved_mb"], s=55)
        axis.annotate(
            method,
            (row["mean_frame_ms"], row["peak_reserved_mb"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(1000 / 30, color="#c1121f", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Mean generation latency per frame (ms)")
    axis.set_ylabel("Peak reserved GPU memory (MiB)")
    axis.set_title("Latency-Memory Trade-off")
    axis.grid(alpha=0.25)
    _save_figure(figure, figure_root / "fig13_latency_memory")


def _write_gain_overhead_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    summary = _group_metric_summary(
        runs,
        group_columns=("dataset", "method"),
        metrics=("TC", "LPIPS", "mean_frame_ms", "peak_reserved_mb"),
    )
    gains = _overhead_gain_frame(summary)
    gains = gains[gains["method"] != "sd_turbo_independent"]
    if gains.empty:
        return
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for method, group in gains.groupby("method", sort=True):
        axis.scatter(group["latency_ratio"], group["tc_absolute_gain"], s=55, label=method)
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.axvline(1, color="#444444", linewidth=0.8)
    axis.set_xlabel("Latency ratio vs SD-Turbo independent")
    axis.set_ylabel("Absolute TC improvement")
    axis.set_title("Temporal Gain vs Runtime Overhead")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(figure, figure_root / "fig14_gain_overhead")


def _write_source_metric_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        return
    numeric = runs.copy()
    for metric in ("TC", "LPIPS"):
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    summary = numeric.groupby("method")[["TC", "LPIPS"]].mean().sort_values("TC")
    positions = np.arange(len(summary), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, ("TC", "LPIPS"), strict=True):
        axis.bar(positions, summary[metric].to_numpy())
        axis.set_xticks(positions, summary.index, rotation=24, ha="right")
        axis.set_ylabel(f"{metric} (lower is better)")
        axis.set_title(f"Source-conditioned {metric}")
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, figure_root / "fig15_source_conditioned")


def _write_source_latency_figure(figure_root: Path, runs: pd.DataFrame) -> None:
    if runs.empty:
        return
    summary = (
        runs.assign(mean_frame_ms=pd.to_numeric(runs["mean_frame_ms"], errors="coerce"))
        .groupby("method", sort=True)["mean_frame_ms"]
        .mean()
        .dropna()
        .sort_values()
    )
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
    positions = np.arange(len(summary), dtype=float)
    axis.bar(positions, summary.to_numpy(dtype=float), color="#2f6690")
    axis.axhline(1000 / 30, color="#c1121f", linestyle="--", linewidth=1.2)
    axis.set_xticks(positions, summary.index, rotation=24, ha="right")
    axis.set_ylabel("Mean generation latency per frame (ms)")
    axis.set_title("Source-conditioned Measured Latency")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, figure_root / "fig04_source_latency")


def _write_motion_bins_figure(
    figure_root: Path,
    frame_level: pd.DataFrame,
    protocol: str,
) -> None:
    values = _motion_bin_frame(frame_level, protocol)
    if values.empty:
        return
    bins = ("0-0.5", "0.5-2", "2-5", ">5")
    datasets = [dataset for dataset in DATASETS if dataset in set(values["dataset"])]
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.3 * len(datasets), 4.5),
        constrained_layout=True,
        sharey=True,
    )
    for axis, dataset in zip(np.atleast_1d(axes), datasets, strict=True):
        selected = values[values["dataset"] == dataset]
        for method, group in selected.groupby("method", sort=True):
            indexed = group.set_index("motion_bin").reindex(bins)
            axis.plot(
                bins,
                indexed["TC_mean"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.35,
                markersize=3.5,
                label=method,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Source motion magnitude (px)")
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("TC (lower is better)")
    np.atleast_1d(axes)[-1].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7
    )
    _save_figure(figure, figure_root / "fig06_motion_bins")


def _write_temporal_lag_figure(
    figure_root: Path,
    per_video: pd.DataFrame,
    protocol: str,
) -> None:
    values = _temporal_lag_frame(per_video, protocol)
    if values.empty:
        return
    datasets = [dataset for dataset in DATASETS if dataset in set(values["dataset"])]
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.3 * len(datasets), 4.5),
        constrained_layout=True,
        sharey=True,
    )
    for axis, dataset in zip(np.atleast_1d(axes), datasets, strict=True):
        selected = values[values["dataset"] == dataset]
        for method, group in selected.groupby("method", sort=True):
            ordered = group.sort_values("lag")
            axis.plot(
                ordered["lag"].to_numpy(dtype=int),
                ordered["TC_mean"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.35,
                markersize=3.5,
                label=method,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Temporal lag (frames)")
        axis.set_xticks((1, 2, 4, 8))
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("Lag TC (lower is better)")
    np.atleast_1d(axes)[-1].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7
    )
    _save_figure(figure, figure_root / "fig07_temporal_lag")


def _write_source_flicker_figure(
    figure_root: Path,
    per_video: pd.DataFrame,
    protocol: str,
) -> None:
    values = _protocol_frame(per_video, protocol)
    columns = ("TC", "flicker_rate", "flow_warp_error")
    if values.empty or any(column not in values for column in columns):
        return
    numeric = values.copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    summary = numeric.groupby(["dataset", "method"], sort=True)[list(columns)].mean()
    if summary.empty or summary["flicker_rate"].notna().sum() == 0:
        return
    datasets = [dataset for dataset in DATASETS if dataset in set(numeric["dataset"])]
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.3 * len(datasets), 4.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, dataset in zip(np.atleast_1d(axes), datasets, strict=True):
        selected = summary.loc[dataset] if dataset in summary.index.levels[0] else pd.DataFrame()
        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T
        for method, row in selected.iterrows():
            if pd.isna(row["flicker_rate"]) or pd.isna(row["TC"]):
                continue
            axis.scatter(row["flicker_rate"], row["TC"], s=42)
            axis.annotate(str(method), (row["flicker_rate"], row["TC"]), fontsize=7)
        axis.set_title(dataset)
        axis.set_xlabel("Flicker rate (lower is better)")
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("TC (lower is better)")
    _save_figure(figure, figure_root / "fig08_flicker_tc")


def _write_ablation_figures(figure_root: Path, ablation: pd.DataFrame) -> None:
    if ablation.empty:
        return
    numeric = ablation.copy()
    for column in ("level", "TC", "LPIPS", "mean_frame_ms"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(subset=["level", "TC", "LPIPS"]).sort_values("level")
    if numeric.empty:
        return
    labels = [f"A{int(value)}" for value in numeric["level"]]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    for axis, metric in zip(axes, ("TC", "LPIPS"), strict=True):
        axis.plot(labels, numeric[metric].to_numpy(dtype=float), marker="o", color="#237a57")
        axis.set_title(f"Source A0-A10 {metric}")
        axis.set_ylabel(f"{metric} (lower is better)")
        axis.set_xlabel("Cumulative TARDIS variant")
        axis.grid(alpha=0.25)
    _save_figure(figure, figure_root / "fig09_source_ablation_metrics")

    incremental = _ablation_incremental_frame(numeric)
    if not incremental.empty:
        figure, axis = plt.subplots(figsize=(10.2, 4.3), constrained_layout=True)
        positions = np.arange(len(incremental), dtype=float)
        axis.bar(positions, incremental["tc_improvement"].to_numpy(dtype=float), color="#d1495b")
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(positions, incremental["to_variant"], rotation=24, ha="right")
        axis.set_ylabel("Incremental TC improvement")
        axis.set_title("Source A0-A10 Incremental Temporal Gain")
        axis.grid(axis="y", alpha=0.25)
        _save_figure(figure, figure_root / "fig10_source_ablation_increment")

    latency = numeric.dropna(subset=["mean_frame_ms"])
    if latency.empty:
        return
    figure, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    axis.scatter(latency["mean_frame_ms"], latency["TC"], s=50, color="#2f6690")
    for _, row in latency.iterrows():
        axis.annotate(
            f"A{int(row['level'])}",
            (row["mean_frame_ms"], row["TC"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean generation latency per frame (ms)")
    axis.set_ylabel("TC (lower is better)")
    axis.set_title("Source A0-A10 Quality-Latency Trade-off")
    axis.grid(alpha=0.25)
    _save_figure(figure, figure_root / "fig11_source_ablation_tradeoff")


def _save_figure(figure: Any, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=300)
    plt.close(figure)


def _write_scripts(output: Path, *, repo: Path) -> None:
    scripts = output / "05_scripts"
    _write_text(
        scripts / "run_baseline.py",
        """#!/usr/bin/env python3
from tardis.experiments.benchmark import main

if __name__ == "__main__":
    main()
""",
    )
    _write_text(
        scripts / "run_queue.py",
        """#!/usr/bin/env python3
from tardis.experiments.queue import main

if __name__ == "__main__":
    main()
""",
    )
    _write_text(
        scripts / "audit_experiments.py",
        """#!/usr/bin/env python3
from tardis.experiments.audit import main

if __name__ == "__main__":
    main()
""",
    )
    _write_text(
        scripts / "export_xlsx.py",
        """#!/usr/bin/env python3
from tardis.experiments.package import initialize_package, parse_args

if __name__ == "__main__":
    initialize_package(parse_args())
""",
    )
    for name, message in (
        ("compute_metrics.py", "Metrics are computed by tardis.experiments.benchmark."),
        ("analyze_tc.py", "Pending: consumes exp01 paired frame-level results."),
        ("analyze_ablation.py", "Pending: consumes TARDIS A0-A10 experiment records."),
        ("make_figures.py", "Figures are regenerated by tardis.experiments.package."),
        ("make_tables.py", "Tables are regenerated by tardis.experiments.package."),
    ):
        _write_text(
            scripts / name,
            f'#!/usr/bin/env python3\n"""{message}"""\nprint({message!r})\n',
        )
    _write_text(scripts / "verify_package.py", _verification_script())
    _write_text(scripts / "verify_primary_claims.py", _primary_claim_verification_script())
    _write_text(
        scripts / "README_scripts.md",
        f"""# Scripts

## Verify

```bash
python {scripts / 'verify_package.py'}
python {scripts / 'verify_primary_claims.py'}
```

## Run one compatible benchmark

```bash
cd {repo}
torchrun --standalone --nproc_per_node=1 -m tardis.experiments.benchmark \
  --method METHOD --dataset DATASET --output OUTPUT_DIR
```

Prompt-only METHOD values: `tardis`, `sd_turbo_independent`,
`animatediff_lightning`, `text2video_zero`.

Source-conditioned METHOD values: `tardis`, `streamdiffusion_img2img`,
`rerender_flow`, `tokenflow_core`, `vid2vid_zero_core`, `controlvideo_canny`,
`stablevideo_propagation`. Except for TARDIS, these adapters are recorded as
`audited core-mechanism reproduction`; they are not claimed as official-repository runs.

## Run the complete resumable paper50 queue

```bash
cd {repo}
python -m tardis.experiments.queue
```

The queue runs 4 compatible methods x 3 datasets x 5 seeds. It forces local/offline model
caches, writes each unit under `TARDIS_PAPER_EXPERIMENTS/main/`, and refreshes this package after
every completed unit.

## Run source-conditioned benchmark queue

```bash
bash /home/TARDIS/scripts/run_source_benchmarks.sh
```

This is an independent `source50` protocol. It uses the source video as both condition and metric
reference with fixed `source_strength=0.45`; core reproductions are explicitly marked in each
`run_manifest.json`.

## Add prompt-only baselines to the source50 comparison

```bash
bash /home/TARDIS/scripts/run_source_prompt_baselines.sh
```

This resumable 45-unit supplement evaluates SD-Turbo independent, AnimateDiff-Lightning and
Text2Video-Zero against the same source references, records and seeds. These methods intentionally
ignore the source condition, providing controlled lower-bound comparisons in the source50 table.

## Run A0-A10 ablation

```bash
bash /home/TARDIS/scripts/run_paper_ablations.sh
```

## Run source-conditioned A0-A10 ablation

```bash
bash /home/TARDIS/scripts/run_source_ablations.sh
```

## Refresh XLSX, tables, figures and manifest

```bash
cd {repo}
python -m tardis.experiments.package --refresh
```
""",
    )
    for path in scripts.glob("*.py"):
        path.chmod(0o755)


def _write_workbook(
    path: Path,
    *,
    description: str,
    sheets: dict[str, pd.DataFrame],
) -> None:
    release = _read_release_metadata(path.parent.parent)
    meta = pd.DataFrame(
        [
            ("package_version", release["version"], ""),
            ("schema_version", "1", ""),
            ("generated_at_utc", _utc_now(), ""),
            ("status", release["status"], ""),
            ("description", description, ""),
            ("hardware", "NVIDIA GeForce RTX 4080 SUPER", ""),
            *FIELD_DICTIONARY,
        ],
        columns=("field", "description_or_value", "unit_or_type"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="_meta", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def _metric_summary_frame(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        item = snapshot["metrics"][dataset]
        rows.append(
            {
                "dataset": dataset,
                "method": "TARDIS",
                "records": 512,
                "TC": item["tc"],
                "LPIPS": item["lpips"],
                "FVD": item["fvd"],
                "FID": item["fid"],
                "CLIP_score": item["clipscore"],
                "SSIM": item["ssim"],
                "provenance": "formal complete test, EMA checkpoint",
            }
        )
    rows.append(
        {
            "dataset": "three_dataset_macro_average",
            "method": "TARDIS",
            "records": 1536,
            "TC": fmean(row["TC"] for row in rows),
            "LPIPS": fmean(row["LPIPS"] for row in rows),
            "FVD": fmean(row["FVD"] for row in rows),
            "FID": fmean(row["FID"] for row in rows),
            "CLIP_score": fmean(row["CLIP_score"] for row in rows),
            "SSIM": fmean(row["SSIM"] for row in rows),
            "provenance": "equal-weight mean over three complete test summaries",
        }
    )
    return pd.DataFrame(rows)


def _benchmark_run_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    columns = (
        "experiment_id",
        "dataset",
        "method",
        "protocol",
        "metric_mode",
        "source_strength",
        "seed",
        "records",
        "TC",
        "LPIPS",
        "FVD",
        "FID",
        "CLIP_score",
        "SSIM",
        "mean_video_ms",
        "p50_video_ms",
        "p95_video_ms",
        "mean_frame_ms",
        "peak_reserved_mb",
        "mean_gpu_utilization_percent",
        "parameter_count",
        "diffusion_steps",
        "elapsed_wall_seconds",
        "provenance",
        "artifact_root",
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        payload = run["payload"]
        settings = payload["settings"]
        macro = payload["metrics"]["macro"]
        latency = payload.get("latency", {})
        resources = payload.get("resources", {})
        generator = payload.get("generator", {})
        method = str(settings["method"])
        dataset = str(settings["dataset"])
        seed = int(settings["global_seed"])
        rows.append(
            {
                "experiment_id": f"exp01_{method}_{dataset}_seed{seed}",
                "dataset": dataset,
                "method": method,
                "protocol": str(settings["protocol"]),
                "metric_mode": str(settings["metric_mode"]),
                "source_strength": _optional_float(settings.get("source_strength")),
                "seed": seed,
                "records": int(payload["coverage"]["completed"]),
                "TC": _optional_float(macro.get("tc")),
                "LPIPS": _optional_float(macro.get("lpips")),
                "FVD": _optional_float(macro.get("fvd")),
                "FID": _optional_float(macro.get("fid")),
                "CLIP_score": _optional_float(macro.get("clipscore")),
                "SSIM": _optional_float(macro.get("ssim")),
                "mean_video_ms": _seconds_to_ms(latency.get("mean_video_seconds")),
                "p50_video_ms": _seconds_to_ms(latency.get("p50_video_seconds")),
                "p95_video_ms": _seconds_to_ms(latency.get("p95_video_seconds")),
                "mean_frame_ms": _optional_float(latency.get("mean_frame_milliseconds")),
                "peak_reserved_mb": _optional_float(resources.get("peak_reserved_mb")),
                "mean_gpu_utilization_percent": _optional_float(
                    resources.get("mean_gpu_utilization_percent")
                ),
                "parameter_count": _optional_float(generator.get("parameter_count")),
                "diffusion_steps": _optional_float(generator.get("diffusion_steps")),
                "elapsed_wall_seconds": _optional_float(payload.get("elapsed_wall_seconds")),
                "provenance": "measured_current_prompt_only_protocol",
                "artifact_root": str(run["run_root"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _benchmark_per_video_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    columns = (
        "experiment_id",
        "dataset",
        "method",
        "variant",
        "video_id",
        "seed",
        "global_seed",
        "prompt_id",
        "prompt",
        "LPIPS",
        "TC",
        "flow_warp_error",
        "tLPIPS",
        "flicker_rate",
        "drift_slope",
        "motion_magnitude",
        "TC_lag_1",
        "TC_lag_2",
        "TC_lag_4",
        "TC_lag_8",
        "FVD",
        "FID",
        "CLIP_score",
        "SSIM",
        "latency_total_ms",
        "gpu_mem_mb",
        "status",
        "protocol",
        "provenance",
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        payload = run["payload"]
        settings = payload["settings"]
        resources = payload.get("resources", {})
        method = str(settings["method"])
        dataset = str(settings["dataset"])
        global_seed = int(settings["global_seed"])
        for item in run["records"]:
            record_id = str(item["record_id"])
            rows.append(
                {
                    "experiment_id": str(item["experiment_id"]),
                    "dataset": dataset,
                    "method": method,
                    "variant": "A10" if method == "tardis" else pd.NA,
                    "video_id": record_id,
                    "seed": int(item["seed"]),
                    "global_seed": global_seed,
                    "prompt_id": _prompt_id(dataset, record_id),
                    "prompt": str(item["prompt"]),
                    "LPIPS": float(item["lpips"]),
                    "TC": float(item["tc"]),
                    "flow_warp_error": _diagnostic_float(item, "flow_warp_error"),
                    "tLPIPS": _diagnostic_float(item, "tlpips"),
                    "flicker_rate": _diagnostic_float(item, "flicker_rate"),
                    "drift_slope": _diagnostic_float(item, "drift_slope"),
                    "motion_magnitude": _diagnostic_float(item, "motion_magnitude"),
                    "TC_lag_1": _diagnostic_lag(item, 1),
                    "TC_lag_2": _diagnostic_lag(item, 2),
                    "TC_lag_4": _diagnostic_lag(item, 4),
                    "TC_lag_8": _diagnostic_lag(item, 8),
                    "FVD": pd.NA,
                    "FID": pd.NA,
                    "CLIP_score": pd.NA,
                    "SSIM": pd.NA,
                    "latency_total_ms": float(item["generation_seconds"]) * 1000,
                    "gpu_mem_mb": _optional_float(resources.get("peak_reserved_mb")),
                    "status": str(item["status"]),
                    "protocol": str(settings["protocol"]),
                    "provenance": "measured per-video paired result",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _benchmark_frame_level_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    columns = (
        "experiment_id",
        "dataset",
        "method",
        "video_id",
        "frame_idx",
        "seed",
        "global_seed",
        "LPIPS",
        "TC",
        "flow_warp_err",
        "tLPIPS",
        "motion_magnitude",
        "brightness",
        "brightness_delta",
        "flicker_flag",
        "CLIP_score",
        "protocol",
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        settings = run["payload"]["settings"]
        dataset = str(settings["dataset"])
        method = str(settings["method"])
        protocol = str(settings["protocol"])
        global_seed = int(settings["global_seed"])
        for item in run["records"]:
            tc_values = [float(value) for value in item["tc_per_transition"]]
            lpips_values = [float(value) for value in item["lpips_per_frame"]]
            if len(tc_values) + 1 != len(lpips_values):
                raise ValueError("benchmark frame metric arrays have incompatible lengths")
            flow_warp_values = _diagnostic_series(
                item,
                "flow_warp_error_per_transition",
                len(tc_values),
            )
            tlpips_values = _diagnostic_series(
                item,
                "tlpips_per_transition",
                len(tc_values),
            )
            motion_values = _diagnostic_series(
                item,
                "motion_magnitude_per_transition",
                len(tc_values),
            )
            brightness_values = _diagnostic_series(
                item,
                "brightness_per_frame",
                len(lpips_values),
            )
            brightness_delta_values = _diagnostic_series(
                item,
                "brightness_delta_per_transition",
                len(tc_values),
            )
            flicker_values = _diagnostic_bool_series(
                item,
                "flicker_flags",
                len(tc_values),
            )
            for frame_idx, lpips_value in enumerate(lpips_values):
                rows.append(
                    {
                        "experiment_id": str(item["experiment_id"]),
                        "dataset": dataset,
                        "method": method,
                        "video_id": str(item["record_id"]),
                        "frame_idx": frame_idx,
                        "seed": int(item["seed"]),
                        "global_seed": global_seed,
                        "LPIPS": lpips_value,
                        "TC": pd.NA if frame_idx == 0 else tc_values[frame_idx - 1],
                        "flow_warp_err": (
                            pd.NA if frame_idx == 0 else flow_warp_values[frame_idx - 1]
                        ),
                        "tLPIPS": pd.NA if frame_idx == 0 else tlpips_values[frame_idx - 1],
                        "motion_magnitude": (
                            pd.NA if frame_idx == 0 else motion_values[frame_idx - 1]
                        ),
                        "brightness": brightness_values[frame_idx],
                        "brightness_delta": (
                            pd.NA
                            if frame_idx == 0
                            else brightness_delta_values[frame_idx - 1]
                        ),
                        "flicker_flag": (
                            pd.NA if frame_idx == 0 else flicker_values[frame_idx - 1]
                        ),
                        "CLIP_score": pd.NA,
                        "protocol": protocol,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _paper50_summary_frame(run_frame: pd.DataFrame) -> pd.DataFrame:
    return _protocol_summary_frame(run_frame, "paper50")


def _preferred_source_protocol(run_frame: pd.DataFrame) -> str:
    methods = {
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
    }
    diagnostics = run_frame[run_frame["protocol"] == "source50_diagnostics"]
    expected_pairs = {(dataset, method) for dataset in DATASETS for method in methods}
    if diagnostics.empty:
        return "source50"
    seed_counts = diagnostics.groupby(["dataset", "method"])["seed"].nunique()
    observed_pairs = {(str(dataset), str(method)) for dataset, method in seed_counts.index}
    if observed_pairs != expected_pairs or not bool((seed_counts == 5).all()):
        return "source50"
    return "source50_diagnostics"


def _protocol_summary_frame(run_frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    columns = (
        "dataset",
        "method",
        "records",
        "seed_runs",
        "TC",
        "TC_std",
        "LPIPS",
        "LPIPS_std",
        "FVD",
        "FVD_std",
        "FID",
        "FID_std",
        "CLIP_score",
        "CLIP_score_std",
        "SSIM",
        "SSIM_std",
        "provenance",
    )
    if run_frame.empty:
        return pd.DataFrame(columns=columns)
    selected = run_frame[
        (run_frame["protocol"] == protocol) & (run_frame["metric_mode"] == "full")
    ]
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in selected.groupby(["dataset", "method"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "records": int(group["records"].sum()),
            "seed_runs": int(len(group)),
            "provenance": f"equal-weight mean over completed {protocol} seed runs",
        }
        for metric in ("TC", "LPIPS", "FVD", "FID", "CLIP_score", "SSIM"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(values.mean()) if not values.empty else pd.NA
            row[f"{metric}_std"] = (
                float(values.std(ddof=0)) if not values.empty else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _paired_statistics_frame(
    per_video: pd.DataFrame,
    *,
    protocol: str = "paper50",
) -> pd.DataFrame:
    from tardis.experiments.statistics import compare_lower_is_better, holm_adjust

    columns = (
        "dataset",
        "benchmark",
        "metric",
        "records",
        "seed_runs",
        "tardis_mean",
        "benchmark_mean",
        "absolute_improvement",
        "relative_improvement_percent",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "win_rate",
        "tie_rate",
        "wilcoxon_statistic",
        "p_value_one_sided",
        "p_value_holm",
        "tardis_better",
    )
    if per_video.empty:
        return pd.DataFrame(columns=columns)
    selected = per_video[per_video["protocol"] == protocol].copy()
    if selected.empty or "tardis" not in set(selected["method"]):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for dataset in sorted(set(selected["dataset"])):
        dataset_frame = selected[selected["dataset"] == dataset]
        tardis = dataset_frame[dataset_frame["method"] == "tardis"]
        tardis_seed_count = int(tardis["global_seed"].nunique())
        if tardis_seed_count != 5:
            continue
        tardis_mean = tardis.groupby("video_id")[["TC", "LPIPS"]].mean()
        if len(tardis_mean) != 50:
            continue
        for method in sorted(set(dataset_frame["method"]) - {"tardis"}):
            benchmark = dataset_frame[dataset_frame["method"] == method]
            benchmark_seed_count = int(benchmark["global_seed"].nunique())
            if benchmark_seed_count != 5:
                continue
            benchmark_mean = benchmark.groupby("video_id")[["TC", "LPIPS"]].mean()
            if len(benchmark_mean) != 50 or set(benchmark_mean.index) != set(tardis_mean.index):
                continue
            ordered = sorted(tardis_mean.index)
            for metric in ("TC", "LPIPS"):
                comparison = compare_lower_is_better(
                    tardis_mean.loc[ordered, metric].to_numpy(dtype="float64"),
                    benchmark_mean.loc[ordered, metric].to_numpy(dtype="float64"),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "benchmark": method,
                        "metric": metric,
                        "records": comparison.sample_count,
                        "seed_runs": tardis_seed_count,
                        "tardis_mean": comparison.ours_mean,
                        "benchmark_mean": comparison.benchmark_mean,
                        "absolute_improvement": comparison.absolute_improvement,
                        "relative_improvement_percent": (
                            comparison.relative_improvement_percent
                        ),
                        "bootstrap_ci_low": comparison.ci_low,
                        "bootstrap_ci_high": comparison.ci_high,
                        "win_rate": comparison.win_rate,
                        "tie_rate": comparison.tie_rate,
                        "wilcoxon_statistic": comparison.wilcoxon_statistic,
                        "p_value_one_sided": comparison.p_value_one_sided,
                        "p_value_holm": pd.NA,
                        "tardis_better": (
                            comparison.absolute_improvement > 0
                            and comparison.ci_low > 0
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    p_values = [float(row["p_value_one_sided"]) for row in rows]
    for row, adjusted in zip(rows, holm_adjust(p_values), strict=True):
        row["p_value_holm"] = adjusted
        row["tardis_better"] = bool(row["tardis_better"] and adjusted < 0.05)
    return pd.DataFrame(rows, columns=columns)


def _dataset_frame(frame: pd.DataFrame, dataset: str, empty: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty.copy()
    return frame[frame["dataset"] == dataset].reset_index(drop=True)


def _protocol_frame(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["protocol"] == protocol].reset_index(drop=True)


def _diagnostic_float(item: dict[str, Any], name: str) -> float | object:
    value = item.get(name)
    return _optional_float(value)


def _diagnostic_lag(item: dict[str, Any], lag: int) -> float | object:
    values = item.get("tc_by_lag")
    if not isinstance(values, dict):
        return pd.NA
    return _optional_float(values.get(str(lag)))


def _diagnostic_series(
    item: dict[str, Any],
    name: str,
    expected_count: int,
) -> list[float | object]:
    values = item.get(name)
    if values is None:
        return [pd.NA] * expected_count
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"benchmark diagnostic {name} has incompatible length")
    return [_optional_float(value) for value in values]


def _diagnostic_bool_series(
    item: dict[str, Any],
    name: str,
    expected_count: int,
) -> list[bool | object]:
    values = item.get(name)
    if values is None:
        return [pd.NA] * expected_count
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"benchmark diagnostic {name} has incompatible length")
    return [bool(value) for value in values]


def _optional_float(value: object) -> float | object:
    return pd.NA if value is None else float(value)


def _seconds_to_ms(value: object) -> float | object:
    return pd.NA if value is None else float(value) * 1000


def _selected_records(
    ledgers: dict[str, list[dict[str, Any]]],
    *,
    count: int,
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for dataset, records in ledgers.items():
        result[dataset] = sorted(
            records,
            key=lambda item: hashlib.sha256(
                f"3407\x1f{dataset}\x1f{item['record_id']}".encode()
            ).hexdigest(),
        )[:count]
    return result


def _write_manifest(output: Path) -> None:
    release = _read_release_metadata(output)
    manifest_path = output / "MANIFEST.json"
    manifest_path.unlink(missing_ok=True)
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "package": output.name,
            "version": release["version"],
            "status": release["status"],
            "generated_at_utc": _utc_now(),
            "file_count_excluding_manifest": len(files),
            "files": files,
        },
    )


def _release_metadata(audit_report: dict[str, Any], *, version: str = "1.0") -> dict[str, str]:
    if not version or version.endswith("-initial"):
        raise ValueError("release version must be a non-initial, non-empty identifier")
    if audit_report.get("status") == "complete":
        return {"version": version, "status": "final_complete"}
    return {"version": f"{version}-initial", "status": "initial_in_progress"}


def _read_release_metadata(output: Path) -> dict[str, str]:
    version_path = output / "VERSION"
    if not version_path.is_file():
        return _release_metadata({"status": "incomplete"})
    version = version_path.read_text(encoding="utf-8").strip()
    if not version:
        return _release_metadata({"status": "incomplete"})
    if version.endswith("-initial"):
        return {"version": version, "status": "initial_in_progress"}
    return {"version": version, "status": "final_complete"}


def _verification_script() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
expected = {item["path"]: item for item in manifest["files"]}
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "MANIFEST.json"
}
if actual != set(expected):
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    raise SystemExit(f"file set mismatch; missing={missing}, extra={extra}")
for relative, item in expected.items():
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != item["size_bytes"] or digest != item["sha256"]:
        raise SystemExit(f"integrity mismatch: {relative}")
print(f"OK: {len(expected)} files verified for {manifest['package']} {manifest['version']}")
'''


def _primary_claim_verification_script() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
workbook = root / "02_raw_data/exp01_main_comparison.xlsx"


def verify_sheet(sheet: str, expected_rows: int, expected_benchmarks: set[str]) -> None:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    required = {
        "dataset",
        "metric",
        "p_value_holm",
        "bootstrap_ci_low",
        "tardis_better",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{sheet}: missing columns {missing}")
    if len(frame) != expected_rows:
        raise SystemExit(f"{sheet}: expected {expected_rows} rows, received {len(frame)}")
    if set(frame["metric"]) != {"TC", "LPIPS"}:
        raise SystemExit(f"{sheet}: unexpected primary metrics")
    expected_pairs = {
        (dataset, benchmark)
        for dataset in ("dataverse", "seedance", "openvid")
        for benchmark in expected_benchmarks
    }
    observed_pairs = set(zip(frame["dataset"], frame["benchmark"], strict=True))
    if observed_pairs != expected_pairs:
        raise SystemExit(f"{sheet}: benchmark/dataset coverage mismatch")
    if not frame["tardis_better"].astype(bool).all():
        failures = frame.loc[~frame["tardis_better"].astype(bool)]
        raise SystemExit(f"{sheet}: TARDIS does not lead all comparisons\\n{failures}")
    if not (pd.to_numeric(frame["bootstrap_ci_low"], errors="raise") > 0).all():
        raise SystemExit(f"{sheet}: at least one bootstrap interval does not support improvement")
    if not (pd.to_numeric(frame["p_value_holm"], errors="raise") < 0.05).all():
        raise SystemExit(f"{sheet}: at least one Holm-adjusted p-value is not significant")


verify_sheet(
    "paired_statistics",
    18,
    {"animatediff_lightning", "sd_turbo_independent", "text2video_zero"},
)
verify_sheet(
    "source50_statistics",
    54,
    {
        "animatediff_lightning",
        "controlvideo_canny",
        "rerender_flow",
        "sd_turbo_independent",
        "stablevideo_propagation",
        "streamdiffusion_img2img",
        "text2video_zero",
        "tokenflow_core",
        "vid2vid_zero_core",
    },
)
print("OK: prompt-only 18/18 and source-conditioned 54/54 primary comparisons verified")
'''


def _usage_guide_cn() -> str:
    fields = "\n".join(
        f"- `{name}`：{description}（{unit}）"
        for name, description, unit in FIELD_DICTIONARY
    )
    return f"""# 数据包使用说明（中文）

## 1. 快速开始

运行 `python 05_scripts/verify_package.py`。校验通过后先读 `README.md`，再按板块打开
`02_raw_data/*.xlsx`。空单元格表示尚未产生测量，不表示数值为零。

## 2. 工作簿

`exp01` 主对比；`exp02` 延时；`exp03` TC 专项；`exp04` 消融；`exp05` 鲁棒性；
`exp06` 效率；`exp07` 泛化；`exp08` 真实用户研究。每个文件包含 `_meta` 与 `summary`。

`paper50` 是 prompt-only 主协议；`source50` 是独立的 source-conditioned 协议，二者不混合。
source50 的 Rerender/TokenFlow/vid2vid-zero/ControlVideo/StableVideo 记录是核心机制复现，
不是官方仓库原码数值，具体范围见每个运行的 `run_manifest.json`。

## 3. 字段字典

{fields}

## 4. 复现与扩展

运行命令见 `05_scripts/README_scripts.md`。新增方法必须提供：方法配置、权重来源与许可、
生成适配器、完整 per-video ledger、环境和命令日志。所有方法共用同一个指标实现。

## 5. 常见问题

- `†`：论文报告值，非当前硬件与数据协议实测。
- `N/A`：协议不兼容或尚无可审计适配器。
- 延时与论文不同：必须同时核对 GPU、精度、分辨率、帧数、采样步数和 warmup。
- 指标无法配对：检查 record_id、seed、split hash 和失败样本是否完全一致。
- 主表主指标结论：只读取 `exp01` 的 `paired_statistics`，同时查看 bootstrap CI、Wilcoxon
  和 Holm 校正；不能用单次 pilot 均值宣称领先。
"""


def _usage_guide_en() -> str:
    return """# Data Package Usage Guide

Run `python 05_scripts/verify_package.py`, then inspect `README.md` and the `_meta` sheet in
each workbook. Empty cells mean not measured, never zero. Reproduction commands are listed in
`05_scripts/README_scripts.md`. A dagger denotes a paper-reported value under a different
protocol; N/A denotes an incompatible or unavailable audited adapter.
"""


def _method_descriptions() -> str:
    return """# Method Descriptions and Status

| Method | Role | Initial package status |
|---|---|---|
| TARDIS A10 | Ours | Full 512-video test measured on all three datasets |
| StreamDiffusion | Real-time image diffusion | Adapter required; no fabricated metric |
| StreamDiffusion img2img | Source-conditioned streaming | source50 pipeline reproduction |
| Rerender-A-Video | Source-conditioned video editing | source50 core-mechanism reproduction |
| TokenFlow | Source-conditioned feature propagation | source50 core-mechanism reproduction |
| vid2vid-zero | Source-conditioned zero-shot editing | source50 core-mechanism reproduction |
| Text2Video-Zero | Prompt-only video generation | Runnable and queued |
| ControlVideo | Source/control-conditioned editing | source50 Canny-core reproduction |
| StableVideo | Source-conditioned propagation | source50 core-mechanism reproduction |
| AnimateDiff-Lightning 2-step | Prompt-only motion adapter | Runnable proxy for AnimateDiff-LCM |
| SD-Turbo independent | Frame-independent trivial baseline | Runnable and queued |

Prompt-only and source50 values are never mixed. A core-mechanism reproduction is measured code
but is not an official repository reproduction; manuscripts must retain that qualifier. Official
repository values require an exact revision and environment record. Paper-reported values are
stored separately and marked with a dagger.
"""


def _hardware_environment(repo: Path) -> str:
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "python": [sys.executable, "--version"],
        "git": ["git", "-C", str(repo), "rev-parse", "HEAD"],
    }
    sections = []
    for name, command in commands.items():
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        payload = (result.stdout or result.stderr).strip()
        sections.append(f"## {name}\n\n```text\n{payload}\n```\n")
    return "# Hardware and Environment\n\n" + "\n".join(sections)


def _markdown_metric_rows(metrics: dict[str, dict[str, float]]) -> str:
    labels = {"dataverse": "DataVerse", "seedance": "Seedance", "openvid": "OpenVid"}
    rows = []
    for dataset in DATASETS:
        item = metrics[dataset]
        rows.append(
            f"| {labels[dataset]} | 512 | {item['tc']:.6f} | {item['lpips']:.6f} | "
            f"{item['fvd']:.4f} | {item['fid']:.4f} | {item['clipscore']:.6f} | "
            f"{item['ssim']:.6f} |"
        )
    return "\n".join(rows)


def _benchmark_markdown_rows(runs: list[dict[str, Any]]) -> str:
    frame = _benchmark_run_frame(runs)
    if frame.empty:
        return "当前尚无已完成的外部 benchmark 运行；正式队列可用且支持断点续跑。"
    lines = [
        "| 方法 | 数据集 | 协议 | 样本 | TC ↓ | LPIPS ↓ | FVD ↓ | FID ↓ | CLIP ↑ | SSIM ↑ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.dataset} | {row.protocol}/{row.metric_mode} | "
            f"{row.records} | {_format_metric(row.TC)} | {_format_metric(row.LPIPS)} | "
            f"{_format_metric(row.FVD)} | {_format_metric(row.FID)} | "
            f"{_format_metric(row.CLIP_score)} | {_format_metric(row.SSIM)} |"
        )
    return "\n".join(lines)


def _summary_markdown_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "当前尚无满足完整 seed 覆盖要求的正式运行。"
    lines = [
        "| 方法 | 数据集 | Seeds | 视频-Seed 对 | TC ↓ | LPIPS ↓ | FVD ↓ | "
        "FID ↓ | CLIP ↑ | SSIM ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.dataset} | {row.seed_runs} | {row.records} | "
            f"{_format_metric(row.TC)} | {_format_metric(row.LPIPS)} | "
            f"{_format_metric(row.FVD)} | {_format_metric(row.FID)} | "
            f"{_format_metric(row.CLIP_score)} | {_format_metric(row.SSIM)} |"
        )
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.6f}"


def _copy_benchmark_logs(
    output: Path,
    *,
    experiment_root: Path,
    runs: list[dict[str, Any]],
    namespace: str,
    reset: bool,
) -> None:
    destination_root = output / "06_logs/benchmark_runs"
    if reset:
        shutil.rmtree(destination_root, ignore_errors=True)
    for run in runs:
        source_root = Path(run["run_root"])
        relative = source_root.relative_to(experiment_root)
        destination = destination_root / namespace / relative
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (
            "metrics.json",
            "per_video.jsonl",
            "run_manifest.json",
            "failures.jsonl",
        ):
            source = source_root / filename
            if source.is_file():
                shutil.copy2(source, destination / filename)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prompt_id(dataset: str, record_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}\x1f{record_id}".encode()).hexdigest()[:12]
    return f"{dataset}_{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _indent(value: str, spaces: int) -> str:
    prefix = " " * spaces
    return "".join(
        prefix + line if line.strip() else line
        for line in value.splitlines(keepends=True)
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = initialize_package(args)
    print(path)


if __name__ == "__main__":
    main()
