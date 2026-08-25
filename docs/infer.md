# Infer 测试接口

`scripts/infer.sh` 一次只评测一个数据集。它完整遍历所选数据集的测试划分，对每条记录生成视频
并与 label 视频计算 TC、LPIPS、FVD、FID、CLIPScore 和 SSIM。全量结果参与指标，但默认只
把随机选择的 5 个生成结果编码为展示 MP4。

## 启动命令

自动加载当前数据集目录下最新的 `best.pt`：

```bash
bash scripts/infer.sh
TARDIS_DATASET=openvid bash scripts/infer.sh
TARDIS_DATASET=seedance bash scripts/infer.sh
```

指定权重时必须同时指定其数据集：

```bash
TARDIS_DATASET=openvid \
TARDIS_CHECKPOINT=/home/TARDIS/checkpoints/openvid/<时间戳>/best.pt \
bash scripts/infer.sh
```

若需要三个数据集的结果，应分别启动三次 infer；单次进程不会加载或评测另外两个数据集。

## 输出目录

```text
outputs/infer/<dataset>/<时间戳>/metrics.xlsx
outputs/infer/<dataset>/<时间戳>/metrics.csv
outputs/infer/<dataset>/<时间戳>/per_video_details.csv
outputs/infer/<dataset>/<时间戳>/per_video_details.jsonl
outputs/infer/<dataset>/<时间戳>/failures.jsonl
outputs/infer/<dataset>/<时间戳>/latency.json
outputs/infer/<dataset>/<时间戳>/resources.json
outputs/infer/<dataset>/<时间戳>/showcases/*.mp4
```

`metrics.xlsx` 和 `metrics.csv` 只包含当前数据集一行，例如 `openvid_test`。展示视频从当前
测试集成功记录中按 `TARDIS_SEED` 可复现地随机选取，默认恰好 5 个。其余生成结果只累计指标，
不会批量落盘为图片或视频。

Infer 是三个数据集 validation SOTA 权重锁定后的统一最终 test 报告步骤，不在每个候选训练
后重复执行。当前 validation 目标为 DataVerse `TC/LPIPS <= 0.060/0.60`、OpenVid
`0.070/0.60`、Seedance `0.100/0.60`；test 结果仅作为最终独立报告和泛化核验，不覆盖
validation 的 `target_pass` 判定，也不得回流训练或用于选择下一组超参数。

## 可调参数

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `TARDIS_DATASET` | `dataverse` | 本进程唯一评测的数据集 |
| `TARDIS_CHECKPOINT` | 当前数据集最新 `best.pt` | 显式权重路径 |
| `TARDIS_TEST_SIZE` | `512` | 当前数据集测试记录数 |
| `TARDIS_VALIDATION_SIZE` | `256` | 划分时保留的验证记录数 |
| `TARDIS_SHOWCASE_COUNT` | `5` | 随机保存的 MP4 数量 |
| `TARDIS_SEED` | `3407` | 生成与展示抽样种子 |
| `TARDIS_SPLIT_SEED` | `3407` | 必须与训练一致 |
| `TARDIS_NUM_WORKERS` | `8` | 本地读取与解码 worker 数 |
| `TARDIS_PREFETCH_FACTOR` | `4` | 每 worker 预取 batch 数 |
| `TARDIS_RESUME_METRICS` | `1` | 恢复逐记录指标状态 |
| `TARDIS_RESUME_OUTPUT` | 空 | 同一数据集的中断输出目录 |
| `TARDIS_USE_EMA` | `1` | 使用 EMA 权重 |
| `TARDIS_PRECISION` | `bf16` | 推理精度 |
| `TARDIS_NPROC` | `1` | `torchrun` 本机进程数 |
| `TARDIS_DIFFUSION_TIME_SAMPLING` | `endpoint` | 与正式权重一致的扩散时间采样 |
| `TARDIS_TRANSPORT_HISTORY_FALLBACK_WEIGHT` | `1.0` | 历史状态回退权重 |
| `TARDIS_LITE_MAX_MAGNITUDE` | `0.75` | 轻量残差最大幅值 |
| `TARDIS_KEYFRAME_LITE_ALIGNMENT` | `1` | 启用关键帧轻量对齐 |
| `TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT` | `1` | 启用采样轨迹对齐 |

分辨率、帧数和主网络结构参数也可覆盖，但必须与权重一致：

```text
TARDIS_PRETRAINED_MODEL
TARDIS_HEIGHT / TARDIS_WIDTH / TARDIS_NUM_FRAMES / TARDIS_FPS
TARDIS_LATENT_CHANNELS / TARDIS_PATCH_SIZE
TARDIS_HIDDEN_SIZE / TARDIS_NUM_LAYERS / TARDIS_NUM_HEADS
TARDIS_ACTIVE_RATIO / TARDIS_DIFFUSION_STEPS
TARDIS_TRANSPORT_QUOTIENT
TARDIS_QUOTIENT_REGULARIZATION / TARDIS_QUOTIENT_RANK_THRESHOLD
TARDIS_INNOVATION_PROPER_TIME / TARDIS_PROPER_TIME_MAXIMUM_HAZARD
```

上述五个生产默认值与正式交付的三份 EMA checkpoint 一致。覆盖其中任意一项会形成不同的
推理协议，所得指标不能与本项目正式 test 报告直接比较。
