# Train 训练接口

`scripts/train.sh` 是正式训练入口，通过 `torchrun` 启动。一个训练进程只使用一个数据集，
由 `TARDIS_DATASET` 指定：`dataverse`、`openvid` 或 `seedance`。训练、验证和测试划分均来自
同一数据集，不会在一个进程中混合三源样本。

## 启动命令

```bash
# 默认训练 DataVerse
bash scripts/train.sh

# 分别训练另外两个数据集
TARDIS_DATASET=openvid bash scripts/train.sh
TARDIS_DATASET=seedance bash scripts/train.sh
```

每个数据集独立保存权重和运行记录：

```text
checkpoints/<dataset>/<时间戳>/latest.pt
checkpoints/<dataset>/<时间戳>/best.pt
outputs/train/<dataset>/<时间戳>/
```

这些项目目录是 `/root/autodl-tmp/TARDIS` 数据盘的符号链接。视频归档直接从本地 TAR、ZIP
或 MP4 读取，训练热路径不下载数据，也不解压形成重复副本。

`latest.pt` 默认每 256 个 micro-batch 在梯度累计边界原子更新，并在每个 epoch 结束时再次
更新，用于精确续训和宿主异常重启后的进度恢复；周期保存不会执行 validation，也不会更新
`best.pt`。`best.pt` 仅在当前数据集完整验证集的比赛加权分严格提高时更新。恢复训练时，
`TARDIS_DATASET` 必须和待恢复权重所属的数据集一致：

```bash
TARDIS_DATASET=openvid \
TARDIS_RESUME=/home/TARDIS/checkpoints/openvid/<时间戳>/latest.pt \
bash scripts/train.sh
```

## 验证与权重选择

每个 epoch 固定显示两个 tqdm：

1. 当前数据集训练进度条，显示 loss、学习率、课程阶段和显存；
2. 当前数据集完整验证进度条。

验证结束后只打印当前数据集的六项指标：TC、LPIPS、FVD、FID、CLIPScore 和 SSIM。六项均
计算和记录，但 `best.pt` 只由赛题明确指定的 TC 与 LPIPS 决定。去掉主观评分 20% 后，对
赛题的 50% TC 和 30% LPIPS 进行归一化：

| 指标 | 越优方向 | best 权重 |
|---|---:|---:|
| TC | 越低越好 | 0.625 |
| LPIPS | 越低越好 | 0.375 |
| FVD | 越低越好 | 0 |
| FID | 越低越好 | 0 |
| CLIPScore | 越高越好 | 0 |
| SSIM | 越高越好 | 0 |

TC 和 LPIPS 按当前数据集冻结目标尺度归一化，再计算
`0.625×(TC/TC目标) + 0.375×(LPIPS/LPIPS目标)`。当前协议固定目标为：DataVerse
`0.060/0.60`、OpenVid `0.070/0.60`、Seedance `0.100/0.60`。

完整 validation 上 TC 与 LPIPS 同时不高于对应阈值时，训练日志写
`target_pass=yes`，这就是当前内部 `TARDIS protocol-best` 判定。通过阈值的候选优先保存为
`best.pt`，其余候选按上述加权分排序。

test 不在每轮训练后运行。三个数据集的 validation 权重全部锁定后，再统一启动一次最终
test Infer，输出 test 指标、六项诊断指标和展示视频。test 指标只用于最终报告与泛化核验，绝不
进入训练、学习率调度、early stopping、checkpoint 选择或超参数搜索；多 seed、Pareto、
FVD、FID、CLIPScore、SSIM、速度和资源均不构成 validation SOTA 门槛。

## 可调超参数

Shell 脚本无需位置参数，以下环境变量可覆盖默认值：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `TARDIS_DATASET` | `dataverse` | 本进程唯一使用的数据集 |
| `TARDIS_EPOCHS` | `20` | 训练轮数 |
| `TARDIS_STEPS_PER_EPOCH` | `64` | 每轮 micro-batch 数 |
| `TARDIS_MICRO_BATCH_SIZE` | `2` | 每卡 micro-batch 视频数 |
| `TARDIS_GRADIENT_ACCUMULATION_STEPS` | `2` | 梯度累计步数 |
| `TARDIS_LEARNING_RATE` | `1e-4` | AdamW 峰值学习率 |
| `TARDIS_WEIGHT_DECAY` | `1e-2` | 权重衰减 |
| `TARDIS_WARMUP_STEPS` | `64` | warmup 优化器步数 |
| `TARDIS_CHECKPOINT_INTERVAL_STEPS` | `256` | 轮内周期保存间隔，单位为 micro-batch；实际保存点必须位于梯度累计边界 |
| `TARDIS_VALIDATION_INTERVAL` | `1` | 验证间隔 |
| `TARDIS_VALIDATION_BATCH_SIZE` | `8` | 验证批大小；与训练批大小解耦 |
| `TARDIS_GRADIENT_CLIP_NORM` | `1.0` | 梯度裁剪阈值 |
| `TARDIS_EMA_DECAY` | `0.999` | EMA 衰减 |
| `TARDIS_TC_LOSS_WEIGHT` | `5.0` | metric-alignment 阶段 TC 训练损失权重 |
| `TARDIS_LPIPS_LOSS_WEIGHT` | `3.0` | metric-alignment 阶段 LPIPS 训练损失权重 |
| `TARDIS_DIFFUSION_LOSS_WEIGHT` | `1.0` | 残差扩散目标权重 |
| `TARDIS_RESIDUAL_LOSS_WEIGHT` | `1.0` | 残差重建目标权重 |
| `TARDIS_TRANSPORT_LOSS_WEIGHT` | `1.0` | 运动传播目标权重 |
| `TARDIS_FLOW_LOSS_WEIGHT` | `0.1` | 光流监督权重 |
| `TARDIS_VISIBILITY_LOSS_WEIGHT` | `0.1` | 可见性监督权重 |
| `TARDIS_ROUTER_LOSS_WEIGHT` | `0.2` | innovation router 校准权重 |
| `TARDIS_SURVIVAL_LOSS_WEIGHT` | `0.2` | proper-time 生存校准权重 |
| `TARDIS_LITE_LOSS_WEIGHT` | `0.2` | 轻量残差分支权重 |
| `TARDIS_BUDGET_LOSS_WEIGHT` | `0.05` | active-token 预算权重 |
| `TARDIS_WARP_LOSS_WEIGHT` | `0.2` | 跨帧 warp 权重 |
| `TARDIS_DRIFT_LOSS_WEIGHT` | `0.1` | 长时漂移权重 |
| `TARDIS_CRCD_LOSS_WEIGHT` | `1.0` | 因果残差蒸馏权重 |
| `TARDIS_TEXT_LOSS_WEIGHT` | `0.1` | 文本对齐权重 |
| `TARDIS_LPIPS_FRAME_CHUNK_SIZE` | `4` | 可微 LPIPS 的分帧计算批大小 |
| `TARDIS_RESUME` | 空 | 当前数据集的 `latest.pt` |

主要数据、视频和运行参数：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `TARDIS_VALIDATION_SIZE` | `256` | 当前数据集验证记录数 |
| `TARDIS_TEST_SIZE` | `512` | 当前数据集测试记录数 |
| `TARDIS_SPLIT_SEED` | `3407` | 固定划分种子 |
| `TARDIS_NUM_WORKERS` | `8` | 本地读取与解码 worker 数 |
| `TARDIS_PREFETCH_FACTOR` | `4` | 每 worker 预取 batch 数 |
| `TARDIS_HEIGHT` / `TARDIS_WIDTH` | `512` | 训练分辨率 |
| `TARDIS_NUM_FRAMES` | `16` | 训练 clip 帧数 |
| `TARDIS_FPS` | `30` | 时间条件和输出帧率 |
| `TARDIS_PRECISION` | `bf16` | `bf16`、`fp16` 或 `fp32` |
| `TARDIS_NPROC` | `1` | `torchrun` 本机进程数 |

主网络参数可通过 `TARDIS_LATENT_CHANNELS`、`TARDIS_PATCH_SIZE`、`TARDIS_HIDDEN_SIZE`、
`TARDIS_NUM_LAYERS`、`TARDIS_NUM_HEADS`、`TARDIS_ACTIVE_RATIO`、
`TARDIS_TRANSPORT_QUOTIENT`、`TARDIS_QUOTIENT_REGULARIZATION`、
`TARDIS_QUOTIENT_RANK_THRESHOLD`、`TARDIS_INNOVATION_PROPER_TIME`、
`TARDIS_PROPER_TIME_MAXIMUM_HAZARD` 和 `TARDIS_DIFFUSION_STEPS` 调整。Infer 与 Apply 必须
使用与训练权重一致的结构参数。

正式交付主网络的默认轨迹配置为：`TARDIS_DIFFUSION_TIME_SAMPLING=endpoint`、
`TARDIS_TRANSPORT_HISTORY_FALLBACK_WEIGHT=1.0`、`TARDIS_LITE_MAX_MAGNITUDE=0.75`、
`TARDIS_KEYFRAME_LITE_ALIGNMENT=1`、`TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT=1`。三个 Shell
入口共享这组默认值；修改后训练得到的权重必须使用同一组参数进行 Infer 和 Apply。

`TARDIS_CATALOG_RECORD_LIMIT` 和 `TARDIS_OPENVID_ARCHIVE_LIMIT` 仅用于诊断；正式训练不要
设置，否则会改变样本空间。2026-08-10 的单卡 DataVerse 正式 20 轮实测使用 validation
batch 8，峰值 reserved VRAM 约 26.7 GB（约 81%），平均 GPU 利用率约 67%。前 16 轮训练段
约 3.5-4.6 分钟；进入 `metric_alignment` 后训练段约 11 分钟，完整 validation 约 6.8 分钟，
因此不能把该旧 run 的所有 epoch 宣称为 6-8 分钟。

2026-08-10 的验证集诊断发现旧训练从真实首帧起步，而 Apply/Infer 从 SD-Turbo prompt 首帧
起步，存在首状态分布错配。当前训练实现已改为：前三个教师阶段仍可使用真实首帧；随着
teacher-forcing 衰减，闭环、CRCD 和 metric-alignment 阶段逐步改用正式生成时的 prompt
首帧。新模型的 motion flow、visibility 和轻量残差输出采用 identity-preserving 初始化，
未经训练时严格复用 SD-Turbo 首帧。六阶段优化预算采用 `5%/5%/10%/20%/20%/40%`，将
大部分优化步用于闭环、CRCD 和 TC/LPIPS 对齐。旧 checkpoint 可继续 Infer，但因训练目标
签名变化，不能作为精确 resume checkpoint。
