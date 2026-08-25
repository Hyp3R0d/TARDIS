# Apply 成果生成接口

`scripts/apply.sh` 是纯 prompt 视频生成接口，不接收 source video。`TARDIS_DATASET` 用于选择
哪一套数据集专属权重；若未显式传入权重，程序只在对应的
`checkpoints/<dataset>/` 中寻找最新 `best.pt`。

## 启动命令

```bash
# 使用 DataVerse 权重与默认 prompt
bash scripts/apply.sh

# 使用 OpenVid 权重生成指定展示视频
TARDIS_DATASET=openvid \
TARDIS_PROMPT="A robot running in the forest" \
TARDIS_STYLE="cinematic, highly detailed" \
TARDIS_DURATION=2 \
bash scripts/apply.sh

# 显式指定权重
TARDIS_DATASET=seedance \
TARDIS_CHECKPOINT=/home/TARDIS/checkpoints/seedance/<时间戳>/best.pt \
TARDIS_PROMPT="A robot running in the forest" \
bash scripts/apply.sh
```

## 输出目录

```text
outputs/apply/<dataset>/<时间戳>/video.mp4
outputs/apply/<dataset>/<时间戳>/video.json
```

默认输出为 `512×512`。写入前会校验生成张量的帧数、宽高和有限性；不符合请求尺寸时直接
报错。`video.json` 记录数据集、prompt、style、权重 SHA-256、分辨率、帧数、随机种子以及
生成和编码耗时。

## 可调参数

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `TARDIS_DATASET` | `dataverse` | 选择数据集专属权重 |
| `TARDIS_CHECKPOINT` | 当前数据集最新 `best.pt` | 显式权重路径 |
| `TARDIS_PROMPT` | `A robot running in the forest` | 视频内容 prompt |
| `TARDIS_STYLE` | `cinematic` | 附加风格 |
| `TARDIS_DURATION` | `2` | 输出时长，单位秒 |
| `TARDIS_HEIGHT` / `TARDIS_WIDTH` | `512` | 输出分辨率 |
| `TARDIS_FPS` | `30` | 生成与 MP4 帧率 |
| `TARDIS_SEED` | `3407` | 采样种子 |
| `TARDIS_USE_EMA` | `1` | 使用 EMA 权重 |
| `TARDIS_PRECISION` | `bf16` | 推理精度 |
| `TARDIS_NPROC` | `1` | `torchrun` 本机进程数 |
| `TARDIS_DIFFUSION_TIME_SAMPLING` | `endpoint` | 与正式权重一致的扩散时间采样 |
| `TARDIS_TRANSPORT_HISTORY_FALLBACK_WEIGHT` | `1.0` | 历史状态回退权重 |
| `TARDIS_LITE_MAX_MAGNITUDE` | `0.75` | 轻量残差最大幅值 |
| `TARDIS_KEYFRAME_LITE_ALIGNMENT` | `1` | 启用关键帧轻量对齐 |
| `TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT` | `1` | 启用采样轨迹对齐 |

主网络结构参数与 Infer 相同，必须与所选权重训练时一致。改变输出分辨率不能弥补低分辨率
训练权重的能力，因此正式展示应使用同为 `512×512` 训练得到的权重。

当前无参生产默认值已经与正式交付 checkpoint 对齐；除 prompt、style、duration、seed 和输出
尺寸外，不建议在展示服务中覆盖上述模型轨迹参数。
