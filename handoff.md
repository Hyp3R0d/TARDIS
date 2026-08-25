# TARDIS 项目交接与 SOTA 迭代手册

更新时间：2026-08-17 17:37 UTC
交接对象：后续实验、复核和论文撰写 Agent

本文是当前项目的技术事实源。后续 Agent 应先读本文件、`README.md`、
`appendix/开发prompt.txt`、`appendix/创新点.md`、`docs/datasets.md` 和交付包 README。
代码与真实实验产物优先于宣传性描述；没有对应 raw ledger、日志和校验结果的结论，不得写成
“已达到 SOTA”。

---

## 当前交付状态（最终可审计快照）

论文写作交付包已经完成并冻结：

```text
/home/TARDIS/RTVD-TC-DataPackage-v1.1
```

包状态为 `VERSION=1.1`、`MANIFEST.status=final_complete`，登记文件数为 `1244`（不含
`MANIFEST.json` 自身）。`v1.0` 保留为历史冻结快照；`v1.1` 是论文写作的当前权威包。
增量实验使用独立 validation-50 在四个候选强度、五个 seed 上完成 `20/20` 个选择单元，
随后在冻结 test-50 上完成 `5/5` 个六指标运行，失败数均为 `0`。

六个真实队列均已完成，失败数均为 0：

| 队列 | 完成/预期 | 状态 |
|---|---:|---|
| prompt | 60/60 | completed |
| ablation | 11/11 | completed |
| source | 105/105 | completed |
| source_prompt_baselines | 45/45 | completed |
| source_ablation | 11/11 | completed |
| source_diagnostics | 150/150 | completed |

最终校验证据：

```text
python RTVD-TC-DataPackage-v1.1/05_scripts/verify_package.py
OK: 1244 files verified for RTVD-TC-DataPackage-v1.1 1.1

python RTVD-TC-DataPackage-v1.1/05_scripts/verify_primary_claims.py
OK: prompt-only 18/18 and source-conditioned 54/54 primary comparisons verified

python -m pytest -q tests/unit/experiments
40 passed in 4.23s

ruff check tardis/experiments tests/unit/experiments
All checks passed!

python -m tardis.experiments.audit
status: complete; all six queues valid; failed=0
```

## 结果边界（必须保留）

当前正式 test 指标来自三个 validation-only 选择的 EMA 权重，三个数据集各 `512/512`：

| 数据集 | TC | LPIPS | FVD | FID | CLIPScore | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| DataVerse | 0.036161 | 0.596104 | 24.9195 | 342.7548 | 0.243508 | 0.162174 |
| Seedance | 0.078698 | 0.591564 | 49.8675 | 360.1857 | 0.222302 | 0.132235 |
| OpenVid | 0.037631 | 0.567961 | 32.8612 | 382.6492 | 0.225873 | 0.037433 |

主协议是 prompt-only：TARDIS、SD-Turbo、AnimateDiff-Lightning、Text2Video-Zero 在同一
记录、同一 seed、同一指标实现下完成 `60` 个运行，TC/LPIPS 配对统计为 `18/18` 胜出。
该结论不扩展到 FVD、FID、CLIPScore、SSIM，也不代表所有 benchmark 的六指标全面胜出。

source-conditioned 协议独立记录 `300` 个运行，source video 同时作为条件和参考。比较器保持
冻结的 `source_strength=0.45`；TARDIS 的 Seedance 强度由独立 validation-50 在
`{0.30, 0.35, 0.40, 0.45}` 上按赛题 TC/LPIPS 权重选择，锁定为 `0.30`，选择过程未读取
test。Seedance 正式 source50 test 的六指标为：TC `0.035088`、LPIPS `0.140594`、FVD
`6.353786`、FID `49.825265`、CLIPScore `0.259705`、SSIM `0.849460`。

source50 的主指标配对统计现为 `54/54` 胜出，即三个数据集 × 九个 benchmark × TC/LPIPS；
所有 bootstrap 区间下界均大于 0，单侧 Wilcoxon 经 Holm 校正后均小于 `0.05`。该结论仅限
TC 和 LPIPS，不能扩展为 FVD、FID、CLIPScore、SSIM 或语义编辑质量的全面领先。验证选择
证据见包内 `01_configs/source_strength_selection.json`，原始 test ledger 位于
`06_logs/benchmark_runs/source/`。

Rerender-A-Video、TokenFlow、vid2vid-zero、ControlVideo、StableVideo 的 source 实现均标记
为 `audited core-mechanism reproduction`，不是官方仓库原码结果。`exp08_user_study.xlsx`
是 `planned` 模板，未执行真实用户研究；不得补写或合成用户票。当前没有证据证明所有配置
满足 `33.3 ms/frame`，延时必须按表中实测值报告。

## 交付包结构与入口

```text
RTVD-TC-DataPackage-v1.1/
├── README.md
├── MANIFEST.json
├── VERSION
├── 00_docs/                 # 中文说明、方法范围、硬件与变更记录
├── 01_configs/              # 全局参数、方法、prompt、split、环境快照
├── 02_raw_data/             # exp01-exp08 XLSX 原始/汇总数据
├── 03_figures/              # 已有真实测量支持的 PDF/PNG 图
├── 04_tables/               # table01-table08 XLSX/LaTeX
├── 05_scripts/              # 重跑、导出、分析、制图、校验脚本
└── 06_logs/                 # benchmark run manifest、metrics、ledger 和日志
```

使用交付包：

```bash
cd /home/TARDIS/RTVD-TC-DataPackage-v1.1
python 05_scripts/verify_package.py
python 05_scripts/verify_primary_claims.py
```

主指标 raw 配对值从 `02_raw_data/exp01_main_comparison.xlsx` 的
`paper50_runs`、`paired_statistics`、`source50_*` sheets 获取；source 时序诊断从
`exp03` 的 `source_*` sheets 获取；所有运行的原始 JSONL 位于 `06_logs/benchmark_runs/`。

## 复核与继续实验

重新审计当前工程：

```bash
cd /home/TARDIS
python -m tardis.experiments.audit
```

重跑单个 benchmark 单元时必须保持协议、split、seed、记录数和指标实现一致，并让新输出
进入独立目录；禁止覆盖已冻结包而不刷新 `MANIFEST.json`。刷新包并校验：

```bash
python -m tardis.experiments.package \
  --output /home/TARDIS/RTVD-TC-DataPackage-v1.1 \
  --source-root /home/TARDIS/TARDIS_SOURCE_EXPERIMENTS_V11 \
  --source-selection /root/autodl-tmp/TARDIS/TARDIS_SOTA/work/experiments/seedance_source_strength_selection_20260817/selection_result.json \
  --release-version 1.1 --refresh
python RTVD-TC-DataPackage-v1.1/05_scripts/verify_package.py
python RTVD-TC-DataPackage-v1.1/05_scripts/verify_primary_claims.py
```

训练、推理、应用接口的正式说明仍以仓库 `docs/train.md`、`docs/infer.md`、`docs/apply.md`
和 `README.md` 为准；三个正式 Shell 入口均使用 `torchrun`，权重、数据和视频在数据盘并
通过符号链接暴露到仓库。

---

## 历史设计与实验记录

以下章节保留先前 Agent 的设计、调优和实验日志，仅作为历史记录。若与本节的最终交付状态
冲突，以本节、交付包 `README.md`、`MANIFEST.json` 和当前代码为准。

---

## 0. 项目目标与不可变约束

### 0.1 项目名称

**TARDIS：Temporal Adaptive Residual Diffusion for Streaming Video**
中文：**面向流式视频的时序自适应残差扩散**

核心口号：

> Transport the predictable world; diffuse only unpredictable events.

中文解释：先传输可预测的生成状态，再只对不可由传输解释的创新进行扩散。

### 0.2 当前优化目标

当前版本只把 **TC** 和 **LPIPS** 作为主要优化目标与 checkpoint 选择依据：

```text
TC      权重 0.625，越低越好
LPIPS   权重 0.375，越低越好
```

0.625/0.375 是将赛题客观项 TC 50%、LPIPS 30% 去掉主观 20% 后重新归一化得到的比例。
FVD、FID、CLIPScore、SSIM 仍由 Infer 计算和记录，用于诊断和展示，但当前不参与 `best.pt`
选择。完整 validation 只负责选择权重：validation `target_pass` 优先，其余候选按固定加权分
排序。当前协议将三个数据集的 LPIPS 目标统一锁定为 `0.60`，TC 目标仍分别为 DataVerse
`0.060`、OpenVid `0.070`、Seedance `0.100`。完整 validation 同时满足当前数据集的 TC
与 LPIPS 目标时，即标记该数据集的内部 `TARDIS protocol-best`；test 不再作为训练期间的
SOTA 判定面。三个数据集 validation 权重全部锁定后，才统一启动最终 test Infer。test 只
用于最终报告和泛化核验，不参与训练、调度、early stopping、权重选择或超参数搜索，也不附加
多 seed、Pareto、其余指标或资源门槛。

本段是当前协议的最高优先级，后文历史实验记录中出现的 `LPIPS=0.30/0.32`、
`protocol_sota` test 判定和“每个数据集达标后立即 test”均为历史规则，不得用于新实验。

### 0.3 必须保持的接口约束

1. **Train**：一个进程只训练一个数据集的 train split，并在同一数据集的 validation split
   上验证和选权重。
2. **Infer**：一个进程只完整评测一个数据集的 test split；不能在单个进程混合三个数据集，
   不能计算跨数据集平均。
3. **Apply**：纯 prompt-to-video；不接收 source video。输入 prompt、style、时长和权重，
   输出 MP4。
4. 三个数据集的权重、训练输出和 Infer 输出必须物理命名空间隔离。
5. Train、Infer、Apply 的正式入口都使用 `torchrun` Shell 脚本。
6. 正式数据、权重和视频位于数据盘 `/root/autodl-tmp/TARDIS`；仓库中的 `data`、
   `checkpoints`、`outputs` 是符号链接。
7. 训练/评测热路径只读取本地 manifest 和本地归档，不访问远端；HF mirror 只用于显式下载
   和准备阶段。
8. 512x512 是正式质量目标。30 FPS、单帧小于 33.3 ms 是必须测量的竞赛约束，当前不能
   在没有实测日志时宣称已经满足。
9. 训练过程中每个 epoch 必须有一个 train tqdm 和一个 validation tqdm；验证进度条结束后
   输出当前数据集的六项诊断指标。

---

## 1. 仓库结构与模块边界

```text
/home/TARDIS/
├── appendix/
│   ├── 赛题要求.md                 赛题原始要求
│   ├── 赛题要求.pdf                赛题 PDF
│   ├── 开发prompt.txt              当前工程约束与验收规则
│   ├── 创新点.md                   TARDIS 主方法设计稿
│   └── 参考搜索范围.md              相关工作与检索边界
├── docs/
│   ├── datasets.md                 本地数据下载、manifest、划分
│   ├── train.md                    Train 参数与恢复说明
│   ├── infer.md                    Infer 参数、指标和产物
│   └── apply.md                    Apply 参数和 MP4 产物
├── scripts/
│   ├── download_datasets.sh        下载并准备本地数据盘内容
│   ├── train.sh                    torchrun Train 终极入口
│   ├── infer.sh                    torchrun Infer 终极入口
│   └── apply.sh                    torchrun Apply 终极入口
├── tardis/
│   ├── cli/
│   │   ├── common.py               公共 argparse 默认值和参数协议
│   │   ├── runtime.py              设备、先验、权重隔离和运行时组装
│   │   ├── train.py                训练、验证、进度、checkpoint、恢复
│   │   ├── infer.py                全量测试、指标、断点和展示视频
│   │   ├── apply.py                纯 prompt 生成和 MP4 编码
│   │   └── generation.py            生成结果/尺寸/输出辅助逻辑
│   ├── data/
│   │   ├── contracts.py            VideoRecord、异常和数据协议
│   │   ├── catalog.py              三源规格、本地根目录和 manifest
│   │   ├── adapters.py             DataVerse/OpenVid/Seedance 适配器
│   │   ├── archives.py             TAR/ZIP 本地索引与有界读取
│   │   ├── http_range.py            有界 Range 读取基础设施
│   │   ├── prepare_local.py        生成本地 manifest
│   │   ├── assembly.py             catalog、稳定划分和 DataLoader
│   │   ├── dataset.py              clip iterable dataset 和解码
│   │   ├── sampler.py              无状态、可恢复、DDP 可分片采样
│   │   └── video.py                视频记录/解码工具
│   ├── models/
│   │   ├── tardis.py               TARDISModel 总网络和 causal rollout
│   │   ├── priors.py               冻结 SD-Turbo VAE、文本编码和首帧先验
│   │   ├── motion.py               训练 motion teacher 与 prompt motion scaffold
│   │   ├── transport.py             latent/state motion warp
│   │   ├── quotient.py              Transport-Orbit Quotient 投影/分解
│   │   ├── router.py                风险/创新概率和 token budget
│   │   ├── clock.py                 Innovation Proper Time / hazard 累积
│   │   ├── residual.py              LiteCorrector 与 SparseResidualDiT
│   │   ├── state.py                 short state、anchor memory、scene reset
│   │   ├── factory.py               先验加载和 TARDIS 组装
│   │   ├── contracts.py             motion teacher 接口
│   │   └── ...                      transport、prior 等辅助模块
│   ├── training/
│   │   ├── curriculum.py            六阶段 curriculum 和 teacher forcing
│   │   ├── objective.py             rollout 后的多目标损失
│   │   ├── losses.py                TC/LPIPS/transport/router 等损失
│   │   ├── engine.py                AMP、EMA、optimizer、scheduler、恢复
│   │   ├── validation.py             验证聚合和 best selector
│   │   └── distillation.py          CRCD 因果残差蒸馏
│   ├── metrics/
│   │   ├── paired.py                官方 TC、LPIPS、SSIM、CLIPScore
│   │   ├── frechet.py               FID/FVD
│   │   ├── features.py              AlexNet LPIPS、Inception、I3D、CLIP
│   │   ├── suite.py                 流式六指标聚合
│   │   └── report.py                CSV/XLSX 单数据集报告
│   └── utils/
│       ├── checkpoint.py             原子保存和 temporal state 校验
│       ├── runtime.py                资源监控、随机种子、分布式辅助
│       └── video_io.py               MP4 读写和尺寸校验
├── tests/                            单元、集成、模型机制和分布式测试
├── datasets.txt                      三个本地数据源的固定路径
├── README.md
└── handoff.md                         本文件
```

`worklist.md` 已按用户要求移除；后续连续工作以本文件和代码中的测试为交接依据，不要为了
形式恢复旧清单。

---

## 2. 数据组织、记录契约与固定划分

### 2.1 本地路径

```text
/home/TARDIS/data         -> /root/autodl-tmp/TARDIS/datasets
/home/TARDIS/checkpoints  -> /root/autodl-tmp/TARDIS/checkpoints
/home/TARDIS/outputs      -> /root/autodl-tmp/TARDIS/outputs
```

`datasets.txt` 当前列出：

```text
/home/TARDIS/data/Vchitect_T2V_DataVerse
/home/TARDIS/data/OpenVid-1M
/home/TARDIS/data/seedance-2-prompts-datasets
```

规范化后的 canonical source 名称为：

```text
dataverse
openvid
seedance
```

每条记录是一个 caption/video 配对：

```python
VideoRecord(
    id: str,
    caption: str,
    media_locator: str,
    source: str,
    metadata: dict,
)
```

训练 batch 的核心形式为 `prompts: list[str]` 和 `video: [B,T,3,H,W]`。当前 Apply 不接受
`VideoRecord`、source video 或 label；它只把外部 prompt 与 style 合并后生成视频。

### 2.2 已准备的数据规模

三源本地 manifest 均为 8,000 个唯一 prompt-video 对，有效媒体体积各约 45 GB。默认
`validation_size=256`、`test_size=512`、`split_seed=3407` 时：

```text
dataset    train    validation    test
dataverse   7,232       256         512
openvid     7,232       256         512
seedance    7,232       256         512
```

TAR/ZIP 保持归档形态，manifest 负责索引，热路径只做有界本地读取，不解压出第二份完整媒体。
精确 revision、归档清单和有效字节数见 `docs/datasets.md`。

### 2.3 数据隔离规则

- `TARDIS_DATASET` 是每个 Train/Infer 进程的唯一数据源选择。
- `build_remote_catalog(..., selected_source=...)` 只 materialize 所选 source 的记录。
- 训练、validation、test loader 的 key 集必须恰好等于所选 source。
- 三个 split 使用相同的稳定分割算法和 `split_seed`；不能在调参时改变 test split。
- 正式实验禁止设置 `TARDIS_CATALOG_RECORD_LIMIT` 或 `TARDIS_OPENVID_ARCHIVE_LIMIT`，这两个
  参数只用于诊断。
- 任何新方案必须先报告 source/id 交集为零，并检查 manifest hash，防止 train/test 泄漏。

---

## 3. 三个接口的真实运作原理

### 3.1 Train 数据流

```text
TARDIS_DATASET
    -> datasets.txt / 本地 manifest
    -> selected_source catalog
    -> stable train/validation split
    -> train DataLoader
    -> TARDISTrainingBatch(prompts, video)
    -> frozen VAE encode + prompt encode
    -> FlowMotionTeacher (训练监督)
    -> TARDIS causal teacher-forced rollout
    -> staged objective / AMP / EMA / scheduler
    -> validation metrics
    -> dataset-local latest.pt + best.pt
```

一个预算 epoch 默认使用 `64 steps/epoch`、micro batch `2`、gradient accumulation `2`；
训练 loader 仍只从当前所选数据集采样，且完整可恢复。validation batch 默认 `8`，验证间隔
默认每个 epoch 一次，并完整遍历当前数据集的 256 条 validation。
Train 进程不会因为其它两个数据集存在于 `datasets.txt` 就加载它们。

每个 epoch 输出两个 tqdm：

```text
Epoch k/20 train       ...
Epoch k/20 validation  ...

Epoch k/20 DataVerse validation: weighted_score=... best.pt=updated
metric              DataVerse
TC                  ...
LPIPS               ...
FVD                 ...
FID                 ...
CLIPScore           ...
SSIM                ...
```

### 3.2 Infer 数据流

```text
指定 dataset + checkpoint
    -> 只组装该 dataset 的 512 条 test records
    -> 逐条 prompt-only model.generate()
    -> 与对应 label video 配对计算 TC/LPIPS/FVD/FID/CLIPScore/SSIM
    -> 只在成功记录中按 seed 选择 5 个 showcase
    -> 输出一行当前 dataset_test 的 CSV/XLSX
```

所有测试记录参与指标；只有 5 条生成视频落盘，避免把整个测试集写成大量 MP4。Infer 输出
不做跨数据集平均。三个数据集的结果必须通过三个独立 Infer 进程获得。

### 3.3 Apply 数据流

```text
prompt + style
    -> effective_prompt
    -> 当前 dataset 的 best.pt / 显式 checkpoint
    -> 首帧 latent
    -> causal frame-by-frame TARDIS rollout
    -> frozen VAE decode
    -> 尺寸/帧数校验
    -> outputs/apply/<dataset>/<timestamp>/video.mp4
```

Apply 的 `dataset` 只用于选择哪一套权重和命名输出，不会读取测试 label，也不接收 source
video。输出 sidecar `video.json` 记录 prompt、style、checkpoint SHA256、帧数、分辨率、
延迟和采样设置。

---

## 4. TARDIS 主网络：从输入到输出

### 4.1 冻结语义先验

`FrozenPriorBundle` 从 `stabilityai/sd-turbo` 加载并冻结：

- VAE encoder：训练视频转成 latent；
- VAE decoder：latent 转回 `[-1,1]` RGB 视频；
- CLIP text conditioner：prompt token；
- first-frame generator：Apply/Infer 的首帧生成。

checkpoint 只保存 temporal TARDIS state，不允许把 `priors.*` 写进 temporal checkpoint。这样
不同数据集可以共享同一冻结先验，同时各自保存独立的 temporal 权重。

### 4.2 PromptMotionScaffold

`PromptMotionScaffold` 使用文本 token、上一时刻 causal short state、时间编码和随机 motion
noise，预测：

- backward flow；
- visibility logits；
- motion tokens。

训练阶段的 `FlowMotionTeacher` 使用视频帧估计 flow/visibility，仅作为监督或 warmup 条件；
部署阶段 `generate()` 不调用 teacher，也不需要 source video。这一点是当前 prompt-only 接口
的关键边界。

### 4.3 MotionStateTransport

把上一帧生成 latent 和有限 causal state 按预测 backward flow 做 `grid_sample` warp，结合
visibility 和 bounded flow correction 得到：

```text
transport.prior       当前帧可复用 latent 先验
warped_latent         对齐后的历史 latent
warped_state          对齐后的 short/anchor/hazard 状态
effective_visibility  有效可见置信度
valid_mask            采样边界有效区域
```

其角色是“预测可复用世界”，不是直接生成完整当前帧。

### 4.4 Transport-Orbit Quotient（TOQ）

`TransportOrbitProjector` 从 transport prior 的局部空间梯度构造正则化 transport orbit basis，
把残差分解为：

```text
tangent       可由局部运动轨道解释的确定性变化
innovation    不能由 transport 解释的法向变化
```

数学目标：

\[
R_t^{TAR}=z_t-\operatorname{sg}(\bar z_t),\qquad
R_t^\perp=(I-P_{J_t})R_t^{TAR}.
\]

代码中使用正则化和 rank threshold 处理退化 basis；不能把普通空间 active mask 误称为完整
创新子空间。

### 4.5 Risk router 与 Innovation Proper Time

`VisibilityCalibratedInnovationRouter` 融合 prior、visibility、flow、short state 和 text，
预测 pixel-level innovation probability，再按 `active_ratio` 选择 patch token，并加 halo。
训练时可由 target latent 生成 oracle innovation probability，采用 router/survival/budget
损失逼近可部署预测。

`InnovationProperTime` 把历史 hazard、当前风险概率和可见性变成：

- instantaneous hazard；
- accrued hazard；
- event probability；
- patch probability；
- active selection/service mask；
- settled hazard。

它把“每帧都算满”改成“事件发生时分配有限随机计算预算”。场景切换时由
`scene_cut_threshold` 触发 state reset。

### 4.6 双频残差更新

`LiteResidualCorrector` 在全 latent 网格上做有界、低频、轻量的切向修正；其输出受
`lite_max_magnitude` 限制。

`SparseResidualDiT` 只 gather router 选择的 active patch token。输入包括：

- noisy innovation residual；
- transported prior；
- diffusion time；
- event probability/proper time；
- text tokens；
- motion tokens；
- anchor state tokens。

网络是带 AdaLN-Zero block 和条件 cross-attention 的 sparse residual DiT，输出散射回完整 latent
网格。最终状态转移是：

\[
\hat z_t
=\bar z_t
+P_t^\parallel C_\omega(\mathcal C_t)
+P_{\mathcal A_t}P_t^\perp R_\theta(\mathcal C_t,\epsilon_t).
\]

代码对应：

```text
latent = transport.prior + lite_residual + sparse_residual
```

### 4.7 CausalStateUpdater

每帧只保留常量大小的：

- current latent；
- short feature state；
- confidence-weighted EMA anchor；
- innovation hazard；
- frame index。

新创新区域更新 short state，稳定区域更多复用历史 anchor；scene cut 时清空 hazard 并重置
short/anchor。推理使用 detach state，避免长视频反向图无限增长。

### 4.8 A10 与消融

`AblationVariant.A10` 是当前完整版本，依次打开：

```text
A1 previous-frame conditioning
A2 temporal residual
A3 source-motion transport
A4 analytical visibility
A5 learned VCIR
A6 dual-frequency residual
A7 fixed-budget routing
A8 innovation proper time
A9 CRCD
A10 metric alignment
```

消融必须在相同 prior、分辨率、帧数、seed、训练预算和推理步数下完成。最关键对照是 A0
逐帧/无时序基线 vs A10 完整模型；不能只比较不同模型大小。

---

## 5. 训练目标与优化机制

### 5.1 Prior-anchored scheduled rollout

`forward_train()`：

1. VAE encode 整段标注视频得到 `target_latents`；
2. 文本编码得到 text tokens；
3. `FlowMotionTeacher` 估计每个相邻帧的 backward flow/visibility；
4. 前期教师阶段从首帧 target latent 初始化；随着 teacher forcing 衰减，逐步切换到与
   Infer/Apply 完全一致的 SD-Turbo prompt 首帧；
5. 逐帧调用 `transition()`；
6. 早期 curriculum 使用 oracle flow/visibility/oracle routing，之后逐步切换到模型预测；
7. 首状态与后续状态均按 teacher-forcing mask 切换，消除“训练看真实首帧、推理看生成
   首帧”的状态分布错配，最后形成闭环训练。

### 5.2 Curriculum 六阶段

`CurriculumSchedule` 按 optimizer step 累计经过：

```text
transport_warmup  -> router_calibration -> residual_teacher
-> closed_loop     -> crcd              -> metric_alignment
```

默认行为：

- 前三阶段 teacher forcing ratio 为 1；
- closed_loop 从 1 逐步下降到约 0.25；
- CRCD 阶段从约 0.25 下降到 0；
- metric_alignment 阶段为 0；
- 六阶段 optimizer-step 预算为 `5%/5%/10%/20%/20%/40%`，优先保障闭环、CRCD 与指标
  对齐，而不是六阶段平均分配；
- residual teacher/closed loop 使用 4 个 residual diffusion steps 的训练教师轨迹；
- CRCD/metric alignment 使用 1 个 residual step 的学生路径。

### 5.3 损失

`TARDISObjective` 先计算候选损失，再用可恢复的 EMA normalizer 归一化，最后按
`LossWeights` 加权。当前默认原始权重为：

```text
diffusion 1.00    residual 1.00      transport 1.00
flow      0.10    visibility 0.10    router 0.20
survival  0.20    lite 0.20          lpips 3.00
tc        5.00    warp 0.20          text 0.10
budget    0.05    drift 0.10         crcd 1.00
```

候选损失包括：

- diffusion / residual reconstruction：法向创新预测；
- transport / warp：运动先验与可见区域复用；
- flow / visibility：motion scaffold 监督；
- router / survival / budget：风险概率、hazard 和 active budget 校准；
- lite：切向低频修正；
- drift：闭环 latent 速度变化稳定；
- CRCD：student residual 对齐 causal teacher residual；
- LPIPS / multi-scale TC：最终 decoded video 的感知与运动差分约束；
- text：文本和视觉 latent token 对齐。

训练损失的权重不等于 checkpoint 选择权重。后者只看 validation TC/LPIPS。

### 5.4 优化、EMA、恢复

训练引擎负责：

- DDP/torchrun；
- AMP（bf16/fp16/fp32）；
- gradient accumulation、clip、非有限梯度记录；
- optimizer 和 learning-rate scheduler；
- EMA；
- 每 epoch `latest.pt`；
- 只在 validation score 严格改善时原子更新 `best.pt`；
- 保存随机数、optimizer、scheduler、normalizer、curriculum、distiller 和梯度状态，以支持
  精确恢复。

恢复 checkpoint 必须属于当前 `checkpoints/<dataset>/<run_id>/`，跨数据集路径会被拒绝。

---

## 6. 指标与 checkpoint 选择协议

### 6.1 生产指标定义

`tardis/metrics/paired.py` 中：

```text
TC     mean |(Y[t+1]-Y[t]) - (X[t+1]-X[t])| over normalized RGB
LPIPS  framewise AlexNet LPIPS v0.1, lower is better
```

TC 是赛题给出的官方帧差分损失，不是“相邻生成帧自身的平滑度”。如果只让输出静止，TC
可能下降但 LPIPS 和视觉质量会恶化，因此实验必须同时报告两者及静态/复制基线。

MetricSuite 以流式方式逐视频更新，保存 macro 和 micro 状态；正式报告使用当前 dataset test
的 macro 结果。所有指标的 provenance 会写进 manifest。

### 6.2 验证集选权重

`ValidationCheckpointSelector` 只接受一个 `*_validation` source，拒绝 test source 和多源
输入。TC 和 LPIPS 按当前数据集冻结目标尺度归一化，之后：

```text
score = 0.625 * normalized(TC) + 0.375 * normalized(LPIPS)
```

达标候选优先于未达标候选；达标状态相同时才比较上述加权分。FVD/FID/CLIPScore/SSIM
权重为零，只作为验证显示。`best.pt` 不能由 test 结果选择，不能根据 test 反复调参后再报
成绩。

### 6.3 当前内部 SOTA 目标（不是已取得结论）

三个数据集没有与本项目完全相同协议的公开排行榜，下面是工程冲榜目标，不是外部论文 SOTA
数字：

| 数据集 | 目标 TC | 目标 LPIPS |
|---|---:|---:|
| DataVerse | <= 0.060 | <= 0.60 |
| OpenVid | <= 0.070 | <= 0.60 |
| Seedance | <= 0.100 | <= 0.60 |

每个数据集只看本行两个阈值是否同时满足，不使用三数据集平均掩盖单源失败，也没有额外
stretch 目标。阈值只能在固定 manifest、固定 split、固定 metric provenance、无泄漏的
前提下使用。历史旧版联合训练的 validation 数值只可作为退化诊断，不能当作当前版本测试
结论。

---

## 7. SOTA Agent 的推荐工作流

后续 Agent 的第一职责不是立刻改网络，而是产出一份可执行、可证伪的 SOTA 方案，按下面顺序
交付。

### Phase 0：协议锁定和泄漏审计

1. 读取本文件与 `appendix/创新点.md`，确认 prompt-only Apply 边界。
2. 对三个 manifest 计算 record id、media hash、caption hash 的 train/validation/test 交集，
   必须为空。
3. 固定 `split_seed=3407`、`test_size=512`、`validation_size=256`，生成一份不可变实验
   manifest。
4. 验证 TC 的输入范围、帧数、颜色归一化和 macro aggregation；验证 LPIPS 的 AlexNet
   权重 provenance。
5. 建立四个诊断基线：

   ```text
   identity/label upper bound（仅指标 sanity check，绝不用于成绩）
   static-video TC lower-risk baseline
   first-frame/repeated-frame baseline
   independent SD-Turbo frame baseline
   ```

   这些基线用于识别 TC 静态化作弊和 LPIPS 尺度问题。

### Phase 1：先测假设，再大规模训练

在每个数据集 validation 上记录：

```text
raw frame residual energy
TAR residual energy
quotient-normal residual energy
tar_to_raw_ratio
quotient_to_tar_ratio
tangent_explained_ratio
router calibration / ECE / Brier
active token ratio
```

必须验证：

\[
E[\|r^{mc}\|_1]/E[\|r^{raw}\|_1] < 1,
\qquad
E[\|r^\perp\|_1] < E[\|r^{TAR}\|_1].
\]

若不成立，优先修正 flow/latent alignment 或 quotient basis，不要靠增加层数掩盖机制失败。

### Phase 2：建立公平 baseline 与消融矩阵

所有 baseline 使用相同 frozen prior、分辨率、帧数、seed、训练样本和延迟预算：

```text
B0  SD-Turbo 独立帧/无 causal reuse
B1  previous-state reuse，无运动对齐
B2  motion warp + full residual diffusion
B3  motion warp + spatial mask residual
B4  TARDIS TAR，无 TOQ
B5  TARDIS TAR + TOQ，无 risk/hazard
B6  TARDIS + risk router，无 sparse DiT
B7  TARDIS + sparse residual，无 CRCD
B8  TARDIS + CRCD，无 metric alignment
A10 完整 TARDIS
```

每个条目至少报告 TC、LPIPS、active ratio、参数量、显存峰值、p50/p95 ms per frame。关键
实验必须保持等计算预算，否则不能把速度/指标差异归因于方法机制。

### Phase 3：按数据集独立调参

三个数据集分别建立权重和搜索记录，推荐搜索顺序：

1. `active_ratio`：先在 `0.15/0.25/0.35/0.50` 中比较 TC/LPIPS 加权分；
2. `quotient_regularization` 与 `quotient_rank_threshold`：检查 quotient energy 是否真的
   下降；
3. `proper_time_maximum_hazard`、router threshold、halo radius：平衡新事件覆盖与预算；
4. teacher-forcing 衰减和闭环训练比例：优先看长时 drift；
5. `tc/lpips/crcd/warp` 损失比例：只用 validation 选择；
6. learning rate、warmup、EMA decay、diffusion steps；
7. 最后才扩大 hidden size/layers，避免把参数量增长误包装成创新。

默认固定 seed 快速迭代；多 seed 只可作为可选分析，不影响达标。当前数据集两项 validation
阈值同时达到后冻结候选并进入下一个数据集；三个数据集都锁定后才统一运行完整 test，test
结果只作最终报告和泛化核验，不能回流搜索。

### Phase 4：蒸馏与实时化

当前 CRCD 已有 teacher/student residual 接口。建议 Agent 给出具体蒸馏方案时必须回答：

- teacher 是 4-step 还是多步 residual trajectory；
- student 是否只预测 `P_A P_perp residual`；
- distillation target 是否 stop-gradient；
- 如何避免 teacher forcing 与 inference closed-loop gap；
- 如何测 operator gap、drift slope 和 long-video error；
- 1-step/2-step 在相同 active token budget 下是否仍优于 full residual。

建议顺序：先训练稳定的 4-step teacher，再冻结 teacher 蒸馏 2-step，最后尝试 1-step。不要
在 teacher 尚未对 validation TC/LPIPS 产生有效改善前就蒸馏，否则只会压缩错误轨迹。

### Phase 5：资源标定与交付

恢复 GPU 后，每个数据集单独做短跑标定：

```text
显存目标：总显存约 60%~85%
记录：reserved/allocated/peak VRAM、GPU util、功耗、p50/p95 latency
训练：epoch wall time、train/validation step time
推理：首帧、steady-state、每帧和 MP4 encode time
```

通过调整 micro batch、gradient accumulation、num workers、active ratio、gradient checkpointing
和 residual width 达到资源目标；不能通过伪造监控或只占显存张量来声称利用率达标。默认配置
必须来自真实标定，不要把旧机器上的数值直接复制到新机器。

### Phase 6：最终测试和展示

对每个数据集分别：

1. 冻结最终 checkpoint SHA256；
2. 启动一次完整 Infer，遍历全部 512 test records；
3. 保存一行 `metrics.csv/xlsx` 和完整 per-video details；
4. 随机保存 5 个 showcase MP4；
5. 对三个数据集结果做离线汇总表，仅用于展示，不反向参与选权重；
6. 用相同 prompt、seed、分辨率、时长生成 baseline/Ours 对比视频。

---

## 8. 推荐给另一个 Agent 的方案交付格式

请后续 Agent 输出一份独立的 `SOTA_PLAN.md` 或等价报告，至少包含：

```text
1. 任务定义和当前接口边界
2. 与 Rerender-A-Video、vid2vid-zero、TokenFlow、StreamDiffusion、VideoLCM 等工作的
   明确差异
3. TARDIS 哪个模块是新建模，哪个只是工程实现
4. 训练/蒸馏的完整目标函数和每项权重
5. baseline 与 A0-A10 消融矩阵
6. 三个数据集分别的超参数搜索空间和停止规则
7. 防止 TC 静态化和 test leakage 的检查
8. 训练、验证、蒸馏、Infer、Apply 的具体步骤和命令
9. 预期指标必须标注为假设/目标，不得写成已取得结果
10. 显存、利用率、延迟和质量的联合验收表
```

方案蒸馏回本仓库时，先改设计文档和测试，再改主网络；不得先改 Shell 入口来掩盖模型问题。

---

## 9. 正式命令与核心参数

### 9.1 Train

```bash
TARDIS_DATASET=dataverse bash scripts/train.sh
TARDIS_DATASET=openvid bash scripts/train.sh
TARDIS_DATASET=seedance bash scripts/train.sh
```

常用覆盖项：

```text
TARDIS_EPOCHS                    默认 20
TARDIS_STEPS_PER_EPOCH           默认 64
TARDIS_MICRO_BATCH_SIZE          默认 2
TARDIS_GRADIENT_ACCUMULATION_STEPS 默认 2
TARDIS_LEARNING_RATE             默认 1e-4
TARDIS_WEIGHT_DECAY              默认 1e-2
TARDIS_WARMUP_STEPS              默认 64
TARDIS_VALIDATION_INTERVAL       默认 1
TARDIS_VALIDATION_BATCH_SIZE     默认 8
TARDIS_NUM_WORKERS / PREFETCH_FACTOR
TARDIS_HEIGHT / WIDTH / NUM_FRAMES / FPS
TARDIS_HIDDEN_SIZE / NUM_LAYERS / NUM_HEADS / PATCH_SIZE
TARDIS_ACTIVE_RATIO / DIFFUSION_STEPS
TARDIS_TRANSPORT_QUOTIENT
TARDIS_QUOTIENT_REGULARIZATION / QUOTIENT_RANK_THRESHOLD
TARDIS_INNOVATION_PROPER_TIME / PROPER_TIME_MAXIMUM_HAZARD
TARDIS_GRADIENT_CHECKPOINTING / TARDIS_COMPILE_MODEL
TARDIS_CHECKPOINT_ROOT / TARDIS_OUTPUT_ROOT / TARDIS_RESUME
```

### 9.2 Infer

```bash
TARDIS_DATASET=dataverse bash scripts/infer.sh
TARDIS_DATASET=openvid \
TARDIS_CHECKPOINT=/home/TARDIS/checkpoints/openvid/<timestamp>/best.pt \
bash scripts/infer.sh
```

关键参数：`TARDIS_CHECKPOINT`、`TARDIS_TEST_SIZE`、`TARDIS_VALIDATION_SIZE`、
`TARDIS_SHOWCASE_COUNT`、`TARDIS_SEED`、`TARDIS_SPLIT_SEED`、`TARDIS_USE_EMA`、
`TARDIS_PRECISION`、模型结构参数以及 `TARDIS_RESUME_OUTPUT`。

正式 Infer 不要设置测试集截断；默认 `TARDIS_TEST_SIZE=512` 意味着完整当前测试划分。

### 9.3 Apply

```bash
TARDIS_DATASET=seedance \
TARDIS_CHECKPOINT=/home/TARDIS/checkpoints/seedance/<timestamp>/best.pt \
TARDIS_PROMPT="A robot running in the forest" \
TARDIS_STYLE="cinematic, highly detailed" \
TARDIS_DURATION=2 \
bash scripts/apply.sh
```

关键参数：`TARDIS_CHECKPOINT`、`TARDIS_PROMPT`、`TARDIS_STYLE`、`TARDIS_DURATION`、
`TARDIS_HEIGHT`、`TARDIS_WIDTH`、`TARDIS_FPS`、`TARDIS_SEED` 和与 checkpoint 一致的模型
结构参数。

所有脚本内部仍以 `torchrun --standalone --nproc_per_node` 启动，不要绕过脚本直接用普通
`python` 作为正式交付命令。

---

## 10. 输出目录与文件语义

```text
checkpoints/<dataset>/<timestamp>/
├── latest.pt       每 epoch 的可恢复状态
└── best.pt         当前 dataset validation TC/LPIPS 最优状态

outputs/train/<dataset>/<timestamp>/
├── manifest.json
├── events.jsonl
└── ...

outputs/infer/<dataset>/<timestamp>/
├── metrics.csv
├── metrics.xlsx
├── per_video_details.csv
├── per_video_details.jsonl
├── completed.jsonl / failures.jsonl
├── latency.json / resources.json
├── manifest.json / result_manifest.json
└── showcases/*.mp4

outputs/apply/<dataset>/<timestamp>/
├── video.mp4
└── video.json
```

任何输出都必须包含 dataset、seed、split/config 关键字段和 checkpoint SHA256，保证跨 Agent
实验可以追溯。

---

## 11. 当前状态、已验证内容与已知限制

### 已完成

- 三个本地数据 manifest 已准备并验收为每源 8,000 条、约 45 GB；Train/Validation/Infer
  只读本地数据盘。
- `selected_source=dataverse/openvid/seedance` 的单源 catalog 和 loader 已验证。
- 三套 checkpoint/output namespace 已实现。
- DataVerse initial-incumbent 已在 RTX 4080 SUPER 上完成 20 轮正式训练：run
  `20260810_033808_634217`，`latest.pt` 为 epoch 20 completed，`best.pt` 为 epoch 3。
- 当前 DataVerse validation best：TC `0.164016`、LPIPS `0.878462`；其余诊断指标为 FVD
  `76.335089`、FID `288.671317`、CLIPScore `0.223633`、SSIM `0.104974`。
- 该 run 的平均 GPU 利用率为 `67.08%`，峰值 allocated/reserved VRAM 分别约
  `20.49/26.67 GB`；validation batch 8 连续运行稳定。
- 恢复训练后 manifest 的陈旧 `error/stop_reason` 清理已添加回归测试；Python 和三个 shell
  入口的正式默认 FPS 已统一为 30。
- 固定 16 条 DataVerse validation 的先验差距诊断已完成：重复 SD-Turbo 首帧为
  `TC=0.103125 / LPIPS=0.764681`，旧 `best.pt` epoch 3 为
  `TC=0.125300 / LPIPS=0.829118`，旧 epoch 20 为
  `TC=0.132767 / LPIPS=0.850872`。旧时序网络同时劣于未加时序的冻结先验，不能作为下一轮
  热启动父权重。
- 主网络已改为 identity-preserving 初始化：motion flow 和 lite residual 均从零开始，
  初始 visibility 保护历史状态；生产 SD-Turbo 上实测 16 帧 latent/pixel delta 均严格为
  `0`。训练已加入 prior-anchored 首状态 scheduled sampling。
- 20 轮课程预算已从平均分配调整为 `5/5/10/20/20/40`，TC/LPIPS 损失默认权重提高为
  `5.0/3.0`，可微 LPIPS 默认按 4 帧 chunk 计算以降低指标对齐阶段开销。

### 当前不能声称的内容

- 当前 DataVerse best 距离冻结达标阈值（TC `0.060`、LPIPS `0.30`）仍很远，不能
  声称达到 protocol-SOTA 或公开 SOTA。
- 旧 DataVerse run 只保留为 initial diagnostic incumbent；它已被简单的 prompt-prior
  static control 同时支配，下一轮必须用新代码从头训练，不能从旧 epoch 3/20 精确续训。
- OpenVid 曾被错误提前启动并完成一个 20 轮中间 run；由于 DataVerse 尚未达到
  `protocol_sota=true`，该 run 不具备调优、Infer 或验收资格。Seedance 尚未启动。
- 尚未运行冻结候选的一次性正式 test Infer，因此没有本版本三个 test split 的真实 TC/LPIPS。
- 第 18-20 轮 `metric_alignment` 训练段约 11 分钟，完整 validation 约 6.8 分钟；当前配置
  不满足“所有 epoch 均为 6-8 分钟”的严格表述，后续需在不损失质量的前提下降低该阶段开销。
- 30 FPS 只是生成帧率协议，端到端每帧小于 33.3 ms 尚未实测通过。
- 过去日志中的三源联合训练数值属于旧架构/旧流程，不能直接比较或写入论文结果表。

### 工作树原则

当前工作树含有项目重构的有意修改。后续 Agent 不得使用 destructive git 命令回退未知修改，
不得删除本地数据、现有测试或用户新增文件；只在确认属于本方案的文件中增量修改。

---

## 12. 下一 Agent 的第一轮行动清单

在提出最终 SOTA 方案前，按以下顺序执行并把结果写入交接报告：

```text
[ ] 读完 appendix/开发prompt.txt 和 appendix/创新点.md
[ ] 检查三源 manifest 的 train/val/test id/hash 无交集
[x] 用 identity/static/repeated-frame 建立首轮指标 sanity table
[ ] 在 validation 上验证 TAR energy ratio 和 quotient energy ratio
[ ] 明确 B0-B8/A0-A10 的公平计算预算
[ ] 为 DataVerse/OpenVid/Seedance 分别确定 validation baseline
[ ] 给出 teacher -> 2-step -> 1-step 的 CRCD 蒸馏方案
[ ] 给出 active ratio、hazard、quotient、loss weight 的搜索空间
[ ] 给出显存/利用率/延迟测量脚本和停止规则
[ ] 给出最终 test 只运行一次的验收流程
```

后续方案通过审查后，回到本仓库的正确实施顺序是：

```text
SOTA_PLAN.md
    -> 机制/数据/指标单测
    -> 主网络增量实现
    -> 小规模 validation 试验
    -> 三源独立正式训练
    -> validation 选 best.pt
    -> 三源独立 full Infer
    -> Apply 展示与资源审计
```

最终判断标准不是“模块数量更多”，而是同等计算预算下同时降低 TC 和 LPIPS，并且通过
消融、泄漏审计、延迟和资源测试证明收益来自 **Transport-Aligned Residual Diffusion in
Innovation Subspaces** 这一统一建模机制。

---

## 13. 2026-08-11 Continuation Status

### DataVerse completed run

- Run directory: `/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260810_110745_753253/`
- `latest.pt`: `run_status=completed`, `epoch=20`, `micro_step=1280`, `optimizer_step=640`.
- Final EMA validation on all 256 records: `TC=0.156141`, `LPIPS=0.760100`, `FVD=56.633893`,
  `FID=110.327479`, `CLIPScore=0.330854`, `SSIM=0.106037`, `weighted_score=2.576596`.
- Independent full validation with the same split and EMA checkpoint reproduced
  `TC=0.156160`, `LPIPS=0.760183`, `FVD=56.695234`, `FID=109.862911`,
  `CLIPScore=0.330674`, `SSIM=0.106021`.
- Raw temporal weights were also evaluated on all 256 validation records:
  `TC=0.158350`, `LPIPS=0.772833`, `FVD=65.219638`, `FID=173.218872`,
  `CLIPScore=0.285827`, `SSIM=0.172520`. Formal Infer/Apply should therefore keep
  `use_ema=true`.
- `best.pt` remains the validation-selected checkpoint from epoch 1 (`TC=0.145897`,
  `LPIPS=0.771790`, `weighted_score=2.484499`); `latest.pt` is the final resumable state.
- No formal test Infer has been run for this completed run yet. No SOTA claim is valid.

### Current next run

- The queue is locked to DataVerse. No GPU job is currently active.
- DataVerse mechanism diagnostics show that transport is the blocking failure:
  `tar_to_raw_ratio=1.112842` during transport warmup, `1.615738` in closed loop, and
  `4.002732` during metric alignment. Per the SOTA protocol, flow/visibility/transport must be
  repaired before residual, router, distillation, or another dataset is tuned.
- The prematurely started OpenVid run `20260810_192951_078100` completed 20 epochs and is kept
  only as an auditable intermediate artifact. It must not be inferred, tuned, promoted, or used
  for an SOTA claim until DataVerse is frozen with `protocol_sota=true`.
- Next action: implement same-dataset weights-only warm-start, expose Stage 1 transport controls,
  then run a 1-3 epoch DataVerse probe from
  `/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260810_110745_753253/best.pt`.

---

## 14. 2026-08-11 达标规则与当前活动运行

历史说明（已被 2026-08-13 协议更新取代）：本节后续旧实验仍按当时的 validation-only
`target_pass` 语言记录，不应解读为最终 SOTA。当前规则以本文末尾“测试集 SOTA 验收协议”为准。

| 数据集 | TC 上限 | LPIPS 上限 |
|---|---:|---:|
| DataVerse | 0.060 | 0.30 |
| OpenVid | 0.070 | 0.32 |
| Seedance | 0.100 | 0.30 |

多 seed、bootstrap、Pareto 收敛、防静止检查、FVD、FID、CLIPScore、SSIM、速度和资源均不再
构成 `protocol_sota` 门槛；后四项指标和资源仍照常记录。任一目标未达到时继续当前数据集，
两项同时达到后立即冻结权重并进入该数据集正式 test Infer，然后才进入队列中的下一个数据集。

权重选择器也已同步为 target-first：达标候选无条件优先；达标状态相同时仅按
`0.625 * (TC/TC目标) + 0.375 * (LPIPS/LPIPS目标)` 排序。其余四指标、Pareto 关系和资源
不能阻止达标候选更新 `best.pt`。新训练日志和 validation 事件会显式写出 `target_pass`。

当前受保护 DataVerse best 为
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260811_160253_094999/best.pt`，
SHA-256 为 `c6013db865a1b5acb6ec583bf9e9250215cf5c31163ec09262cc909ad9bb6478`，
完整 validation 为 `TC=0.1458196015`、`LPIPS=0.7518049756`、加权分 `2.4587104016`，
`target_pass=false`。

最近完成的候选是 `dataverse-stage3-direct-metric-alignment-p1-seed3407`，run id
`20260811_171404_209095`。它仅启用 TC/LPIPS 可微损失，完整 validation 最佳值为
`TC=0.1516329759`、`LPIPS=0.7987257761`、加权分 `2.5779173854`。两项均劣于受保护权重，
因此已拒绝；FVD、FID、CLIPScore、SSIM 和资源数据未参与该决定。当前无活动 GPU 进程，
DataVerse 尚未达标，OpenVid 与 Seedance 仍禁止进入正式调优或 Infer。

## 15. 2026-08-12 DataVerse 最新审计与结构诊断

当前受保护 DataVerse checkpoint 已晋升为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260812_031057_954890/best.pt
SHA-256: bf86e80c0caa5ebf9bfe80fdf2a51348409b51b17bc194fad7f8bf40654c1aca
```

固定 256 条完整 validation 指标为：

```text
TC       0.1458170953
LPIPS    0.7504540676
score    2.4569956608
```

已按该 run 的原始 512x512 factory options 在 CUDA/BF16 上重建模型并严格加载：temporal
state `213/213`、EMA shadow `197/197`，不存在部分加载、静默丢键或 prior 混入。该权重只
是当前可靠回滚点，`target_pass=false`，不得称为已达到 SOTA。

后续 motion annealing run `20260812_052323_715121` 在完整 validation 上的三个 epoch 为：

```text
epoch 1  TC 0.145899  LPIPS 0.750043
epoch 2  TC 0.146174  LPIPS 0.749617
epoch 3  TC 0.147218  LPIPS 0.748877
```

其 LPIPS 小幅下降但 TC 单调退化，最优加权分仍差于受保护 parent，因此已拒绝。机制诊断
表明生成运动能量仅约参考视频的 1.2%，但继续训练 motion head 无法恢复 prompt 到具体运动
轨迹的一对多信息。当前首要结构瓶颈是冻结 SD-Turbo 首帧只接一个 bounded
`LiteResidualCorrector`，不足以把 prompt prior 映射到目标视频外观；启用
`keyframe_lite_alignment` 时 transition lite branch 又被完全关闭，后续帧只能依赖稀疏 DiT。

下一步只在 DataVerse 上实施并验证高容量 prompt-conditioned keyframe residual/generation
path，同时恢复独立的 transition residual 更新。任何候选仍必须在完整 validation 上按 TC
0.625、LPIPS 0.375 的固定归一化分数优于上述 incumbent，或同时达到 `TC<=0.060` 与
`LPIPS<=0.300`，才允许晋升；test、OpenVid 和 Seedance 继续冻结。

## 16. 2026-08-12 新结构预检与活动候选

用户已取消人为的 epoch 时长和显存目标，资源配置由 Agent 根据稳定性决定；模型选择规则仍
严格锁定为完整 validation 上的 TC/LPIPS，不得缩短 validation、挑样本或使用 test 调参。

新结构预检已完成：完整回归 `437 passed, 2 warnings`；受保护 checkpoint 的 SHA-256、EMA
前向迁移、CUDA/BF16 和确定性生成均通过。新 `keyframe_residual_dit` 与
`transition_lite_corrector` 的输出头在迁移后严格为零；真实 DataVerse 512x512x16 单批反向
确认两分支梯度非零。micro-batch 2 的完整 AdamW 更新峰值 allocated/reserved 显存约
`19.4/21.8 GB`，validation batch 8 的生成峰值约 `17.8/27.1 GB`，两者均稳定。

已清理候选账本中明确淘汰的历史 checkpoint，日志和账本保留。当前只保留受保护父权重：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260812_031057_954890/best.pt
SHA-256: bf86e80c0caa5ebf9bfe80fdf2a51348409b51b17bc194fad7f8bf40654c1aca
```

已登记候选 `dataverse-stage5-dual-residual-capacity-p1-seed3407`。它只验证高容量 prompt
keyframe residual 与独立 transition residual 这一统一机制：3 epoch、每 epoch 256 个真实
microbatch、micro-batch 2、accumulation 2，并在每个 epoch 后完整验证全部 256 条。候选只有
分数低于 incumbent `2.456995660764303` 或直接同时达到两项阈值才可晋升。

该候选已启动：run id `20260812_111056_199420`，tmux
`tardis_dataverse_s5_dual_residual_p1`。manifest 已核验 warm-start EMA 哈希、固定划分和全部
超参；稳态约 `3.1 s/microbatch`、进程显存约 `24.4 GB`、GPU 利用率采样为 `100%`。

第 1 轮完整 256 条 validation 的权威事件为 `TC=0.1458122904`、`LPIPS=0.7388752093`、
固定尺度分数 `2.4424720366`。相对受保护父权重分别改善 `0.0000048049`、`0.0115788583`
和 `0.0145236241`；候选按预注册规则存活并继续第 2 轮，但尚未达到 DataVerse 双阈值。

第 2 轮为 `TC=0.1457995491`、`LPIPS=0.7596230654`、分数 `2.4682741345`。虽然 TC 比第 1
轮低 `0.0000127413`，LPIPS 却回退 `0.0207478561`，selector 正确保留第 1 轮 best。当前
`5e-5` 学习率已越过 LPIPS 泛化最优点；完成预注册第 3 轮后只能从第 1 轮 best 低率续训，
不能从 latest 继续。

第 3 轮已完成：`TC=0.1457960245`、`LPIPS=0.7768812451`、分数 `2.4898101451`。它延续了
LPIPS 退化，因此第 1 轮仍是唯一最优轮次。Stage 5 已完成并正式晋升为当前 DataVerse
受保护 checkpoint：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260812_111056_199420/best.pt
SHA-256: 5ad57b233ce0e187b409c5c393a62b3f61aa435f8c68d3c848fe0ce05a61b052
best epoch: 1
TC:          0.1458122904057743
LPIPS:       0.7388752092665527
score:       2.4424720366433395
```

相对旧受保护权重，TC、LPIPS、score 分别变化 `-0.0000048049`、`-0.0115788583`、
`-0.0145236241`。严格审计已通过：EMA temporal state `369/369`，CUDA/BF16 prompt-only
生成可执行且同 seed 输出逐元素确定，输出尺寸为 `1x2x3x512x512`；模型共有 337 个可训练
tensor、116,485,937 个可训练参数。该权重仍为 `target_pass=false`，只能称为固定协议下的
当前最佳，不得称为公开 SOTA。

最新根因诊断是 diffusion-time 训练/部署错配：keyframe 和 transition 训练时均匀采样
`t~U(0,1)`，而 prompt-only 部署从纯噪声端点 `t=1` 起步。训练重构损失继续下降而完整
validation LPIPS 在第 2、3 轮快速恶化，与此曝光错配一致。下一候选必须只改变
diffusion-time 采样策略，优先隔离 keyframe 并使用 `t=1` 端点训练；从上述 `best.pt` 的
EMA 做 weights-only 热启动，不得从 `latest.pt` 恢复优化器或第 3 轮权重。DataVerse 达标
前仍禁止正式 test Infer、OpenVid 调优和 Seedance 调优。

## 17. 2026-08-12 Stage 6 端点对齐结果

Stage 6 已完成全部 256 条固定 DataVerse validation。权威 `validation` 事件为：

```text
TC        0.14580285389072578
LPIPS     0.735257896289113
score     2.4378520983897847
FVD      61.19464590167162
FID      143.95376398987105
CLIP      0.3363623410245983
SSIM      0.19170274468270243
```

相对 Stage 5 保护权重，TC、LPIPS 和固定尺度分数分别变化
`-0.0000094365`、`-0.0036173130`、`-0.0046199383`，因此按锁定协议正式晋升。新保护权重：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260812_132330_756378/best.pt
SHA-256: 3fb8610c42380a009794c6863c2b9d4727d62456399341f2fb8f18eefabbcc06
```

严格审计通过：temporal state `369` 个张量，EMA 完整覆盖全部 `337` 个可训练张量，
可训练参数 `116,485,937`；CUDA/BF16 prompt-only 生成成功，同 seed 逐元素确定，输出尺寸为
`1x2x3x512x512`。该结果仍远高于 `TC<=0.060`、`LPIPS<=0.300`，只能称为当前固定协议最佳。

端点对齐的收益为真实但很小，说明它不是剩余数量级差距的主因。下一阶段不得继续重复
keyframe-only 低率微调；优先增强可部署的 prompt-conditioned motion prior，并用完整训练集
覆盖和固定 validation 检验其是否能恢复参考视频的运动分布。正式 test、OpenVid 和 Seedance
仍保持冻结。

## 18. 2026-08-12 Stage 7 完整因果路径候选

用户已撤销 epoch 时长、显存占用和 GPU 利用率的硬门槛；Agent 可按稳定性和质量自行配置资源，
但完整固定 validation、双指标阈值、数据集顺序和 test 隔离规则不变。

已登记 `dataverse-stage7-full-causal-reactivation-p1-seed3407`。该候选从 Stage 6 的 EMA
weights-only 热启动，保留 `diffusion_time_sampling=endpoint`，并恢复 TARDIS 的完整累计
因果 curriculum：motion、visibility、transport、router、proper-time、dual-frequency residual、
CRCD、TC 与 LPIPS。训练和 prompt-only 部署统一采用 `training_noise_scale=0.1`，使
`PromptMotionScaffold` 能从 prompt、时间、有限状态和随机运动变量建模一对多运动分布，而不是
在零噪声下退化为条件均值。模型尺寸、冻结 SD-Turbo、两步采样、DataVerse 7232/256/512
固定划分、seed 3407 和 512x512x16 协议均不变。

预注册预算为 8 epoch，每 epoch 512 个真实 microbatch，micro-batch 2、accumulation 2、
`lr=1e-5`、`EMA=0.995`，每轮完整验证 256 条。只有完整验证分数低于保护值
`2.4378520983897847` 或同时满足 `TC<=0.060`、`LPIPS<=0.300` 才可晋升；连续完整验证显示
该机制无实质收益时应提前终止并转向更强的可部署视频运动先验，而不是机械续训。正式 test、
OpenVid 和 Seedance 继续冻结。

## 19. 2026-08-12 Stage 7 中期结果与采样轨迹对齐修正

Stage 7 前三轮均已对固定 256 条 DataVerse validation 做完整评测：

```text
epoch 1  residual_teacher  TC 0.1460150855  LPIPS 0.7339422011  score 2.4384182254
epoch 2  closed_loop       TC 0.1464029973  LPIPS 0.7296010072  score 2.4370324807
epoch 3  closed_loop       TC 0.1458843294  LPIPS 0.7294590243  score 2.4314522112
```

第 3 轮相对 Stage 6 保护点的 TC、LPIPS、score 变化分别为
`+0.0000814755`、`-0.0057988720`、`-0.0063998872`。它是当前 Stage 7 内部 best，文件为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260812_144534_902723/best.pt
epoch-3 SHA-256: 8532e1d7c6d68ece35797c85af3a650567858cfa0db8993a8261b1884ad7ca1e
```

Stage 7 尚未结束，因此全局保护点仍保持 Stage 6，不得提前晋升。完整 curriculum 在第 4 轮
结束时处于 CRCD，第 5 轮才首次进入 metric alignment；至少观察这两次完整验证后再判断该
机制是否耗尽。DataVerse 仍未达到 `TC<=0.060` 且 `LPIPS<=0.300`。

同时定位并修复了一个明确的训练/部署轨迹错配：部署 `diffusion_steps=2` 会执行 `t=1 -> 0`
两次 residual denoiser，而有 target 的训练此前只执行一次 `context.predict()`。新增
`sampler_trajectory_alignment`，默认 `false` 以兼容旧权重；开启时要求
`diffusion_time_sampling=endpoint`，并让 keyframe 与 transition 在训练中执行与部署相同的
完整采样轨迹。由于 DiT 输出头零初始化会使后一步对前一步的初始局部雅可比为零，训练路径还
加入了只改变反向梯度、不改变前向数值的逐步梯度桥。验证结果：

```text
定向测试                         58 passed
全量回归                         444 passed, 2 warnings
对齐开关前后训练前向最大绝对差   0.0
两步 denoiser 时间               [1.0, 0.0]
两步输出梯度                     均非零
脚本 bash -n / compileall         passed
```

三个正式 shell 入口均支持环境变量 `TARDIS_SAMPLER_TRAJECTORY_ALIGNMENT=1`。Stage 7 是在该
修正前启动，仍使用默认关闭的旧训练语义；若 Stage 7 的 CRCD/metric-alignment 结果仍只有小幅
收益，下一候选应从其严格审计后的最优 EMA 做 weights-only 热启动，开启该开关并保持 endpoint
两步采样。test、OpenVid、Seedance 在 DataVerse 双目标达标前继续冻结。

## 20. 2026-08-13 Stage 7 终止审计与下一阶段入口

Stage 7 运行目录 `/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260812_144534_902723/`
已确认没有活动进程。实际日志显示完整 validation 已完成 5 轮，第 6 轮只执行到 9 个
micro-batch 后收到 SIGINT；因此不能把该 run 标记为 completed，也不能恢复其 `latest.pt`。

完整 validation 结果如下：

```text
epoch 1  TC 0.1460150855  LPIPS 0.7339422011  score 2.4384182254
epoch 2  TC 0.1464030000  LPIPS 0.7296010072  score 2.4370324807
epoch 3  TC 0.1458843294  LPIPS 0.7294590243  score 2.4314522112  candidate best
epoch 4  TC 0.1458888186  LPIPS 0.7329716325  score 2.4358897346
epoch 5  TC 0.1460592221  LPIPS 0.7408641046  score 2.4475303607
```

Epoch 3 `best.pt` SHA-256 为
`8532e1d7c6d68ece35797c85af3a650567858cfa0db8993a8261b1884ad7ca1e`，严格 EMA、CUDA/BF16
生成和同 seed 确定性均已审计。Stage 7 终止后，epoch 3 因完整 validation 固定加权分
`2.4314522112` 优于 Stage 6 的 `2.4378520984`，已按既有 selector 晋升为全局 protected
incumbent。双目标是否达标不影响未达标候选之间按固定加权分选权；它只影响最终 test
`protocol_sota` 验收。Stage 7 的 `latest.pt` 禁止恢复。

资源仅作诊断：Stage 7 平均 GPU 利用率约 52.77%，峰值 allocated/reserved 约
21.62/27.54 GiB；本轮不再把资源门槛当作 SOTA 接受条件。

根因诊断保持不变：生成运动能量约为参考运动的 0.8%-1.2%，且 SD-Turbo 首帧先验与目标
首帧 LPIPS 约 0.71，说明机械增加 epoch 不能解决数量级差距。下一阶段先验证部署轨迹对齐的
prompt-conditioned keyframe residual，再依据完整 256 条 validation 决定是否冻结候选。冻结
候选可以运行一次完整 test 验收，但 test 结果不得回流训练或候选选择；OpenVid 和 Seedance
在 DataVerse `protocol_sota=true` 前继续冻结。

## 21. 2026-08-13 测试集 SOTA 验收协议与 DataVerse 基线

用户只修改了最终 SOTA 判定面，其余训练、验证、热启动和 checkpoint 选择流程保持不变：

```text
train split       只用于参数优化
validation split 只用于 target-first 加权分选择 best.pt
test split        只用于冻结权重的最终 SOTA 验收
```

训练日志中的 `target_pass` 仍是 validation selector 标志，不再等价于 `protocol_sota`。只有冻结
权重完整遍历 512 条 test 后，TC 与 LPIPS 同时达到本数据集锁定阈值，才能写
`protocol_sota=true`。test 不进入 loss、学习率、early stopping、权重选择或超参数搜索。

当前 DataVerse 冻结权重及完整 test 结果：

```text
checkpoint  /root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
            20260812_144534_902723/best.pt
SHA-256     8532e1d7c6d68ece35797c85af3a650567858cfa0db8993a8261b1884ad7ca1e
test rows   512/512
TC          0.0351056114  (<= 0.060, pass)
LPIPS       0.7145033086  (> 0.300, fail)
```

因此 DataVerse `protocol_sota=false`。报告位于
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/experiments/dataverse_protocol_v2_prompt_only_test_full_0619.json`；
逐条 JSONL 覆盖 512 个唯一 test record，EMA、split、checkpoint SHA 和宏平均均已核验。

下一候选固定为 `dataverse-stage8-deployment-trajectory-keyframe-p1-seed3407`：从上述 EMA 做
weights-only 热启动，使用 `train_mode=keyframe_only`，保持 endpoint 两步扩散并开启
`sampler_trajectory_alignment`，直接训练部署时的 prompt-conditioned keyframe trajectory。
每轮完整 validation 仍是唯一选择信号；旧权重不得覆盖或删除。候选冻结后才允许完整 test
验收，OpenVid 与 Seedance 在 DataVerse test 双目标通过前继续冻结。

2026-08-13 Stage 8 首次协议匹配运行 `20260813_064926_021032` 在 epoch 1 的
`micro_step=960/1808` 后随外部容器重启终止。事件日志无 traceback、OOM、CUDA 错误或
协作式信号，且尚未完成 validation；当时训练仅在 epoch 边界保存，因此 checkpoint 目录为空，
该运行不能续训或参与候选选择。训练循环现已增加 `TARDIS_CHECKPOINT_INTERVAL_STEPS=256`
的轮内原子 `latest.pt` 保存，只在梯度累计边界执行，不改变 validation、selector、`best.pt`
或 test 协议。重新启动时仍从上述 Stage 7 EMA 做 weights-only warm start。

已重新启动协议一致的 Stage 8 run：`20260813_082614_642814`，tmux 会话为
`tardis_dataverse_s8_keyframe_p1r2`。启动时已核对 `LPIPS=1.5`、`epochs=2`、
`steps_per_epoch=1808`、`micro_batch=4`、累积步数 `2`、`checkpoint_interval_steps=256`、
`validation=256`、`test=512`、endpoint 两步轨迹对齐，以及父权重 SHA
`8532e1d7c6d68ece35797c85af3a650567858cfa0db8993a8261b1884ad7ca1e`。当前训练已进入 GPU
计算阶段；首个完整 validation 结束前不做 test 评测。

Stage 8 run `20260813_082614_642814` 已完成两轮。完整 validation 从 Stage 7 保护点
`TC=0.145884 / LPIPS=0.729459 / score=2.431452` 改善至 epoch 2 的
`TC=0.038720 / LPIPS=0.711022 / score=1.292116`；epoch 2 优于 epoch 1，已由 selector 保存为
`best.pt`。冻结权重 SHA 为
`6b3aa8a22b0c9e193caa9f36ccb0c928938024da095d424daf7e222c905ac274`，checkpoint 记录
`epoch=2`、`selector.best_epoch=2`、369 个 temporal state、337 个 EMA shadow，且不包含 frozen
prior。下一步只允许对该冻结权重执行一次 512 条 DataVerse test 验收。

## 22. 2026-08-13 Stage 8 完整 Test 与 Stage 9

Stage 8 冻结权重已在固定 DataVerse test split 上完成正式全量验收。首次运行因
`0001496725.mp4` 是 TAR 中一个约 2.32 GB、3.5 小时的异常长视频而得到 `511 completed + 1
failed`；已补充本地 TAR member 有界分块暂存、基于时间戳的 seek 抽帧，以及 infer 恢复时只重试
失败条目且在 DataLoader 前排除已完成 ID 的逻辑。随后使用同一权重 SHA、同一输出目录恢复，最终
账本为 `512 completed / 0 failed`，临时媒体位于数据盘并在退出时删除，未将 2.32 GB 整体载入
内存。全套测试为 `463 passed, 2 warnings`。

正式 test 结果：

```text
TC         0.0351660366  pass (<= 0.060)
LPIPS      0.6996565821  fail (> 0.300)
FVD       18.7416584274
FID      112.5723693234
CLIP       0.3289329475
SSIM       0.2302446027
```

输出目录为
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/outputs/infer/dataverse/20260813_100654_426881`。
相对 Stage 7 正式 test，LPIPS 改善 `0.0148467265`（约 `2.08%`），但仍未达到锁定阈值，因此
DataVerse `protocol_sota=false`。该 test 结果只用于验收，不进入下一候选的超参选择。

Stage 9 已预注册为
`dataverse-stage9-perceptual-trajectory-reweight-p1-seed3407`。其依据仅是 Stage 8 完整 validation
中 LPIPS 从 epoch 1 的 `0.718757` 连续下降到 epoch 2 的 `0.711022`，同时 TC 已低于 validation
目标。Stage 9 从 Stage 8 EMA `best.pt` 做 weights-only 热启动，开启新的 optimizer 与 cosine
周期；只将 keyframe-only 的 LPIPS 权重从 `1.5` 提高到 `3.0`，其余 7232 条完整训练覆盖、两轮
预算、两步 endpoint 部署轨迹、模型结构、seed、学习率、EMA 和 256 条完整 validation 均保持
不变。只有完整 validation 优于 Stage 8 保护分数 `1.2921159742` 或双目标同时达标才可晋升。

## 23. 2026-08-13 Stage 9 完成与 Stage 10

Stage 9 run `20260813_120300_568578` 已完成两轮全量训练和完整 validation。两轮均继续降低
LPIPS，epoch 2 由 selector 保存为 `best.pt`：

```text
epoch 1  TC 0.0387368102  LPIPS 0.7056910363  score 1.2856222349
epoch 2  TC 0.0387806941  LPIPS 0.6989732226  score 1.2776820920  best
```

冻结权重：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260813_120300_568578/best.pt
SHA-256 b6c2c55f2503b9365d31af63df60d93502930a3f22a54c6c795e1f4ab11a7cba
```

训练资源摘要为平均 GPU 利用率 `82.63%`、峰值 allocated/reserved
`20.41/26.69 GiB`。checkpoint 含 369 个 temporal state tensor、337 个 EMA tensor，未包含
冻结先验。

该冻结权重已独立完成 512 条 DataVerse test 验收，512 成功、0 失败，并输出 5 个
`512x512@30fps` showcase MP4：

```text
TC          0.0352200821  pass
LPIPS       0.6889019534  fail
FVD        18.8594680372
FID       120.3716242475
CLIP        0.3250937217
SSIM        0.2291146528
```

test 输出目录为
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/outputs/infer/dataverse/20260813_134404_842528`。
因此 DataVerse 仍为 `protocol_sota=false`。该 test 只作冻结候选验收，不进入下一候选的超参
选择。

Stage 10 已预注册为
`dataverse-stage10-perceptual-trajectory-reweight-p2-seed3407`。依据仅来自 Stage 9 完整
validation 的 LPIPS 连续下降，同时 TC 持续低于目标。Stage 10 从 Stage 9 EMA weights-only
热启动，只把 keyframe-only LPIPS 权重从 `3.0` 提高到 `6.0`；两轮预算、7232 条训练覆盖、
256 条 validation、模型、seed、学习率、EMA、两步 endpoint 采样和其余参数不变。只有完整
validation 优于保护分数 `1.2776820920` 或双指标达标才可晋升并进行一次完整 test 验收。

## 24. 2026-08-13 Stage 10 完成与 Stage 11

Stage 10 run `20260813_141140_860872` 已完成两轮 7232 条完整训练覆盖和每轮 256 条完整
validation。仅将 keyframe-only LPIPS 权重从 `3.0` 提高到 `6.0` 后，两轮 validation 均继续
降低固定加权分，epoch 2 由 selector 保存为 `best.pt`：

```text
epoch 1  TC 0.0388361067  LPIPS 0.6933019658  score 1.2711702349
epoch 2  TC 0.0389095736  LPIPS 0.6863500706  score 1.2632456467  best
```

冻结权重为
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_141140_860872/best.pt`，
SHA-256 为 `8f1d4af0f4f6a187ae13b8c80c3abbf29354a7da83aa4592255a5a730a13c1da`。
checkpoint 记录 `epoch=2`、`selector.best_epoch=2`、369 个 temporal state tensor、337 个
EMA shadow tensor，且不包含 frozen prior；所有模型和 EMA tensor 均为有限值。训练资源摘要为
平均 GPU 利用率 `82.83%`、峰值 allocated/reserved `20.39/26.69 GiB`。

该 validation-selected 权重随后在固定 DataVerse test split 上完成独立正式验收，结果为
`512 completed / 0 failed`、512 个唯一 record ID，并输出 5 个 H.264、`512x512@30fps`、
16 帧 showcase MP4：

```text
TC          0.0353393069  pass (<= 0.060)
LPIPS       0.6778379988  fail (> 0.300)
FVD        19.3570270329
FID       138.0606537942
CLIP        0.3169414482
SSIM        0.2223196298
```

test 输出目录为
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/outputs/infer/dataverse/20260813_155321_483611`；
`metrics.csv` SHA-256 为
`422f20b66ff47887327e2e84171be7d45fd29d9afc72ac0d7ac7bb9e2836d6dd`，
`per_video_details.jsonl` SHA-256 为
`87251131c56fbf21ac3596b52b40008f874a158fbafb0715cef569d9d10f7155`。
正式 infer 平均 `1.3300 s/video`、`83.13 ms/frame`，平均 GPU 利用率 `53.44%`，峰值
allocated/reserved `12.54/14.12 GiB`。DataVerse 仍为 `protocol_sota=false`；这些 test 指标
仅作冻结权重验收，不反馈给后续训练、调度、checkpoint 选择或超参搜索。

Stage 11 预注册为 `dataverse-stage11-perceptual-trajectory-reweight-p3-seed3407`。依据仅来自
Stage 10 完整 validation：LPIPS 从 epoch 1 的 `0.693302` 继续下降到 epoch 2 的 `0.686350`，
同时 TC 保持低于目标。代码检查确认 LPIPS 在 512x512 VAE 解码结果上计算，且梯度穿过最终两步
部署轨迹。Stage 11 只把 keyframe-only LPIPS 权重从 `6.0` 提高到 `12.0`；其余两轮预算、
7232 条训练覆盖、256 条 validation、两步 endpoint 采样、模型、seed、学习率、EMA 和数据划分
全部固定。只有完整 validation 优于 Stage 10 保护分数 `1.2632456467` 或双指标同时达标，才可
晋升并进行一次完整 test 验收。

## 25. 2026-08-13 Stage 11 完成与 Stage 12

Stage 11 run `20260813_165601_304501` 已完成两轮 `7232` 条训练覆盖和每轮完整 `256` 条
DataVerse validation。它在验证集上继续改善 LPIPS，并由 validation-only selector 晋升为当前
DataVerse 保护权重：

```text
epoch 1  TC 0.0389646544  LPIPS 0.6820588701  score 1.2584554040
epoch 2  TC 0.0390457076  LPIPS 0.6753896185  score 1.2509631444  best
```

冻结权重：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_165601_304501/best.pt
SHA-256 3dfa630ccd4b2ee3b73bfb21af2b417b43d99b6db1a8ad8a08b39174d0d86772
```

训练资源为平均 GPU 利用率 `82.61%`，峰值 allocated/reserved `21.51/27.24 GiB`。checkpoint
包含 `369` 个 temporal state tensor 和 `337` 个 EMA tensor，模型及 EMA tensor 均为有限值。

该冻结权重随后完成固定 DataVerse test 的独立验收：`512/512` 成功、`0` 失败、`512` 个唯一
record ID，并输出 `5` 个 H.264 `512x512@30fps`、16 帧 showcase MP4。六项 test 指标为：

```text
TC         0.0354743376  pass (<= 0.060)
LPIPS      0.6680766818  fail (> 0.300)
FVD       19.9683230799
FID      159.2701206529
CLIP       0.3101611099
SSIM       0.2147300447
```

test 输出位于
`/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/outputs/infer/dataverse/20260813_183138_311671`，
其中 `metrics.csv` SHA-256 为
`699957ed14ab002f00c58b62d5f626962dd9812b3f7beffd1a9e637dd5478961`，
`per_video_details.jsonl` SHA-256 为
`8534babe7da6a191d9845c78a8df83242e95ad341afac7c22dc6fe03c845e3f9`。正式 infer 平均
`1.3296 s/video`、`83.10 ms/frame`，峰值 allocated/reserved `16.12/17.20 GiB`，平均 GPU
利用率 `53.50%`。DataVerse 的 test `protocol_sota` 仍为 `false`，因为 LPIPS 尚未达到
`0.300`；test 指标不反馈给后续训练、调度、checkpoint 选择或超参搜索。

Stage 12 已预注册为 `dataverse-stage12-full-trajectory-perceptual-p1-seed3407`，并保持
DataVerse -> OpenVid -> Seedance 队列。Stage 8-11 的 `keyframe_only` 训练只让首帧承担主要
LPIPS 优化，而正式 validation/test 对完整 16 帧部署轨迹评分。Stage 12 唯一算法变化是切换
到 `full_temporal`，让完整因果 rollout 的每一帧和 transition 参与 LPIPS/TC 优化，并采用
`metric_alignment` curriculum；`micro_batch=2`、gradient accumulation `4` 仅是 32 GB 单卡的
显存适配，有效 batch 仍为 `8`。

固定配置：

```text
dataset=dataverse
warm_start=Stage 11 best.pt, warm_start_use_ema=true
epochs=2
steps_per_epoch=3616
train_records_per_epoch=7232
validation_size=256
test_size=512
split_seed=3407
train_mode=full_temporal
curriculum_profile=metric_alignment
micro_batch_size=2
gradient_accumulation_steps=4
diffusion_steps=2
diffusion_time_sampling=endpoint
sampler_trajectory_alignment=true
keyframe_lite_alignment=true
keyframe_residual_generation=true
learning_rate=1e-6
weight_decay=0
warmup_steps=64
ema_decay=0.999
lpips_loss_weight=12.0
tc_loss_weight=5.0
transport_history_fallback_weight=1.0
lite_max_magnitude=0.75
precision=bf16
```

Stage 12 只有在两轮完整 validation 后才能决定是否晋升；保护分数为 `1.2509631444`，选择
公式仍为 `0.625*(TC/0.060)+0.375*(LPIPS/0.300)`，不得提前运行 test。Stage 11 权重和所有
历史候选必须保留为可审计回滚点。

Stage 12 已正式启动：run id `20260813_192542_254984`，tmux 会话
`tardis_dataverse_s12_full_p1`。manifest 已核验为 DataVerse `7232/256/512` 固定划分、16 帧
完整训练、EMA warm start，且父权重 SHA 与预注册一致。进入 `metric_alignment` 后连续 23 个
micro-batch 无 non-finite 或 CUDA 错误，稳定均值约 `10.97 s/micro-batch`；监控样本显示进程
显存约 `25.37 GiB`、GPU 利用率 `100%`。按该速率估算单轮训练约 `11 h`，其后还需完整
validation；这属于完整 16 帧 VAE 解码、LPIPS/TC 和全时序反向的实际计算成本。进程不得因预计
耗时而改成抽样训练或截断帧数。

## 26. 2026-08-13 Stage 12 首个恢复点审计

Stage 12 在 `micro_step=256`、`optimizer_step=64` 时按注册协议原子写入首个 `latest.pt`：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260813_192542_254984/latest.pt
SHA-256 83270d560e2f3563198e90a6a798a97a0443bd39d19b6656916bce73a9091881
```

CPU-only 审计确认 checkpoint 的恢复位置为 zero-based `epoch=0`、
`next_batch_index=256`，与 `micro_step=256` 精确一致；gradient accumulation 已处于边界
`accumulation_index=0`。模型、EMA 和 optimizer 分别含 `369`、`337` 和 `1011` 个张量，所有
浮点值均为有限值，`nonfinite_ledger` 为空。checkpoint 同时包含 scheduler、scaler、objective、
selector、rank-0 RNG 状态和数据游标，可从该批次边界精确恢复。训练在保存后继续运行；该
`latest.pt` 只用于故障恢复，不参与 validation-only 候选选择，也不替代 Stage 11 保护权重。

同一运行随后在 `micro_step=512`、`optimizer_step=128` 原子覆盖 `latest.pt`。更新后的
SHA-256 为 `44d38747809b5c0bb33559b597d58e1b423455a66c2b4e9fa055e3468a3a3401`；恢复位置
为 zero-based `epoch=0`、`next_batch_index=512`、`accumulation_index=0`。模型、EMA 和
optimizer 张量再次全部通过有限性检查，`nonfinite_ledger` 仍为空。权威恢复位置以
`TARDIS_SOTA/work/pipeline_state.json` 中的最新记录为准。

之后 `latest.pt` 又按周期原子推进至 `micro_step=768`、`optimizer_step=192`，最近一次审计
SHA-256 为 `f88ff39e25ad155d9e7644c3851affe70db3c682ec06008fa38c6347b8f79cde`，
`accumulation_index=0` 且 `nonfinite_ledger` 仍为空。注意 `latest.pt` 是运行中会持续覆盖的
恢复文件；记录的 SHA 只对应标注的 micro-step，不应被当作运行结束后的永久摘要。

## 27. 2026-08-14 Stage 12 运行中审计

Stage 12 仍在 tmux 会话 `tardis_dataverse_s12_full_p1` 中运行。2026-08-14 04:24 UTC
人工核验已推进到第 1 轮 `micro_step>=2947/3616`（约 `81.5%`），尚未产生本候选的完整
validation 事件，因而
当前正式可比较的 LPIPS 仍是 Stage 11 的 validation `0.6753896185` 和 test
`0.6680766818`。Stage 12 训练 micro-batch 的 LPIPS 只用于数值健康诊断，不得作为候选改善、
退化或达标证据。进程持续写入事件账本和原子 `latest.pt`，未观察到 non-finite micro-batch、
CUDA 错误或数据失败；该运行不得在首轮完整 validation 前提前执行 test。

代码级部署一致性审计确认：`metric_alignment` 阶段的 teacher-forcing ratio 为 `0`；首帧与
后续 15 帧均走两步 endpoint sampler trajectory，完整 16 帧参与可微 LPIPS/TC，正式
validation 则按每条记录固定的 sample seed 调用同一 `model.generate()` 路径。当前发现一个
尚未启用、尚未由 validation 证实的候选差异：训练 objective 对 VAE 解码结果直接计算
LPIPS/TC，而公共部署路径会先执行 `clamp(-1, 1)`。只有 Stage 12 两轮完整 validation 结束且
未达到目标后，才允许把“训练度量范围与部署范围一致化”预注册为下一阶段的单一机制变量；
在此之前不得修改运行代码、不得把该诊断写成已验证收益，也不得使用 Stage 12 或历史 test
结果为它选参。

## 28. 2026-08-14 Seedance split audit finding

在不访问视频字节、不占用训练 GPU 的 CPU 审计中，按当前固定 `StablePartition(seed=3407,
validation_size=256, test_size=512)` 重建了三个本地 manifest 的划分。DataVerse 和 OpenVid
的 train/validation/test caption hash 交集均为零；Seedance 的 record ID 和 media locator
在三个 split 间均无交集，但发现 `8000` 条记录只有 `6770` 个唯一 caption，存在 `1000` 个
重复 caption 组、共 `2230` 条记录。当前按 record ID 的 Seedance 划分因此产生：

```text
caption(train, validation) = 66
caption(train, test)       = 115
caption(validation, test)  = 7
```

这属于潜在 prompt 泄漏，不能用于最终 Seedance SOTA 声明。DataVerse Stage 12 不受该发现
影响；在 Seedance 队列启动前必须预注册一个 caption-group 隔离方案（或先做有审计记录的
 caption 去重），重新生成并哈希锁定 Seedance manifest/splits，然后从 Seedance 自己的
 训练集重新开始。不得把当前按 ID 的 Seedance split 与新 split 混用，也不得用该发现回溯
 修改 DataVerse 的 test 或 validation。

CPU 探针已经证明 caption-group 隔离可以保持精确 `7232/256/512`：按 NFC+strip caption
SHA-256 和固定 `3407` salt 排序后选择 `218` 个 validation groups、`437` 个 test groups，
得到零 caption/ID 交集。候选 split ID hashes 为 train
`c4d8940787b1fc1f170f6d854cf3c8ac3bab1b58203f25925171d780544b2aa9`、validation
`8562a7287225740677cc1a41ea9111f8912d0f02001e4408cd69ac20958c1a71`、test
`35bd09a1103dceeaa03782274d4ebc3e7d4f23dbdd7abbb559f51ab929027f2b`。这只是待实现、待测试、
待原子更新 lock 的候选，Seedance 训练仍未启动。

## 29. 2026-08-14 Validation-only protocol revision

用户将当前验收规则改为 validation-only：三个数据集的 LPIPS 目标统一为 `0.60`，TC 目标
保持 DataVerse `0.060`、OpenVid `0.070`、Seedance `0.100`。`best.pt` 仍由完整 validation
的 target-first selector 选择，未达标候选按 `0.625*(TC/TC_target) + 0.375*(LPIPS/0.60)`
排序；FVD/FID/CLIPScore/SSIM 只作诊断。三个数据集 validation 全部达到双目标后，才统一
启动最终 test Infer；test 不再作为训练期间的 SOTA 判定或候选筛选依据。

原 Stage 12 `20260813_192542_254984` 按旧 LPIPS `0.30` selector 运行至第 1 轮约
`micro_step=3134/3616` 后被协议变更中止。其 `latest.pt` 保留为不可晋升的中断恢复审计点；
本次没有完整 validation，因此没有新的正式 LPIPS，也没有运行 test。新候选必须从受保护的
Stage 11 `best.pt` 以新 selector 重新启动，避免把旧协议的选择结果混入新协议。

## 30. 2026-08-14 新 Validation-only 候选已启动

新协议候选 run id：`20260814_052612_428443`，tmux 会话：
`tardis_dataverse_s12_validation_p1`。配置保持 Stage 12 的 `full_temporal`、
`metric_alignment`、完整 16 帧、512x512、训练 `7232` 条、validation `256` 条、两步
endpoint trajectory、EMA warm start 和单卡显存适配；唯一协议变化是选择尺度改为：

```text
DataVerse TC <= 0.060, LPIPS <= 0.60
OpenVid   TC <= 0.070, LPIPS <= 0.60
Seedance  TC <= 0.100, LPIPS <= 0.60
score = 0.625 * (TC / TC_target) + 0.375 * (LPIPS / 0.60)
```

test 在三个 validation 权重全部锁定前不运行。新 run 的首批事件已确认 selector 由当前代码
构造，未读取旧 run 的 validation/test 结果；训练和验证热路径仍只使用 DataVerse。完整
validation 完成后，只有 target-first selector 晋升的权重才进入最终统一 test 阶段。

## 31. 2026-08-14 Validation-only 审计状态同步

仓库审计锁已同步到当前 v5 协议：三个数据集的 LPIPS 目标均为 `0.60`，test 仅在三个
validation 权重全部锁定后统一报告，不参与训练、调度、选权重或 `protocol-best` 判定。
Stage 11 保护权重按新尺度重算后的 DataVerse 分数为 `0.8288446324`，TC 已通过，LPIPS
距离目标仍差 `0.0753896185`。

当前活动候选仍为 run `20260814_052612_428443`、tmux
`tardis_dataverse_s12_validation_p1`。截至 `2026-08-14T10:38:29Z` 已推进至第 1 轮
`micro_step>=1655/3616`，没有完整 validation，因此不得判断改善或晋升。资源抽样显示显存
`25414/32760 MiB`；计算阶段 GPU 利用率达到 `100%`，批次边界存在数据准备低谷。旧 run
`20260813_192542_254984` 明确标记为协议切换前中断的恢复审计点，不参与新 selector。

当前活动 run 的 `latest.pt` 已在 `micro_step=1536`、`optimizer_step=384` 的梯度累计边界完成
原子更新；该可变恢复点的本次审计 SHA-256 为
`54790c03cc9d11a2ca5cdb13e574b3caf2a4f9fc8163a68886f452574ca0cefd`。它只用于同 run 故障
恢复，后续 checkpoint interval 会覆盖并改变 SHA，不能参与 validation 晋升。

为避免第 1 轮已达标后仍浪费完整第 2 轮预算，已启动 tmux
`tardis_dataverse_target_watch`。该守护只读取本 run 的事件流；仅当完整 validation 写出
`target_pass=true`，并且随后出现对应 `epoch_completed`（说明 epoch checkpoint 已持久化）
时，才向训练主进程发送 `SIGTERM`。未达标、validation 未完成或 checkpoint 尚未落盘时均不
干预。训练主会话仍为 `tardis_dataverse_s12_validation_p1`。

## 32. 2026-08-14 自适应热启动轮数与停训裁决

用户明确授权后续候选不再拘泥于预设热启动轮数：每个候选至少取得一个完整、固定
validation 结果，再依据正式 TC、LPIPS 和综合分决定继续、回滚或切换单变量方向；训练批次
中的损失值仍只用于数值健康诊断，不能作为改善证据。

当前 DataVerse Stage 12 在 `2026-08-14T12:02:04Z` 推进至第 1 轮
`micro_step=2101/3616`，仍没有完整 validation。此时 GPU 利用率为 `100%`，显存为
`27196/32760 MiB`，稳定速度约 `11.1 s/micro-batch`。因此尚不能回答它是否优于早晨的
Stage 11 保护点；权威比较基线保持：

```text
TC        0.03904570764892057
LPIPS     0.6753896184600308
score     0.8288446324
```

tmux `tardis_dataverse_target_watch` 已替换为首轮自适应守护。它只在对应 validation 事件后
继续等待同一 epoch 的 `epoch_completed`，确保 checkpoint 已持久化，然后执行以下预注册
裁决：

```text
target_pass=true                         -> 停止冗余后续轮次
baseline_score - candidate_score < 0.002 -> 判定平台期并停止
否则                                      -> 允许第 2 轮继续
```

这里的 `0.002` 是综合分绝对改善下限，约对应 TC 不变时 LPIPS 至少改善 `0.0032`。守护不会
在 validation 完成前停止进程，也不会修改受保护的 Stage 11 `best.pt`。若 Stage 12 首轮未
达到实质改善，下一单变量候选仍按 Section 27 预注册方向执行：统一训练 objective 与部署生成
路径的 decoded-video `clamp(-1, 1)` 范围，并先补测试后修改实现。

## 33. 2026-08-14 协议校准与恢复游标修复

当前执行规范已与最新 `appendix/开发prompt.txt` 对齐，并以 `t2v_sota.md` 的校准版本为操作
文档：

```text
validation-only target-first:
DataVerse TC<=0.060, LPIPS<=0.60
OpenVid   TC<=0.070, LPIPS<=0.60
Seedance  TC<=0.100, LPIPS<=0.60
score = 0.625*(TC/TC_target) + 0.375*(LPIPS/0.60)
resource target for Train/Infer = 60%-85% VRAM, no fake allocation
```

旧记录中 LPIPS `0.30`、固定两轮、训练期间 test 反馈和 `60%-90%` 显存口径均已降级为审计
历史，不得用于当前选择。用户后续要求不再维护 `worklist.md`，交接只继续更新本文件、候选
账本、事件账本和 `pipeline_state.json`。

为修复 Stage 12 恢复时重新解码已完成 batch 的缺陷，`RemoteClipIterableDataset` 现在使用
共享 `start_batch` 游标；`RemoteDataLoaders.set_start_batch()` 在真实数据集上转发游标，在
旧测试或自定义 dataset 不支持该接口时返回兼容回退。`run_train_epoch_loop()` 优先使用该
游标，只有不支持 seek 的 loader 才消费旧 batch。单 worker、持久化多 worker、DDP 中断恢复和
全仓回归均已验证：`465 passed, 4 warnings`，Ruff、compileall、Shell `bash -n` 通过。

DataVerse Stage 12 已于 `2026-08-14 13:25 UTC`（tmux 本地显示 `21:25`）在 tmux
`tardis_dataverse_s12_validation_p1` 恢复同一 run `20260814_052612_428443`，受保护 Stage 11
权重不变。本次恢复原点：

```text
checkpoint:
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260814_052612_428443/latest.pt
micro_step=2048, optimizer_step=512, next_batch_index=2048
sha256=87c3d8c8d9bb0647eb1baacb88141a95fd64af4852e451071df4d6d3a6d1ecb1
```

恢复保持了同一 dataset、split、source revision、model factory、optimizer/scheduler/EMA、
`full_temporal`、`metric_alignment`、`micro_batch=2`、accumulation `4`、`steps_per_epoch=3616`、
完整 `7232/256` train/validation 和 `512x512x16` 配置，且未设置 `TARDIS_WARM_START`。事件
账本第一条新的 `microbatch` 已核验为 `batch_index=2048`；稳定显存约 `25.2/32.8 GiB`，GPU
利用率 `100%`。首个完整 validation 之前不得运行 test；之后只按 validation score 和
target-first 规则决定停止、回滚或追加下一轮。

## 34. 2026-08-14 t2v_sota 最终校准后运行状态

已再次逐项核对 `appendix/开发prompt.txt`、用户后续汇总要求、`t2v_sota.md` 与当前代码。
`t2v_sota.md` 已新增“2026-08-14 最终校准结果”表，明确当前唯一执行口径：

```text
dataverse -> openvid -> seedance
一个进程一个数据集
完整 7232 train + 256 validation
validation-only TC/LPIPS target-first selector
test 仅在三个 validation 权重全部冻结后运行
Train/Infer 目标显存 60%-85%，禁止伪造占用
```

开发 prompt 末尾保留的 `worklist.md` 条款属于早期要求，已被用户后续明确取消；当前不恢复
该文件，只维护本交接文档、候选/事件账本和 `pipeline_state.json`。

当前 DataVerse run `20260814_052612_428443` 仍在
`tardis_dataverse_s12_validation_p1` 中运行。最近一次人工观测约为：

```text
epoch=1 (zero-based)
micro_step/batch_index ~= 2239 / 3616
optimizer_step ~= 559
VRAM ~= 25172 / 32760 MiB
GPU utilization = 100%
```

该 run 尚未写出完整 validation 事件，`adaptive_decision.json` 仍为 pending。完成本轮后，
只能依据完整 validation 的 TC、LPIPS、target-first 和 `0.002` 实质改善下限决定冻结、停止
或追加候选；在此之前不运行 test、不切换 OpenVid、不修改受保护 DataVerse 权重。

## 35. 2026-08-14 Seedance caption-group split 已正式修复

等待 DataVerse GPU validation 期间，已完成不占训练 GPU 的 Seedance split 前置修复。旧的
record-ID split 会把重复 caption 分散到不同划分；当前实现改为：

```text
seedance only:
NFC + strip caption grouping
-> deterministic caption-group hash order
-> exact subset-sum selection
-> whole caption groups assigned to validation/test
```

DataVerse 和 OpenVid 继续使用原 `record_identity_v1`，其 split ID 未变化。Seedance 的 Train、
Infer 和 curation 现在统一启用 `caption_group_v1`；真实 manifest 已原子重写，但 8,000 条媒体
和总媒体字节均未变化。正式复核结果：

```text
records: train=7232, validation=256, test=512
caption groups: train=6121, validation=208, test=441
caption intersections: train/validation=0, train/test=0, validation/test=0
persisted/runtime split mismatches=0
manifest sha256=a82ebf193a099b93d428b656ed4e19d411078f71b7b3286eec7cc5a2d12109df
split id sha256:
  train=d5871438f252c4677f8321b9ef9da248912781dba9e8b652c1a75feed8f30315
  validation=88861f9fe038df19f17c0b674c7e2299d39fcff91831ea2c91c941ad31e4d051
  test=35a3d9dc68fb3eccc5785ab31a8fcd2946ba22f3a2f910eb902c20786bd7e792
```

`TARDIS_SOTA/configs/split_manifest_lock.json` 已升级到 v2 并写入三个数据集的 split strategy 与
ID hash。相关 focused tests 为 `68 passed`。该修复只准备未来 Seedance 队列，没有启动
Seedance 训练，也没有影响当前 DataVerse run；后者在修复完成时约推进到 `2493/3616`。

## 36. 2026-08-14 当前接管状态复核

本轮重新核对了用户汇总要求、`appendix/开发prompt.txt`、`t2v_sota.md`、`source_audit.md`、
当前代码和运行状态。`t2v_sota.md` 的“2026-08-14 最终校准结果”继续作为活动协议，冲突处理
顺序为：用户最新标准 -> 开发 prompt 的未被后续指令覆盖部分 -> 本交接文档活动章节 ->
经过测试的代码。旧历史中的 LPIPS `0.30`、固定轮数、训练期间 test 和 `60%-90%` 显存口径
均不再具有执行效力。

最新全仓验证命令为：

```text
PYTHONPATH=/home/TARDIS python -m pytest -q
467 passed, 2 skipped, 4 warnings
```

当前唯一活动候选仍是 DataVerse：

```text
run_id=20260814_052612_428443
tmux=tardis_dataverse_s12_validation_p1
epoch=1 (zero-based), batch_index~=2628/3616
train records per epoch=7232
validation records=256
VRAM=27214/32760 MiB (~83.1%)
GPU utilization=100%
```

该候选尚未写出完整 validation 事件，不能根据训练 micro-batch loss 推断 TC/LPIPS 改善，
也不能运行 test 或切换 OpenVid。`tardis_dataverse_target_watch` 继续等待完整 validation
和对应 epoch checkpoint 持久化；之后按 target-first 及受保护分数绝对改善 `0.002` 的预注册
规则停止、冻结或允许下一轮。当前受保护 DataVerse 权重和回滚点不变：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_165601_304501/best.pt
validation TC=0.03904570764892057
validation LPIPS=0.6753896184600308
validation score=0.8288446323955958
```

## 37. 2026-08-14 DataVerse Stage 12 完整 validation 裁决

当前 run `20260814_052612_428443` 已完整遍历 `7232` 条训练记录，并完成 `256` 条 validation。
第 1 个 epoch 的权威六指标为：

```text
TC        0.03886387028845118
LPIPS     0.6804854870715644
FVD       24.34342560601374
FID       210.16996552802294
CLIPScore 0.30861890228965216
SSIM      0.21751924633030684
score     0.8301354115910943
```

TC 达到 DataVerse 阈值 `0.060`，LPIPS 仍高于 `0.60`，因此 `target_pass=false`。相对受保护
分 `0.8288446323955958`，候选分变差 `0.0012907791954984704`，低于预注册的 `0.002` 实质
改善门槛；`tardis_dataverse_target_watch` 在 epoch checkpoint 持久化后给出 `plateau` 并发送
SIGTERM，未运行第 2 轮、未运行 test、未修改受保护 Stage 11 权重，也未切换 OpenVid。

## 38. 2026-08-14 DataVerse Stage 13 预注册

Stage 12 暴露出一个可归因的部署/训练度量范围错配：部署 `TARDISModel.generate()` 对解码视频
执行 `clamp(-1,1)`，而训练 `metric_alignment` objective 原先直接把未裁剪 VAE 输出送入
LPIPS/TC。Stage 13 只改这一点：两个 objective 通过共享 `_decode_video_for_metric()` 在 LPIPS/TC
前执行同样的 `clamp(-1,1)`；模型、数据划分、seed、采样器、loss 权重、全时序 rollout、有效
batch 和 optimizer 均保持不变。行为测试已先失败后通过，当前 focused objective tests 为
`8 passed`，Ruff 与 compileall 通过。

候选已写入 `TARDIS_SOTA/work/experiments/candidate_ledger.jsonl`，父权重仍为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_165601_304501/best.pt
sha256=3dfa630ccd4b2ee3b73bfb21af2b417b43d99b6db1a8ad8a08b39174d0d86772
```

Stage 13 的 validation 选择仍严格使用 DataVerse `TC<=0.060`、`LPIPS<=0.60` 和
`0.625*(TC/0.060)+0.375*(LPIPS/0.60)`；test 继续冻结，只有三个数据集 validation 权重全部锁定
后才允许最终 test。

## 39. 2026-08-14 DataVerse Stage 13 已启动

Stage 13 已从受保护 Stage 11 EMA 权重启动，未加载 Stage 12 optimizer 或 scheduler 状态：

```text
run_id=20260814_184042_288434
tmux=tardis_dataverse_s13_clamp_p1
watcher=tardis_dataverse_s13_target_watch
parent=/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_165601_304501/best.pt
parent_sha256=3dfa630ccd4b2ee3b73bfb21af2b417b43d99b6db1a8ad8a08b39174d0d86772
train=7232 records/epoch
validation=256 records
frames=16, resolution=512x512, diffusion_steps=2
micro_batch=2, accumulation=4, precision=bf16
```

Stage 13 唯一算法变化是训练 LPIPS/TC 输入范围与部署路径一致化；当前初始显存观测为
`24986/32760 MiB`，GPU 采样利用率为 `100%`。在该 run 完成第一个完整 validation 前，
不得读取 test、修改候选或启动 OpenVid。

## 40. 2026-08-14 Stage 13 watcher 修复与回归

Stage 13 初次启动后发现 watcher 的旧进程发现逻辑只在训练命令行包含 run id 时有效；weights-only
warm start 的命令行不包含新 run id，因此旧 watcher 错误写入了
`training_exited_before_epoch_decision`。训练进程本身没有停止，已把决策文件恢复为 pending。

已修复 `watch_validation_candidate.py`：支持显式 `--train-pid`，同时保留旧的 run-id 搜索兼容
路径，并在目标 PID 已退出时安全处理 `ProcessLookupError`。修复后的 watcher 已绑定：

```text
tmux=tardis_dataverse_s13_target_watch
train_pid=367702
run_id=20260814_184042_288434
```

当前全仓回归为 `472 passed, 4 warnings`，Ruff、compileall、Shell 语法检查均通过。Stage 13
训练仍在运行，watcher 的 pending 文件不构成 validation 结果；只有完整 validation 与
`epoch_completed` 事件出现后才会裁决。

## 41. 2026-08-15 Stage 13 运行快照

截至 `2026-08-15T01:47:17Z`，Stage 13 训练进程仍存活，当前第 1 个 epoch 已推进到
`micro_step=2305/3616`，约完成该 epoch 的 `4608/7232` 条 train records；完整 validation
尚未开始，`adaptive_decision.json` 仍为 `pending`。训练没有退出，也没有产生可用于选择的
中途指标。

当前资源采样为显存 `27196/32760 MiB`（约 `83.0%`）和 GPU 利用率 `100%`，落在当前
`60%-85%` 目标区间内。受保护的 Stage 11 `best.pt` 未被修改，OpenVid/Seedance 尚未启动，
test 仍未读取。下一步继续等待 Stage 13 完整 train + validation，再按 `t2v_sota.md` 的
target-first 和 `0.002` 实质改善规则裁决。

为覆盖第二个完整 epoch，已另行启动 `tardis_dataverse_s13_epoch2_watch`，使用同一训练 PID
`367702`，并将结果写入当前 run 的 `adaptive_decision_epoch2.json`；它不会覆盖第 1 轮的
决策文件，也不会改变训练参数。

## 42. 2026-08-15 Stage 13 中止与 Stage 14 快速候选

由于 Stage 13 的 `full_temporal` 训练实测约 `11` 小时/epoch，且尚未进入 validation，已在
保留最新 checkpoint、manifest 和 events 的前提下停止；它标记为
`interrupted_before_validation`，不参与权重选择，也没有读取 test。

为在时间约束下继续优化，已从受保护 Stage 11 EMA 权重启动 Stage 14：

```text
candidate=dataverse-stage14-keyframe-deployment-range-alignment-p1-seed3407
run_id=20260815_021511_391197
tmux=tardis_dataverse_s14_keyframe_p1
parent=/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_165601_304501/best.pt
train_mode=keyframe_only
train_records_per_epoch=7232
training_frames_per_record=1
steps_per_epoch=904
micro_batch=8
gradient_accumulation=1
validation_records=256
validation_frames_per_record=16
```

Stage 14 仍完整遍历 `7232` 条 train records，只有训练阶段使用首帧 keyframe；validation 仍完整
使用 `256` 条记录和 `16` 帧，并每个 epoch 计算六指标。当前启动后约 `184/904` step，实测
显存约 `9.7/32.8 GiB`、GPU 利用率采样 `74%`，预计训练 epoch `42-45` 分钟，validation 约
`7` 分钟。第 1、2 轮 watcher 分别为 `tardis_dataverse_s14_target_watch` 和
`tardis_dataverse_s14_epoch2_watch`。

## 43. 2026-08-15 Stage 14 第 1 轮 validation

Stage 14 已完整遍历第 1 个 epoch 的 `7232` 条 train records 和 `256` 条 validation records。
正式 validation 结果为：

```text
TC        0.039095048329398294
LPIPS     0.6712395780050429
FVD       24.364263986567806
FID       221.78802023688803
CLIPScore 0.3082739710711791
SSIM      0.20517658846693854
score     0.8267648230177174
```

相对受保护 Stage 11 分数 `0.8288446323955958` 改善 `0.0020798093778784388`，超过预注册的
`0.002` 继续阈值，因此 watcher 判定 `continue` 并进入第 2 个 epoch。TC 已达标，LPIPS 仍未
达到 `0.60`，所以 DataVerse 尚未 SOTA；当前仍不读取 test，也不切换 OpenVid/Seedance。

## 44. 2026-08-15 Stage 14 中断与 Stage 15 接管

Stage 14 在第 1 轮 validation 后被用户中断，第 2 轮训练推进到约 30% 时收到
`SIGTERM`。日志中的 `DataLoader worker ... Terminated` 是该外部停止的连带结果，不是
CUDA OOM、NaN 或数据损坏；Stage 14 的 `best.pt`、`latest.pt`、manifest 和 events 全部保留，
但未被提升为新的全局保护权重。

当前由 Agent 重新接管 DataVerse，启动 Stage 15：

```text
candidate=dataverse-stage15-keyframe-lpips24-p1-seed3407
run_id=20260815_032625_092091
tmux=tardis_dataverse_s15_lpips24
parent=/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260813_165601_304501/best.pt
train_mode=keyframe_only
train_records_per_epoch=7232
training_frames_per_record=1
steps_per_epoch=904
micro_batch=8
gradient_accumulation=1
validation_records=256
validation_frames_per_record=16
lpips_loss_weight=24.0
tc_loss_weight=5.0
```

Stage 15 只把 keyframe 感知损失权重从 `12.0` 提升到 `24.0`，保留 Stage 14 已验证的
deployment-range clamp、端点两步采样、trajectory alignment、EMA warm start 和完整
validation 覆盖。当前目标仍是 DataVerse validation `TC<=0.060` 且 `LPIPS<=0.60`；在双目标
达标前不读取 test，也不切换 OpenVid/Seedance。训练与 validation 由 tmux 进程持续运行，
每轮完整 validation 后按 target-first 和固定 weighted score 裁决。

## 45. 2026-08-15 Stage 15 完成与 raw/EMA 裁决

Stage 15 已完成两个完整 epoch，每轮均覆盖 `7232` 条训练记录和 `256` 条完整 validation。
第 2 轮 EMA 六指标为：

```text
TC        0.039199296334955135
LPIPS     0.6638098825933412
FVD       24.949261368143794
FID       238.1354830792735
CLIPScore 0.3020705776004428
SSIM      0.19943219503651505
score     0.8232071801099542
```

同一 `best.pt` 在完全相同的 256 条 validation、seed、512x512x16 prompt-only 协议下关闭 EMA
后得到 `TC=0.039281869378100964`、`LPIPS=0.6581640830263495`、
`score=0.8205386912466869`。raw 比 EMA 的 LPIPS 再低 `0.0056457995669917`，TC 仍低于
`0.060`，因此当前受保护状态选为该 checkpoint 内的 raw temporal state：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260815_032625_092091/best.pt
SHA-256 65118f1f5d20b8e6bd871dcdbb0b25684be26a2118ebb7d2f08a87a363e10867
selected_state=raw
temporal tensors=369
EMA tensors=337
all tensors finite=true
```

raw 评测 JSONL 恰好覆盖 256 个记录。Stage 15 训练资源为 mean GPU utilization `90.29%`、
peak allocated/reserved `20448/26694 MiB`，无 non-finite microbatch。DataVerse 的 TC 已达标，
LPIPS 距离 `0.60` 还差 `0.0581640830`，所以仍不读取 test、不切换数据集。

## 46. 2026-08-15 Stage 16 预注册

Stage 16 从上述 validation 选出的 raw temporal state 做 weights-only 热启动。Stage 15 的
raw/EMA 全量对照已经证明 `ema_decay=0.999` 存在可量化滞后，因此 Stage 16 只把 EMA decay
改为 `0.99`；学习率仍为 `1e-6`，LPIPS 权重仍为 `24`，其它模型、数据、sampler、batch、
完整 train/validation 覆盖均不变。最大预算为 8 个 epoch，每个 epoch 后完整 validation；
一旦持久化的 validation 同时满足 `TC<=0.060` 与 `LPIPS<=0.60`，watcher 即停止冗余训练。

Stage 16 已启动：

```text
run_id=20260815_050509_306798
tmux=tardis_dataverse_s16_fastema
parent_state=raw
ema_decay=0.99
epochs_max=8
train=7232 records/epoch
validation=256 records/epoch
```

manifest 与 `weights_only_warm_start` 事件已核验：父 checkpoint SHA 匹配，`used_ema=false`，
DataVerse split 仍为 `7232/256/512`。已为 epoch 1-8 分别绑定 watcher；任何完整 validation
达到双目标都会在 epoch checkpoint 持久化后停止训练。当前仍未读取 test。

Stage 16 第 1 轮完整 validation 为 `TC=0.0394349113`、`LPIPS=0.6449276879`、
`score=0.8138601315`。相对 raw 父状态的 LPIPS 单轮下降 `0.0132363952`，综合分改善
`0.0066785598`，证明缩短 EMA horizon 后完整 validation 能及时反映 keyframe perceptual
学习。TC 继续达标，LPIPS 尚差 `0.0449276879`；watcher 已允许第 2 轮，配置保持不变。

第 2 轮完整 validation 为 `TC=0.0394988426`、`LPIPS=0.6321557533`、
`score=0.8065436225`。LPIPS 相对第 1 轮再降 `0.0127719346`，距离 `0.60` 还差
`0.0321557533`；TC 仍安全达标。watcher 已允许第 3 轮，未修改超参或读取 test。

## 47. 2026-08-15 Stage 16 第 3 轮 validation

Stage 16 第 3 轮已完整覆盖 `7232` 条训练记录和 `256` 条 validation，六指标为：

```text
TC        0.03955788136844769
LPIPS     0.6231873176438967
FVD       27.71567432625585
FID       333.9134599493622
CLIPScore 0.2628592442055502
SSIM      0.1726340638043693
score     0.8015533377820989
```

TC 持续低于 `0.060`，LPIPS 相对第 2 轮下降 `0.0089684356353245`，距离 `0.60` 还差
`0.0231873176438967`。相对受保护 Stage 15 raw 状态的综合分改善
`0.018985353464587962`。watcher 判定 `continue`，第 4 轮已自动开始；参数保持不变，仍未读取
test，也未切换数据集。

## 48. 2026-08-15 Stage 16 第 4 轮 validation

Stage 16 第 4 轮完整 validation 六指标为：

```text
TC        0.03961333433519384
LPIPS     0.617303926810564
FVD       27.894836794005855
FID       349.96243172246795
CLIPScore 0.2568207524447902
SSIM      0.166578958552175
score     0.7984538535815383
```

LPIPS 相对第 3 轮下降 `0.0058833908333327`，距离目标还差 `0.017303926810564`；TC 仍稳定
达标。相对受保护 Stage 15 raw 状态的综合分改善 `0.02208483766514857`。watcher 继续保留该
有效下降方向，第 5 轮已经启动；搜索仍严格限于 DataVerse validation。

## 49. 2026-08-15 Stage 16 第 5 轮 validation

Stage 16 第 5 轮完整 validation 六指标为：

```text
TC        0.03962308430568236
LPIPS     0.6138230917422334
FVD       28.159552619586695
FID       357.210984258707
CLIPScore 0.25359576883403756
SSIM      0.16838116615512247
score     0.7963798938564204
```

LPIPS 相对第 4 轮下降 `0.0034808350683306`，距离 `0.60` 还差 `0.0138230917422334`；TC
继续达标。watcher 已允许第 6 轮。由于 EMA 改善斜率正在变缓，Stage 16 预算结束后必须在相同
完整 validation 协议下审计最佳 checkpoint 的 raw state，避免 EMA 滞后掩盖可用权重。

## 50. 2026-08-15 Stage 16 收敛裁决与 Stage 17 预注册

Stage 16 第 6 轮完整 validation 为：

```text
TC        0.039656791618401556
LPIPS     0.6111148344280082
FVD       28.277871391230285
FID       363.70414858484804
CLIPScore 0.2520270871946552
SSIM      0.1612548600523533
score     0.7950383508758547
```

同一 `best.pt` 的 raw state 在完全相同的 256 条 validation 上为
`TC=0.03967521227787074`、`LPIPS=0.6111040856922045`、`score=0.7952235147854481`，
所以按锁定 selector 选择 EMA。checkpoint 为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260815_050509_306798/best.pt
SHA-256 cf7c030b6e90be10b5dbbb11680846bf8d258f1f796389f133ce470d0e603303
selected_state=ema
```

第 5→6 轮综合分增益只有 `0.0013415429805657`，低于 `0.002` 实质改善尺度，且余弦学习率
已接近尾部。因此 Stage 16 在 epoch 6 checkpoint 持久化后停止冗余尾段，epoch 7 的部分训练
不参与选择。

Stage 17 预注册为 `dataverse-stage17-ema-cosine-restart-p1-seed3407`。它从上述 EMA 状态做
weights-only 热启动，只重置 optimizer 和余弦学习率周期；模型、损失权重、数据、sampler、
batch、精度以及 `7232 train + 256 validation` 协议全部保持不变。最大预算 4 轮，首次
`TC<=0.060 && LPIPS<=0.60` 即停止并冻结。

Stage 17 已启动：

```text
run_id=20260815_094727_293514
tmux=tardis_dataverse_s17_cosine_restart
parent_state=ema
epochs_max=4
train=7232 records/epoch
validation=256 records/epoch
```

manifest 与首个 `weights_only_warm_start` 事件均已核验，父 SHA 匹配且 `used_ema=true`。
四个 epoch watcher 已绑定训练 PID，目标达标后会在 checkpoint 持久化后停止后续轮次。首次启动
访问 Hugging Face 官方端点超时后已成功回退到本地 SD-Turbo 缓存，不影响训练数据或权重。

Stage 17 第 1 轮完整 validation 为 `TC=0.03970832167873425`、
`LPIPS=0.6046372335986234`、`score=0.791526621819288`。LPIPS 相对 Stage 16 EMA 下降
`0.0064776008293848`，距离目标仅 `0.0046372335986234`；TC 继续达标。watcher 已允许第 2 轮，
未读取 test 或修改超参。

## 51. 2026-08-15 DataVerse validation 达标并冻结

Stage 17 第 2 轮完整遍历 `7232` 条训练记录和 `256` 条 validation 后，得到：

```text
TC        0.03969371772753058
LPIPS     0.5997937173524406
FVD       29.125317824067913
FID       379.87299160393513
CLIPScore 0.2426249523305361
SSIM      0.15607341200321095
score     0.7883472996737189
target    pass
```

该完整 validation 同时满足 DataVerse 锁定阈值 `TC<=0.060` 与 `LPIPS<=0.60`。watcher 在 epoch 2
checkpoint 持久化后发送 `SIGTERM`；训练循环已开始的 1 个 epoch-3 microbatch 不参与选择，也未
覆盖 `best.pt`。冻结权重为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260815_094727_293514/best.pt
SHA-256 676e1735e696f75f218c931f1f024ac7b18da452a4697baca1c4ddf518739199
selected_state=ema
```

checkpoint 审计通过：`epoch=2`、`micro_step=1808`、369 个 temporal model tensors、337 个
EMA shadow tensors，所有 model/EMA tensor 均为有限值，`nonfinite_ledger` 为空，内嵌
`validation_score.target_pass=true`。DataVerse 已标记 `protocol_sota=true`；搜索期间未读取
test，也未启动 OpenVid 或 Seedance。

最终校验：`tests/unit/cli/test_prompt_only_protocol.py` 为 `2 passed`；pipeline state 与 139 条
candidate ledger 均可完整解析；重新计算的 checkpoint SHA-256 与冻结记录一致。当前没有训练、
validation 或 watcher tmux 会话存活。

## 52. 2026-08-15 Seedance 调优启动

用户明确要求跳过 OpenVid，直接接管 Seedance。DataVerse 已冻结且不作为 Seedance 的训练热启动；
Seedance 当前没有历史保护权重，因此从本地缓存的 SD-Turbo 基座独立初始化，权重目录严格隔离：

```text
dataset=seedance
source=/root/autodl-tmp/TARDIS/datasets/seedance-2-prompts-datasets
split_strategy=caption_group_v1
train/validation/test=7232/256/512
tc_target=0.100
lpips_target=0.600
```

首个候选 `seedance-stage1-base-keyframe-p1-seed3407` 使用已验证的 keyframe residual 配置：
`micro_batch=8`、`steps_per_epoch=904`、`LPIPS weight=24`、`TC weight=5`、EMA decay `0.99`、
完整 validation 每轮执行。当前不读取 test；只要完整 validation 同时跨过 Seedance 的两个阈值，
watcher 即冻结对应权重。

用户随后明确授权跨数据集热启动。Stage 1 在首个 validation 前终止，仅保留审计目录；当前改为
`seedance-stage2-dv-ema-keyframe-p1-seed3407`，从 DataVerse 冻结 EMA 权重
`676e1735e696f75f218c931f1f024ac7b18da452a4697baca1c4ddf518739199` 做 weights-only 初始化。
Seedance 的 optimizer、scheduler、EMA、输出目录和 validation selector 独立重置；这是唯一跨数据集
初始化，后续 checkpoint 不回写 DataVerse。

跨数据集开关已加入训练 CLI：默认关闭，Seedance 本次显式使用
`--allow-cross-dataset-warm-start`。相关测试 `75 passed`；训练 manifest 已记录
`allow_cross_dataset_warm_start=true`，事件已核验 `cross_dataset=true`、父 checkpoint SHA 匹配。
当前运行：

```text
run_id=20260815_183735_639349
tmux=tardis_seedance_s2_dv_ema
train/validation/test=7232/256/512
```

## 53. 2026-08-15 Seedance validation 达标并冻结

跨数据集热启动候选在第 1 轮完整遍历 `7232` 条训练记录和 `256` 条 Seedance validation 后，
得到：

```text
TC        0.0802350501249348
LPIPS     0.5962811568679172
FVD       52.6357348156601
FID       390.81224098623557
CLIPScore 0.22172699807385443
SSIM      0.12939932629203413
score     0.8741447863232907
target    pass
```

该结果同时满足 Seedance 锁定阈值 `TC<=0.100` 与 `LPIPS<=0.600`。watcher 在 epoch 1
checkpoint 持久化后发送 `SIGTERM`；随后已开始的 1 个 epoch-2 microbatch 不参与选择，也没有
覆盖 `best.pt`。冻结权重为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/seedance/20260815_183735_639349/best.pt
SHA-256 683cb741c75a40ad2846676011fbdafd18c24df8793a8f6de1f61201ae525ceb
selected_state=ema
```

checkpoint 审计通过：`epoch=1`、`micro_step=904`、369 个 temporal model tensors、337 个
EMA shadow tensors，所有 model/EMA tensor 均为有限值，`nonfinite_ledger` 为空，内嵌
`validation_score.target_pass=true`。本次搜索只使用 Seedance train/validation，未读取 test；
DataVerse 权重保持字节不变。下一阶段按用户授权，以该 Seedance EMA 做 weights-only 跨数据集
初始化，独立训练和选择 OpenVid 权重。

## 54. 2026-08-15 OpenVid 跨数据集热启动达标并冻结

OpenVid 使用 Seedance 冻结 EMA 做 weights-only 跨数据集初始化，独立重置 optimizer、scheduler、
EMA 与 selector。第 1 轮完整遍历 `7232` 条训练记录和 `256` 条 OpenVid validation 后得到：

```text
TC        0.03744649342531595
LPIPS     0.5708089759355062
FVD       30.729247682689426
FID       409.7643564883564
CLIPScore 0.22366745833254004
SSIM      0.030536155722354248
score     0.6910993012571552
target    pass
```

该结果同时满足 OpenVid 锁定阈值 `TC<=0.070` 与 `LPIPS<=0.600`。watcher 在 epoch 1
checkpoint 持久化后发送 `SIGTERM`；随后已开始的 1 个 epoch-2 microbatch 不参与选择，也没有
覆盖 `best.pt`。冻结权重为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/openvid/20260815_193107_986644/best.pt
SHA-256 f9d6cc758b74d095b7ac9bcc40690efa609ee073c710668f545e1a1a23b8a794
selected_state=ema
```

checkpoint 审计通过：`epoch=1`、`micro_step=904`、369 个 temporal model tensors、337 个
EMA shadow tensors，所有 model/EMA tensor 均为有限值，`nonfinite_ledger` 为空，内嵌
`validation_score.target_pass=true`。训练事件的 `run_finished.error_type=null`；日志末尾的
DataLoader worker abort 仅发生在 SIGTERM 清理阶段，不影响已持久化 checkpoint。当前三个数据集
均已达到本项目 validation-only protocol-best，搜索期间未读取 test。


## 55. 交付核验状态

当前无训练进程、无 validation watcher、无 tmux 会话，GPU 已释放。三份冻结权重如下：

| dataset | checkpoint | validation TC | validation LPIPS | SHA-256 |
|---|---|---:|---:|---|
| DataVerse | /root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/20260815_094727_293514/best.pt | 0.0396937177 | 0.5997937174 | 676e1735e696f75f218c931f1f024ac7b18da452a4697baca1c4ddf518739199 |
| Seedance | /root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/seedance/20260815_183735_639349/best.pt | 0.0802350501 | 0.5962811569 | 683cb741c75a40ad2846676011fbdafd18c24df8793a8f6de1f61201ae525ceb |
| OpenVid | /root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/openvid/20260815_193107_986644/best.pt | 0.0374464934 | 0.5708089759 | f9d6cc758b74d095b7ac9bcc40690efa609ee073c710668f545e1a1a23b8a794 |

每份 checkpoint 均通过 369 个 model tensors、337 个 EMA shadow tensors 的 finite audit，且 nonfinite_ledger 为空。最后一次全套测试：475 passed, 4 warnings in 119.66s。本轮按最新协议只使用 validation 选择权重；test 未启动、未读取、未参与调度或 SOTA 判定。

## 56. 2026-08-16 三数据集正式 test 与交付

用户在三份 validation 权重冻结后显式授权最终 test。三个 Infer 均使用各自 EMA checkpoint、
固定 `512` 条 test、`512x512`、16 帧、BF16，并与训练 manifest 对齐为 `endpoint`、
`history_fallback=1.0`、`lite_max=0.75`、keyframe-lite alignment 和 sampler-trajectory
alignment。test 未回流训练、调度、checkpoint 选择或超参数搜索。

| dataset | completed/expected | TC | LPIPS | threshold | pass |
|---|---:|---:|---:|---|---|
| DataVerse | 512/512 | 0.0361613201 | 0.5961036682 | 0.060/0.600 | yes |
| Seedance | 512/512 | 0.0786981254 | 0.5915643804 | 0.100/0.600 | yes |
| OpenVid | 512/512 | 0.0376308897 | 0.5679612435 | 0.070/0.600 | yes |

三源宏平均为 `TC=0.0508301117`、`LPIPS=0.5852097640`。每个输出目录均有 0 failures、
`metrics.csv`、`metrics.xlsx`、512 条逐视频明细及 5 个展示 MP4。Seedance 首次遇到一条
135,295,078 字节的直连本地 MP4，超过 128 MiB 内存读取上限；数据层已改为对超大 `file://`
媒体直接按路径采样解码，并通过回归测试和断点恢复补齐至 512/512。

正式交付入口与权重配置已同步。`TARDIS_SOTA/weights`、`infer_outputs` 和 `scripts` 均通过
符号链接引用数据盘本体；汇总位于 `TARDIS_SOTA/reports/final_test_metrics.csv`，机器可读
manifest 位于 `TARDIS_SOTA/delivery_manifest.json`。三个标准 Shell 与全局 Python 默认架构均
锁定正式权重配置，避免无参 Infer/Apply 回落到不兼容的旧采样轨迹。

最终回归：`python -m pytest -q` 为 `477 passed, 4 warnings in 128.02s`；Ruff、六个正式
Shell 的 `bash -n`、交付 JSON/CSV、checkpoint SHA、自动 checkpoint 发现和符号链接审计均通过。

## 57. 2026-08-17 论文初始实验数据包与正式 benchmark 队列

已按《实验方案》建立可审计的初始数据包：

```text
/home/TARDIS/RTVD-TC-DataPackage-v1.0
/home/TARDIS/RTVD-TC-DataPackage-v1.0/README.md
```

初始包冻结了三个完整 test split 的 TARDIS 正式结果、1,536 条推理 ledger、六项聚合指标、
延时与资源数据，并导入了 5 个真实 prompt-only pilot。`exp01_main_comparison.xlsx` 现在包含
`measured_runs`、`pilot_runs`、`paper50_runs`、逐视频 TC/LPIPS 和逐帧 TC/LPIPS；所有空实验板块
继续明确标记 `planned`，未填造 benchmark 或用户研究数值。包内 benchmark 原始
`metrics.json`、`per_video.jsonl`、`run_manifest.json` 归档在 `06_logs/benchmark_runs/`。

完整性校验命令：

```bash
python /home/TARDIS/RTVD-TC-DataPackage-v1.0/05_scripts/verify_package.py
```

正式队列实现于 `tardis/experiments/queue.py`，固定运行 4 个当前协议兼容方法
（TARDIS、SD-Turbo independent、AnimateDiff-Lightning、Text2Video-Zero）x 3 数据集 x
5 seeds x 50 test records，统一 512x512、16 帧和六指标。source-conditioned 方法在没有等价
prompt-only 适配器前保持 `N/A`。队列强制使用本地离线缓存，每完成一个单元自动刷新数据包，
支持 benchmark ledger/state 断点恢复。

当前后台入口：

```text
tmux session: tardis-paper
launcher: /home/TARDIS/scripts/run_paper_experiments.sh
queue state: /home/TARDIS/TARDIS_PAPER_EXPERIMENTS/queue_manifest.json
raw runs: /home/TARDIS/TARDIS_PAPER_EXPERIMENTS/main/
```

首个正式 `SD-Turbo/DataVerse/seed3407` 已完成 50/50，耗时约 100 秒：
`TC=0.4249805`、`LPIPS=0.7580533`、`FVD=92.8403`、`FID=186.1105`、
`CLIPScore=0.3345191`、`SSIM=0.0880385`。交接时队列仍在继续，实时进度以
`queue_manifest.json` 为唯一来源，不要依据本段静态计数判断完成度。

新增实验代码验证：`python -m pytest -q tests/unit/experiments` 为 `13 passed`；
`ruff check tardis/experiments tests/unit/experiments` 通过；数据包刷新和 SHA256 校验均通过。

## 58. 2026-08-17 正式 paper50、消融与 source50 扩展

prompt-only 正式队列已经完成 `60/60`：TARDIS、SD-Turbo independent、AnimateDiff-Lightning、
Text2Video-Zero x DataVerse/Seedance/OpenVid x 5 seeds x 50 records，所有单元 `returncode=0`、
`50/50`、六指标完整。当前 `exp01` 已计算 18 项 TARDIS-vs-baseline 的 paired bootstrap CI、
单侧 Wilcoxon 和 Holm 校正；TC/LPIPS 为 `18/18` 主指标胜出，但 FID/CLIPScore 不声称全面胜出。

DataVerse A0-A10 prompt-only 消融也已完成 11/11。A4/A5 首次失败是代码部署时 generator
签名尚未同步，随后只重跑这两个单元并通过。结果已写入 `exp04_ablation.xlsx`，需要注意：
prompt-only 下 source-motion 相关开关没有真实 source 条件，不能把该表误读为源运动机制的因果
证据。

为对齐赛题原文的 source video + prompt 任务，新增独立 `source50` 协议和六个核心机制复现：
`streamdiffusion_img2img`、`rerender_flow`、`tokenflow_core`、`vid2vid_zero_core`、
`controlvideo_canny`、`stablevideo_propagation`，以及 TARDIS `generate_source_conditioned`。
所有方法固定 source strength `0.45`，source video 同时作为条件与指标 reference；非官方原仓
执行的适配器在 provenance 里明确写 `audited core-mechanism reproduction`，不能冒充官方原码。

source50 正式队列：

```text
tmux: tardis-source
state: /home/TARDIS/TARDIS_SOURCE_EXPERIMENTS/queue_manifest.json
raw: /home/TARDIS/TARDIS_SOURCE_EXPERIMENTS/main/
protocol: source50, 105 units, 7 methods x 3 datasets x 5 seeds
```

source pilot 已全部适配器跑通；正式队列启动后进度约 `16/105`，失败数为 0。正式完成后执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 \
python -m tardis.experiments.package --refresh
python /home/TARDIS/RTVD-TC-DataPackage-v1.0/05_scripts/verify_package.py
```

source A0-A10 消融入口为 `/home/TARDIS/scripts/run_source_ablations.sh`，应在 source50 主队列
完成后启动，避免单卡争用。新增验证为 `18 passed`，Ruff 已通过。
