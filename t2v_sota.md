# TARDIS 无外部 Baseline 的 SOTA 调优与交付 Pipeline

## 0. 文档用途与任务定义

本文是供下一个 Agent 直接执行的 TARDIS 调优规范。工程技术事实以 `handoff.md`、TARDIS 仓库代码和测试为准；本文负责定义从现有框架开始，经过三个数据集独立训练、验证集调优、正式测试和交付打包的完整 Pipeline。

## 0.1 最新项目标准覆盖规则

本文中的旧实验记录、示例预算和交付草图，均受以下当前项目标准覆盖：

1. 优先级依次为：用户最新要求与 `appendix/development_prompt.txt`、`handoff.md` 的最新审计章节、已测试的当前代码和测试、本文其余内容。发现冲突时必须按此顺序执行，并把冲突写入 `TARDIS_SOTA/reports/source_audit.md`。
2. 三个数据集独立运行：一个 Train/Infer 进程只允许使用一个数据集；最终报告可以离线提供三个数据集的展示平均值，但该平均值不能进入训练、调度、权重选择或 SOTA 判定。
3. 训练每个 epoch 必须完整遍历该数据集的 `7232` 条 train records，并在同一 epoch 后完整遍历 `256` 条 validation records；不能把截断、抽样或 batch 诊断指标写成正式 validation 结果。
4. 当前 validation-only 选择协议固定为：DataVerse `TC<=0.060`、OpenVid `TC<=0.070`、Seedance `TC<=0.100`，三个数据集均为 `LPIPS<=0.60`；分数为 `0.625*(TC/TC_target)+0.375*(LPIPS/0.60)`，越低越好，达标状态优先。
5. 按用户最新覆盖，本轮交付只以三个数据集的完整 validation 判定 SOTA；不启动、不读取、不汇报 test 集合，test 不得影响训练、学习率、early stopping、checkpoint、超参或候选顺序。
6. 训练和评测只从 `/root/autodl-tmp` 的本地数据盘及其符号链接读取；正式热路径不得访问 Hugging Face、hf-mirror 或其它远端数据源。
7. 热启动轮数由 Agent 根据完整 validation 的改善幅度和机制诊断决定，不机械跑固定 2、10、12 或 20 轮。每个候选至少完成一个完整 validation；若达到目标或出现实质平台期，应在 checkpoint 持久化后停止；若有可复现改善，才继续下一轮。
8. Train 和 Infer 的显存目标为总显存约 `60%-85%`，GPU 利用率应尽量保持高位；不得通过无用张量伪造占用。每次训练和评测都必须记录 allocated/reserved/peak VRAM、GPU 利用率、功耗和吞吐。超出区间的正式长跑应先调整真实 batch、累积、checkpointing、worker/prefetch 或模型计算配置，再重新启动。
9. 受保护 current best、完整审计日志和可回滚权重不得因候选失败而删除。清理只允许删除已归档且明确标记为可再生的中间物。
10. 当前不再维护或重建 `worklist.md`；持续交接只维护 `handoff.md`、候选账本、事件账本和 `pipeline_state.json`。历史文档中要求固定轮数、训练期间反复运行 test、LPIPS 目标 `0.30` 或跨数据集联合训练的内容均仅作审计记录，不再具有执行效力。

## 0.2 2026-08-14 最终校准结果

下表是本轮接管后逐项核对 `appendix/development_prompt.txt`、用户后续明确要求、`handoff.md`、
当前代码与测试所得的唯一有效执行协议。后文历史记录若与本表冲突，一律以本表为准。

| 项目 | 当前唯一有效规则 |
|---|---|
| 数据集队列 | `dataverse -> openvid -> seedance`，当前数据集未达标前不启动下一个数据集的正式调优 |
| 进程隔离 | 一个 Train/Infer 进程只允许使用一个数据集，checkpoint、日志和输出按数据集物理隔离 |
| 正式 epoch | 完整遍历 `7232` 条 train records，随后完整遍历同源 `256` 条 validation records |
| 选择依据 | 只使用完整 validation 的 TC 与 LPIPS；test、其余四指标和跨数据集平均均不得参与选择 |
| 达标阈值 | DataVerse `TC<=0.060`，OpenVid `TC<=0.070`，Seedance `TC<=0.100`；三者均 `LPIPS<=0.60` |
| 未达标排序 | `0.625*(TC/TC_target)+0.375*(LPIPS/0.60)`，`target_pass=true` 优先，其后分数越低越好 |
| 热启动预算 | 不设机械固定轮数；每个候选至少完成一个完整 validation，再按达标、实质改善和平台期决定继续或停止 |
| Test 时机 | 本轮按用户覆盖不执行 test；SOTA 和交付判定只使用各数据集完整 validation |
| Infer 产物 | 每个独立 Infer 计算六指标和 per-video 明细，仅随机保存当前测试集默认 `5` 条展示 MP4 |
| 数据路径 | Train/Validation/Infer 只读取 `/root/autodl-tmp` 本地数据盘及仓库符号链接，热路径禁止远端访问 |
| 资源目标 | Train/Infer 显存占总显存 `60%-85%`，GPU 利用率尽量保持高位，并记录真实资源与吞吐数据 |
| 状态维护 | 不维护或恢复 `worklist.md`；持续维护 `handoff.md`、事件账本、候选账本和 `pipeline_state.json` |

`appendix/development_prompt.txt` 末尾仍保留的 `worklist.md` 句子属于早期需求，已被用户后续“无需
worklist，继续维护 handoff”的明确指令覆盖。除此之外，当前调优协议与开发 prompt 的最新
数据、接口、validation-only、资源和本地存储要求一致。

本任务与图像重建 SOTA Pipeline 的关键差异是：

1. 没有 `创新表` 输入。TARDIS 主网络、方法叙事和代码框架已经确定，禁止只靠改名制造“新方法”。
2. 没有可直接比较的外部 baseline 表或同协议公开排行榜。
3. 每个数据集已有严格隔离的 train、validation、test 三个 split。
4. 需要在 `dataverse`、`openvid`、`seedance` 三个数据集上分别得到一份最优权重。
5. checkpoint 只能由 validation 选择，test 只允许对冻结候选做一次正式 Infer。
6. 最终输出仍是一个可开箱复用的唯一交付包，其中包含三份权重、完整相关代码、正式指标、资源报告和使用说明。

Agent 的职责不是只写方案，而是持续接管代码审计、实验设计、训练、监控、恢复、验证、筛选、正式 Infer、清理和打包，直到满足本文的交付定义。

---

## 1. 输入、工程根目录与唯一输出

### 1.1 输入

本 Pipeline 没有创新表和 baseline 文件输入。唯一工程输入是已经构建好的 TARDIS 代码仓库，默认：

`TARDIS_ROOT=/home/TARDIS`

执行前必须依次读取：

- `handoff.md`
- `<TARDIS_ROOT>/README.md`
- `<TARDIS_ROOT>/appendix/development_prompt.txt`
- `<TARDIS_ROOT>/docs/datasets.md`
- `<TARDIS_ROOT>/docs/train.md`
- `<TARDIS_ROOT>/docs/infer.md`
- `<TARDIS_ROOT>/docs/apply.md`
- 实际模型、训练、metric 和 checkpoint 代码

如果文档与代码不一致，以经过测试验证的接口契约为准，并把差异写入 `reports/source_audit.md`，不得静默猜测。

### 1.2 唯一输出

本轮只允许一个业务交付目录，默认：

`<TARDIS_ROOT>/TARDIS_SOTA/`

用户另行指定包名时以用户指定为准。所有训练代码副本、配置、日志、候选权重和中间输出必须位于该目录的 `work/` 中，不得在工程根目录新增零散的 `stage*`、`iter*`、`checkpoints_new*` 或 `outputs_new*`。

最终包必须包含：

- `dataverse`、`openvid`、`seedance` 各一份唯一最优 temporal checkpoint。
- 与三份权重严格匹配的 TARDIS 代码和配置。
- Train、Infer、Apply 及 queue wrapper。
- 三个完整 test split 的六指标报告。
- validation 搜索、SOTA 自决、数据泄漏、资源和实时性能报告。
- README、MODEL_CARD、manifest、环境依赖和 SHA-256。

### 1.3 统一启动契约

交付包必须提供：

```bash
bash <TARDIS_ROOT>/TARDIS_SOTA/scripts/run_sota_queue.sh \
  --project-root <TARDIS_ROOT> \
  --output <TARDIS_ROOT>/TARDIS_SOTA \
  --datasets dataverse,openvid,seedance \
  --resume
```

`run_sota_queue.sh` 是最终交付时的队列包装器。当前调优阶段若该包装器尚未生成，以正式 `scripts/train.sh`、`scripts/infer.sh`、`scripts/apply.sh` 加 tmux 自适应守护和 `TARDIS_SOTA/work/pipeline_state.json` 为权威工作流，不得假设一个不存在的脚本已经可用。队列包装器只负责按 queue 调度已有正式 `torchrun` 入口、记录状态和恢复，不得绕过正式 Shell 入口直接把普通 `python` 命令冒充正式流程。

运行原入口时，launcher 必须显式设置：

```text
TARDIS_CHECKPOINT_ROOT=<TARDIS_ROOT>/TARDIS_SOTA/work/checkpoints
TARDIS_OUTPUT_ROOT=<TARDIS_ROOT>/TARDIS_SOTA/work/outputs
```

若实际 CLI 将 Train、Infer、Apply 输出拆成不同环境变量，则逐项映射到 `work/` 下对应 namespace。最终确认权重后再原子复制到 `weights/`，正式 Infer 和 Apply 的交付产物分别整理到 `infer_outputs/` 与 `apply_outputs/`。

---

## 2. 不可变接口和模型边界

### 2.1 正式入口

权威入口是：

- Train：`scripts/train.sh`
- Infer：`scripts/infer.sh`
- Apply：`scripts/apply.sh`

三者内部均使用 `torchrun --standalone --nproc_per_node`。允许在交付包内建立 wrapper 或 launcher，但必须保持环境变量、数据隔离、checkpoint 和输出语义兼容。

### 2.2 Train / Infer / Apply 边界

- 一个 Train 进程只训练一个数据集的 train split，并只用同一数据集的 validation split 选权重。
- 一个 Infer 进程只完整评测一个数据集的 test split。
- 三个数据集不能在一个 Train/Infer 进程内混合，不能用跨数据集平均选择 checkpoint。
- Apply 是纯 prompt-to-video，只接收 prompt、style、时长、采样配置和权重，不接收 source video 或 test label。
- 三个数据集的 checkpoint、日志、Infer 输出和 Apply 输出必须物理命名空间隔离。
- 训练和评测热路径只读取本地 manifest 与本地归档，不访问远端。

### 2.3 固定主网络

主模型仍是 `TARDISModel`，核心机制保持：

- frozen SD-Turbo VAE、text conditioner 和 first-frame prior；
- PromptMotionScaffold；
- MotionStateTransport；
- Transport-Orbit Quotient；
- risk router 与 Innovation Proper Time；
- LiteResidualCorrector 与 SparseResidualDiT；
- CausalStateUpdater；
- CRCD 与 metric alignment curriculum。

没有创新表意味着 Agent 不需要重命名网络或重新编造方法层级。优先调节和修复现有机制，只有诊断证明确有结构瓶颈时才允许增量改动内部实现。公共类名、Train/Infer/Apply 契约、prompt-only 边界和 A0-A10 消融语义不得被破坏。

### 2.4 冻结先验与可训练参数

`FrozenPriorBundle` 按设计保持冻结，不写入 temporal checkpoint。这里的“全参数训练”指全部 TARDIS temporal 参数参与 backward，不包括按方法定义永久冻结的 SD-Turbo 先验。

最终代码必须同时提供：

- `train_mode=full_temporal`：全部 temporal 参数可训练。
- `train_mode=selective`：按 motion、transport、quotient、router、clock、residual、state 等组选择性训练。

每个 stage 必须记录总 temporal 参数、可训练参数、冻结参数和参数组。无论选择哪种模式，都必须执行完整 causal forward；不得通过永久旁路模块制造虚假的训练提速。

---

## 3. 三个数据集及固定划分

canonical 数据集名称固定为：

```text
dataverse
openvid
seedance
```

默认 `split_seed=3407`、`validation_size=256`、`test_size=512` 时：

| dataset | split strategy | train | validation | test |
|---|---|---:|---:|---:|
| dataverse | `record_identity_v1` | 7232 | 256 | 512 |
| openvid | `record_identity_v1` | 7232 | 256 | 512 |
| seedance | `caption_group_v1` | 7232 | 256 | 512 |

开始实验前必须冻结并保存：

- 三个源 manifest 的路径和 SHA-256。
- stable split 算法版本、`split_seed`、validation/test size。
- 每个 split 的 record id、media hash 和 caption hash 清单。
- 视频解码、帧数、FPS、resize、颜色范围和异常样本策略。

必须证明每个数据集内 train/validation/test 的 id、media hash、caption hash 交集均为空；也必须检查不同数据源之间是否存在重复媒体或 caption 泄漏。

DataVerse 和 OpenVid 使用 `record_identity_v1`，固定本地 manifest 已通过该审计。Seedance
原 record-ID split 的 caption 泄漏已于 2026-08-14 修复：Train、Infer 与 curation 统一使用
`caption_group_v1`，按 NFC+strip caption 整组分配，并通过确定性 exact subset-sum 保持
`7232/256/512`。新 manifest SHA、split strategy 和三个 split ID SHA 已原子写入
`TARDIS_SOTA/configs/split_manifest_lock.json` v2；三个 split 的 caption 交集和持久化/运行时
split mismatch 均为零。旧 Seedance split 与旧权重仅作审计，不得与新 split 混用。

正式训练和 Infer 禁止设置 `TARDIS_CATALOG_RECORD_LIMIT`、`TARDIS_OPENVID_ARCHIVE_LIMIT` 或其他样本截断参数。诊断性短跑允许限量，但产物必须显式标记 `diagnostic_only=true`，绝不能用于正式 SOTA 结论。

---

## 4. 统一质量、采样和资源口径

三个数据集必须使用同一套质量和 metric 实现：

- 正式分辨率：512x512。
- 正式 FPS：30。
- 固定并记录 `NUM_FRAMES`、duration、采样 seed 和 diffusion/residual steps。
- TC：normalized RGB 上的官方帧差分误差，越低越好。
- LPIPS：AlexNet LPIPS v0.1，逐帧后 macro 聚合，越低越好。
- FVD：越低越好。
- FID：越低越好。
- CLIPScore：越高越好。
- SSIM：越高越好。
- 正式报告使用每个数据集 test split 的 macro 结果，并保留 per-video details。

主优化目标及 checkpoint 权重固定为：

```text
TC      0.625，越低越好
LPIPS   0.375，越低越好
```

FVD、FID、CLIPScore、SSIM 仅用于记录和展示，不参与 checkpoint 选择或达标判定。

竞赛实时目标必须实测：

- steady-state 30 FPS；
- 单帧小于 33.3 ms；
- 记录首帧、steady-state、p50/p95 per-frame、端到端 MP4 编码时间；
- 记录 allocated/reserved/peak VRAM、GPU utilization 和功耗。

未实测时只能写“未验证”，不能写“满足实时约束”。

---

## 5. 无外部 Baseline 时如何定义 SOTA

### 5.1 SOTA 名称边界

三个数据集没有完全相同协议的公开排行榜，也没有用户提供的 baseline 标准。因此最终结论只能称为：

`TARDIS protocol-best` 或 `本工程固定协议下的内部 SOTA`

除非后来获得可核验且完全同协议的公开结果，否则不得宣称“超过公开领域 SOTA”或“超过所有 baseline”。最终 MODEL_CARD 和论文记录必须保留这个限定。

### 5.2 不使用 baseline 的稳定选择分数

现有 selector 若依赖“相对冻结 baseline”做归一化，应在交付代码副本中改为“固定目标尺度归一化”，同时保留 validation-only 限制。下面的工程目标既是归一化尺度，也是各数据集唯一的达标阈值；它们不是已取得结果，也不是外部 baseline：

| dataset | TC scale/target | LPIPS scale/target |
|---|---:|---:|
| dataverse | 0.060 | 0.60 |
| openvid | 0.070 | 0.60 |
| seedance | 0.100 | 0.60 |

对某数据集，预先锁定：

```text
validation_score = 0.625 * (TC / TC_scale)
                 + 0.375 * (LPIPS / LPIPS_scale)
```

分数越低越好。scale 在搜索开始前写入 `configs/selection_scale_lock.json`，搜索过程中禁止修改。加权分数只用于未达标阶段的候选排序，不构成额外达标条件。

### 5.3 初始 incumbent，而非外部 baseline

每个数据集首先用当前默认 TARDIS A10 配置完成一个可复现训练，形成 `initial incumbent`。它只用于回答“本轮调优是否真实改善了现有框架”，不作为外部论文 baseline，也不进入最终交付包的 baseline 声明。

若当前目录没有权重，必须从头完成初始训练；禁止借用另一个数据集的 temporal checkpoint。三个数据集可以共享同一冻结先验和同一初始化规则，但不能跨数据集 resume。

### 5.4 单数据集内部 SOTA 判定

一个数据集的候选权重在固定的完整 validation split 上同时满足以下两个条件，即标记
`target_pass=true`，并可称为该数据集的 `TARDIS protocol-best`：

1. `TC <= TC target`。
2. `LPIPS <= LPIPS target`。

这是当前训练阶段唯一的质量达标规则。多 seed、bootstrap、Pareto 收敛、FVD、FID、CLIPScore、
SSIM、防静止检查、延迟和资源数据均不作为 validation SOTA 门槛；这些信息继续记录用于诊断
和展示。每个候选只用完整 validation 选 `best.pt`，test 不在候选之间重复运行。

### 5.5 三数据集整体完成

只有 dataverse、openvid、seedance 三行均为 validation `target_pass=true`，训练阶段的
`sota_acceptance_all` 才能为 `true`。不得用三数据集平均掩盖某一个数据集失败，也不得用
一个数据集的权重代替另一个数据集。

---

## 6. 科研完整性与防作弊

以下任一行为会使结果无效：

- test 指标参与学习率、loss、结构、early stopping、checkpoint 或 seed 选择。
- 反复运行 test 后根据结果继续调参。
- 在 prompt-only Infer/Apply 中读取 source video 或 test label。
- 输出 GT、复制 label video、按 test label 拟合后处理或逐样本选择权重。
- 通过 static video、重复首帧或极低运动幅度人为降低 TC，却隐藏 LPIPS 和运动诊断。
- 删除失败测试样本，只计算成功或好看的样本。
- 改变 test split、seed、帧数、分辨率、颜色范围或 aggregation 后仍直接比较。
- 在一个进程混合三个数据源，或汇总平均后选择 checkpoint。
- 把工程内部 protocol-best 写成同协议公开排行榜 SOTA。
- 把目标值或预期值写成已测结果。

正式 Infer 可以读取 label video 计算指标，但 label 只能进入 metric evaluator，不能进入 `model.generate()`、采样器、校准器或输出生成路径。

---

## 7. 开始训练前的协议锁定

### 7.1 源代码和环境审计

开始前必须：

1. 记录原仓库关键文件、三个正式 Shell 入口和依赖锁的 SHA-256。
2. 运行现有 unit/integration/mechanism tests，并记录通过、失败和 skip 数。
3. 运行 Ruff/format/compileall、Shell `bash -n` 和适用的静态检查。
4. 确认 CUDA、驱动、PyTorch、xformers/attention、ffmpeg、LPIPS、I3D、CLIP 和 SD-Turbo prior 可用。
5. 确认数据、checkpoint、output 符号链接指向本地数据盘。
6. 检查热路径不会访问 Hugging Face 或其他远端。

原仓库可能包含用户的未提交修改，不得使用 destructive git 命令回退。调优代码应复制到唯一任务目录的 `work/source/` 或使用明确的工作分支；最终只打包实际训练权重对应的代码快照。

### 7.2 metric sanity controls

正式搜索前，在 validation 上建立以下 sanity controls：

- identity/label upper bound，仅验证 metric 实现，不计入成绩。
- static video。
- repeated first frame。
- independent SD-Turbo frames。
- 当前默认 TARDIS A10。

这些不是本轮外部 baseline，也不用于宣称超过其他方法；它们只用于发现 TC 尺度错误、静态化作弊、LPIPS 范围错误和时序聚合问题。

### 7.3 机制诊断

每个数据集至少记录：

```text
raw frame residual energy
TAR residual energy
quotient-normal residual energy
tar_to_raw_ratio
quotient_to_tar_ratio
tangent_explained_ratio
router ECE / Brier
active token ratio
motion magnitude / temporal variance
closed-loop drift slope
```

应验证：

```text
E[|r_TAR|] / E[|r_raw|] < 1
E[|r_perp|] < E[|r_TAR|]
```

不成立时优先修正 flow、visibility、latent alignment 或 quotient basis，不能靠盲目加宽网络掩盖机制失败。

---

## 8. 委员会竞投与候选台账

每个新 stage 前，由四个委员会提出候选：

1. 机制委员会：motion、transport、TOQ、router、proper time、causal state。
2. 生成委员会：LiteCorrector、SparseResidualDiT、diffusion steps、text conditioning、closed-loop rollout。
3. 目标委员会：TC、LPIPS、warp、CRCD、drift、router/budget、text loss 及动态权重。
4. 系统委员会：optimizer、LR、EMA、AMP、checkpointing、active ratio、延迟、显存和吞吐。

候选写入 `work/experiments/candidate_ledger.jsonl`，至少包含：

- dataset、stage、seed 和 parent checkpoint SHA-256。
- 假设、代码/config 改动和机制依据。
- 预期改善 TC/LPIPS 的原因。
- 可能退化的 FVD/FID/CLIPScore/SSIM 或实时性能。
- 训练预算、可训练参数、显存和耗时预估。
- 短探针淘汰条件、完整 validation 判定条件和回滚方案。
- 真实结果、TC/LPIPS 加权分、是否达标和未采用原因。

按以下顺序竞投：

- 是否直接解决当前最大验证缺口或机制失败。
- 是否能在短 probe 中被证伪。
- 是否复用当前已收敛权重。
- 对已改善指标的回退风险。
- 与 TARDIS 统一机制和 A0-A10 消融的可解释性。
- 训练、显存、延迟和实现成本。

每次优先执行一个候选或一个强耦合、可归因的小组合。禁止在同一 stage 无记录地同时改变数据 split、metric、网络、loss 和采样，造成结果无法解释。

---

## 9. TARDIS 定向调优候选池

### 9.1 Motion 与 Transport

- FlowMotionTeacher 监督质量、PromptMotionScaffold 容量和可部署 motion gap。
- backward flow 范围、visibility calibration、valid mask 和 warp 边界。
- transport correction magnitude、遮挡区域和 scene-cut reset。
- teacher flow 到 prompt-predicted flow 的 curriculum 转换速度。
- 先确保 TAR residual energy 下降，再扩大生成模块。

### 9.2 TOQ、Router 与 Proper Time

- `quotient_regularization`、rank threshold、basis 退化处理。
- `active_ratio` 优先搜索 `0.15/0.25/0.35/0.50`。
- router threshold、halo radius、ECE/Brier calibration。
- maximum hazard、event threshold、service/settled mask 和 token budget。
- 在同等 active token 预算下比较 TC、LPIPS 和延迟，避免只靠增加计算量提点。

### 9.3 Residual 生成与闭环状态

- LiteCorrector 的频率范围和最大修正幅度。
- SparseResidualDiT hidden size、layers、heads、patch size 和 cross-attention。
- residual diffusion steps、噪声 schedule、AdaLN-Zero 初始化。
- short state、anchor EMA、hazard 和 scene reset。
- teacher forcing 衰减、closed-loop 比例、长视频 drift 和曝光偏差。
- 最后才扩大网络规模；优先修复机制和目标错配。

### 9.4 Loss 和 Metric Alignment

当前候选 loss 包括：

```text
diffusion, residual, transport, flow, visibility,
router, survival, lite, lpips, tc, warp, text,
budget, drift, crcd
```

可采用：

- EMA-normalized 多目标 loss。
- TC/LPIPS 在后期 curriculum 中逐步加权。
- 主干和 metric-alignment 参数组使用不同 LR。
- 动态 loss weighting、GradNorm 或 target-aware schedule。
- 针对静态化增加 motion-energy/flow-consistency guard。
- 针对闭环漂移增加 multi-horizon rollout loss。

训练 loss 权重不等于 checkpoint 选择权重。所有调整只用 validation 判断。

### 9.5 优化与训练策略

- AdamW、LR、warmup、weight decay、gradient clipping。
- cosine/OneCycle 等 scheduler，但恢复必须精确还原状态。
- bf16/fp16 AMP、gradient accumulation、gradient checkpointing、compile。
- temporal 全参训练与模块选择性训练。
- EMA decay、EMA/non-EMA validation 对照。
- 默认使用单个固定 seed 快速迭代；额外 seed 仅为可选分析，不影响达标。
- 超参随机搜索或 Optuna 只能读 validation，不能读 test。

### 9.6 CRCD 蒸馏与实时化

正确顺序是：

1. 先得到稳定的 4-step teacher。
2. 冻结 teacher，蒸馏 2-step student。
3. 在相同 active token budget 下比较质量和延迟。
4. 2-step 稳定后再尝试 1-step。

蒸馏必须说明 target 是否 stop-gradient、student 是否只预测 active normal residual、operator gap、closed-loop gap、drift slope 和 long-video error。teacher 尚未对 validation TC/LPIPS 产生有效改善时不得提前蒸馏。

### 9.7 论文与仓库检索

遇到机制瓶颈时，检索与 prompt-to-video、video diffusion、temporal consistency、latent transport、sparse diffusion、consistency distillation 和 real-time generation 相关的顶会/顶刊论文及可信代码。

保存到 `references/literature_review.md`：标题、年份、venue、URL、仓库 commit/tag、许可证、可复用机制、与 TARDIS 的差异和采用/不采用理由。禁止只罗列论文名或伪造无法核验的信息。

---

## 10. 单数据集分 Stage 调优状态机

三个数据集严格按 queue 执行：

`dataverse -> openvid -> seedance`

前一个数据集完成内部 SOTA 判定并冻结唯一权重后，才进入下一个。三个数据集之间禁止 checkpoint resume。

### 当前执行锚点（2026-08-14）

不得从 Stage 0 重启 DataVerse。当前活动候选是
`dataverse-stage12-validation-full-trajectory-perceptual-p1-seed3407`，run id
`20260814_052612_428443`，tmux 会话 `tardis_dataverse_s12_validation_p1`。它在
`2026-08-14 13:25 UTC`（tmux 本地显示 `21:25`）使用同一 run、同一训练签名和同一原子
`latest.pt` 完成恢复，未使用新的 warm start。它最初从以下
DataVerse 受保护权重的 EMA 做 weights-only warm start：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260813_165601_304501/best.pt
SHA-256 3dfa630ccd4b2ee3b73bfb21af2b417b43d99b6db1a8ad8a08b39174d0d86772
validation TC     0.03904570764892057
validation LPIPS  0.6753896184600308
validation score  0.8288446323955958
```

当前候选采用完整 `512x512x16` 因果 rollout、`7232` 条训练记录、`256` 条完整 validation、
两步 endpoint sampler、`full_temporal`、`metric_alignment`、`micro_batch=2`、accumulation `4`。
它在外部中断前推进到 `micro_step=2251`，当前最新原子恢复点为：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260814_052612_428443/latest.pt
micro_step=2048, optimizer_step=512, next_batch_index=2048
SHA-256 87c3d8c8d9bb0647eb1baacb88141a95fd64af4852e451071df4d6d3a6d1ecb1
```

上述 checkpoint 是本次恢复原点，后续会被新的原子 `latest.pt` 覆盖。恢复实现已把数据游标
直接定位到 `next_batch_index`；第一条新 micro-batch 事件为 `batch_index=2048`，没有重新读取
或解码已完成 batch。稳定训练显存约 `25.2/32.8 GiB`，GPU 利用率采样为 `100%`。
在第一次完整 validation 前不得判断改善或启动 test。恢复后由同 run 的自适应守护在同一
epoch 的 checkpoint 持久化后执行裁决：

```text
target_pass=true                         -> 停止冗余轮次并冻结候选
protected_score - candidate_score < 0.002 -> 平台期，停止并回滚
否则                                      -> 允许下一完整轮次
```

当前候选的 `epochs=2` 是本 stage 的最大预算，不是必须机械跑满的轮数。若 Stage 12 首个完整
validation 未产生实质改善，下一候选只允许改变一个机制变量：将训练 objective 中参与
LPIPS/TC 的 decoded video 与部署生成路径统一为 `clamp(-1, 1)`，先补测试再修改实现。DataVerse
未达到双目标前，不得正式调优 OpenVid 或 Seedance，也不得运行新的 test Infer。

#### 2026-08-15 活动执行覆盖：Stage 13

上面的 Stage 12 内容保留为历史审计，不再是当前运行锚点。当前唯一活动候选是
`dataverse-stage13-deployment-range-alignment-p1-seed3407`，run id
`20260814_184042_288434`，训练会话 `tardis_dataverse_s13_clamp_p1`，第 1 轮 watcher
为 `tardis_dataverse_s13_target_watch`，第 2 轮 watcher 为
`tardis_dataverse_s13_epoch2_watch`。

该候选从受保护 Stage 11 EMA 权重启动：

```text
/root/autodl-tmp/TARDIS/TARDIS_SOTA/work/checkpoints/dataverse/
20260813_165601_304501/best.pt
SHA-256 3dfa630ccd4b2ee3b73bfb21af2b417b43d99b6db1a8ad8a08b39174d0d86772
```

Stage 13 唯一机制变量是将 LPIPS/TC 的 decoded video 在训练 objective 中统一裁剪到
`[-1,1]`，与部署生成路径一致；模型、数据、split、seed、sampler、loss 权重、full-temporal
rollout、optimizer 和有效 batch 均保持不变。正式预算为每 epoch 完整 `7232` 条 train records，
随后完整 `256` 条 validation records，`512x512x16`、两步 endpoint sampler、`micro_batch=2`、
accumulation `4`、`bf16`。

截至当前运行快照，Stage 13 第 1 个 epoch 为 `micro_step=2305/3616`，validation 尚未开始，
`adaptive_decision.json` 仍为 `pending`。资源采样为 `27196/32760 MiB`（约 `83.0%`）和
GPU 利用率 `100%`。在该候选完成完整 validation 并按 target-first 与 `0.002` 实质改善门槛
裁决前，不得读取 test、修改受保护权重或启动 OpenVid。

#### 2026-08-15 活动执行覆盖：Stage 14 快速候选

Stage 13 因 `full_temporal` 训练约 `11` 小时/epoch 且尚未进入 validation，已在保留其
checkpoint、manifest 和 events 后停止，状态为 `interrupted_before_validation`，不参与选择。
当前活动候选切换为
`dataverse-stage14-keyframe-deployment-range-alignment-p1-seed3407`，run id
`20260815_021511_391197`，训练会话 `tardis_dataverse_s14_keyframe_p1`。

Stage 14 从受保护 Stage 11 EMA 权重启动，保持完整 `7232` 条 train records/epoch 和完整
`256` 条 validation records/epoch；训练阶段使用 `keyframe_only`（每条记录 1 个训练帧），
`steps_per_epoch=904`、`micro_batch=8`、`gradient_accumulation=1`，validation 仍使用完整
16 帧。该加速只改变训练计算路径，不改变 validation 记录覆盖、指标实现或 validation-only
checkpoint 选择协议。第 1、2 轮 watcher 为 `tardis_dataverse_s14_target_watch` 和
`tardis_dataverse_s14_epoch2_watch`；在完整 validation 达标前不得读取 test 或切换数据集。

Stage 14 第 1 个完整 validation 已得到 `TC=0.0390950483`、`LPIPS=0.6712395780`、
`score=0.8267648230`，相对受保护分改善 `0.0020798094`，因此按注册规则进入第 2 个 epoch。
TC 已达标但 LPIPS 仍未达 `0.60`；当前继续 validation-only 调优，不执行 test。

### Stage 0：协议锁定与 initial incumbent

- 冻结 manifest、split、metric、512x512、帧数、FPS、seed 和采样配置。
- 跑 metric sanity、泄漏审计和机制诊断。
- 使用默认单卡配置从头训练 initial incumbent。
- 正式 initial incumbent 的每个 epoch 必须完整覆盖 `7232` 条训练记录并完整验证 `256` 条；不再使用 `64` 个 micro-batch 作为正式 epoch。诊断性短跑只能标记为 `diagnostic_only=true`，不能生成或替换正式 `best.pt`。
- 用 validation 选择 initial best，不运行正式 test。

### Stage 1：Transport 和可见性修复

- 从该数据集 initial best 热启动。
- 根据 TAR energy、flow/visibility 和 closed-loop gap 修复 motion/transport。
- 可先选择性训练 motion/transport，再联合解冻 temporal 参数。
- 正式候选至少完成一个完整 train epoch 和一个完整 validation。其后是否继续由 target 状态、相对受保护权重的综合分改善、指标斜率和机制诊断决定；诊断性短跑只用于排除 OOM、NaN、无梯度或明显实现错误。

### Stage 2：TOQ、Router 和事件预算

- 从 Stage 1 当前 best 热启动。
- 搜索 quotient、active ratio、hazard、halo 和 router calibration。
- 同时记录质量、active token ratio 和 per-frame latency。
- 固定预算下只比较 validation TC/LPIPS 及其加权分；该阶段不增加额外达标门槛。

### Stage 3：Residual、闭环与 metric alignment

- 从 Stage 2 当前 best 热启动。
- 调整 residual capacity、diffusion steps、teacher forcing、closed-loop 和 TC/LPIPS loss。
- 重点监控长时 drift、static collapse 和 CLIP/SSIM 退化。
- 每次只原子替换完整 validation 确认更优的 best。

### Stage 4：Validation 复核与候选冻结

- 子集只用于快速淘汰，不能用于达标判定。
- 幸存候选必须完整遍历固定的 256 条 validation。
- 只用完整 validation 的 TC/LPIPS `target_pass` 与固定加权分选择唯一 `best.pt`。
- 冻结候选权重、配置、EMA 选择和 SHA-256；不读取 test 指标参与任何选择。

### Stage 5：可选蒸馏和资源标定

- 可在未达标阶段用于提速或继续优化，也可在达标后仅作资源分析。
- 依次验证 4-step、2-step、1-step，质量和资源分别报告。
- 在目标 GPU 上测显存、GPU util、功耗和 p50/p95 latency。
- 默认目标显存为总显存 60%-85%；不能分配无用张量伪造占用。
- 延迟和资源不改变 validation 的 TC/LPIPS `target_pass` 或排序；但 Train/Infer 若不在当前
  `60%-85%` 显存目标内，就不能作为默认交付运行时通过，必须先调整真实计算配置并重新实测。

### Stage 6：最终统一 Test 报告与泛化核验

- 三个数据集都完成 validation target 判定后，冻结三份 checkpoint/config/EMA 选择。
- 写入三份 checkpoint SHA-256，关闭所有搜索代码路径。
- 最后统一启动三个独立 Infer 进程，分别完整遍历 512 条 test records。
- 保存六指标、per-video details、失败清单、资源数据和随机 5 条 showcase。
- test 只用于最终泛化核验和报告，不覆盖 validation `target_pass`，也不得回流调优。
- 若发现协议错误，本次正式结果作废，修复后必须生成新的预注册 split/实验版本并如实记录，不能悄悄覆盖。

热启动阶段不设机械默认轮数。每个候选先完成一个完整 epoch 和完整 validation；若达到双目标则立即冻结，若综合分没有达到预注册的实质改善下限则停止并回滚，只有改善具有继续训练价值时才追加下一轮。每个 stage 结束必须读取正式指标和机制诊断后再决定下一步。

---

## 11. Checkpoint 选择、热启动与替换

### 11.1 validation-only selector

selector 只能接收一个 `*_validation` source，遇到 test 或多源输入必须报错。候选比较顺序：

1. 两项均达标的候选无条件优先于未达标候选。
2. 若候选达标状态相同，只比较固定尺度 TC/LPIPS weighted score，越低越好。

FVD、FID、CLIPScore、SSIM、Pareto 关系、closed-loop drift、static guard、资源和稳定性可以记录用于诊断，但均不得阻止 validation 达标候选更新 `best.pt`，也不得参与 `target_pass` 判定。
资源区间另属于运行时验收门槛，不得用它改变候选排序，也不得在未达标时宣称默认部署配置已通过。

### 11.2 同数据集链式热启动

- 首次从该数据集 initial incumbent 开始。
- 后续 stage 只从本轮该数据集上一版受保护 best 热启动。
- 禁止每一轮回到 initial checkpoint。
- 禁止加载另一个数据集的 temporal 权重。
- 恢复必须同时还原 optimizer、scheduler、AMP scaler、EMA、loss normalizer、curriculum、distiller 和 RNG 状态；仅做权重微调时必须明确标记 `weights_only_warm_start=true`。

### 11.3 原子 best 替换

每个数据集任何时刻保留：

- 一个受保护 current best；
- 一个 latest 恢复点；
- 当前临时候选。

候选必须完成完整 validation、checkpoint load test 和 SHA-256 后，才能原子更新 current-best 指针。旧受保护 best 在新权重通过严格加载、哈希和恢复审计前必须字节级保留；失败候选的日志、manifest、指标与裁决必须保留，checkpoint payload 只有在明确归档且不再承担回滚作用时才可清理。最终交付目录每个数据集只暴露一份权重，但工作审计区可以保留必要回滚点。

### 11.4 Temporal checkpoint schema

checkpoint 只保存 TARDIS temporal state，不包含 `priors.*`。最终 manifest 必须记录 frozen prior 的模型 ID、revision、配置和文件哈希，保证另一台机器能获得同一先验。

三份权重必须对同一交付代码 schema 执行 strict 100% temporal load。若三数据集使用不同结构配置，必须通过 manifest 显式构造后再 strict 加载，不能忽略 missing/unexpected keys。

---

## 12. 资源、DataLoader 与进程监控

- 训练和正式 Infer 使用本地 manifest、本地 TAR/ZIP 有界读取和 DataLoader。
- `TARDIS_NUM_WORKERS` 初始建议 8，但必须在真实机器验证 spawn-worker、共享内存、解码吞吐和稳定性；发生 137/OOM 时根据实测调整并记录。
- micro batch、gradient accumulation、prefetch、persistent workers 和 pin memory 通过短跑标定。
- 显存目标 60%-85%，同时记录 allocated/reserved/peak，不能只看 `nvidia-smi` 单点。
- 正式 stage 使用 tmux，保存 session、PID、GPU、dataset、stage、epoch、ETA 和日志路径。
- `work/pipeline_state.json` 实时记录当前数据集、run_id、parent/best SHA-256、stage、epoch、seed、验证指标、资源、下一步动作和更新时间。
- 进程退出、卡死、NaN、OOM、DataLoader 137 或 GPU 长时间空闲时，Agent 必须主动诊断、恢复或换参，不能只报告进程消失。
- 每个 epoch 在终端只保留一个完整 train 进度条和一个完整 validation 进度条；validation 完成后立即输出当前数据集的 TC、LPIPS、FVD、FID、CLIPScore、SSIM、weighted score 和 target-pass 状态。micro-batch loss 只能作为健康诊断，不能写入正式指标表。

默认一个 Train/Infer 进程只占一个数据集。若硬件允许多 GPU 并行，仍必须保持三数据集 namespace、日志、采样器和 GPU 互不相交；本 Pipeline 默认串行以确保一个数据集收敛后再处理下一个。

---

## 13. 正式 Infer 与报告口径

每个数据集必须独立运行：

```bash
TARDIS_DATASET=dataverse \
TARDIS_CHECKPOINT=<dataverse_best.pt> \
bash scripts/infer.sh

TARDIS_DATASET=openvid \
TARDIS_CHECKPOINT=<openvid_best.pt> \
bash scripts/infer.sh

TARDIS_DATASET=seedance \
TARDIS_CHECKPOINT=<seedance_best.pt> \
bash scripts/infer.sh
```

每个 Infer 必须遍历完整 512 条 test records，并输出：

- `metrics.csv`、`metrics.xlsx`。
- `per_video_details.csv`、`per_video_details.jsonl`。
- `completed.jsonl`、`failures.jsonl`。
- `latency.json`、`resources.json`。
- `manifest.json`、`result_manifest.json`。
- 固定 seed 从当前数据集 test prompts 中随机选取 `5-6` 条 showcase，默认保存 `5` 条 `showcases/*.mp4`；六指标仍必须使用完整 `512` 条 test records 计算。

三个独立结果还要离线写入同一个总表：

`reports/三个数据集正式测试指标汇总.xlsx`

同一 sheet 以数据集为行，至少包含：

```text
Dataset, TC, LPIPS, FVD, FID, CLIPScore, SSIM,
validation_score, validation_sota, target_pass,
checkpoint_sha256, sample_count, failed_count,
p50_ms_per_frame, p95_ms_per_frame, peak_vram_gb
```

允许提供三者平均作为展示行，但必须明确 `display_only=true`，不能参与权重选择或掩盖单数据集失败。

---

## 14. Apply 和可用性验证

每份交付权重至少运行一次 prompt-only Apply 可用性验收：

```bash
TARDIS_DATASET=<dataset> \
TARDIS_CHECKPOINT=<dataset_best.pt> \
TARDIS_PROMPT="A robot running in the forest" \
TARDIS_STYLE="cinematic, highly detailed" \
TARDIS_DURATION=2 \
bash scripts/apply.sh
```

验证：

- 不读取 source video/test label。
- 输出可解码 MP4。
- 分辨率、帧数、FPS 和 duration 正确。
- sidecar 记录 prompt、style、seed、采样参数、checkpoint SHA-256 和延迟。
- 三数据集权重不会发生 namespace 串用。

Apply 验收视频只证明接口可用并用于展示，不参与 SOTA 选择。

---

## 15. 唯一交付包结构

最终 `<TARDIS_ROOT>/TARDIS_SOTA/` 至少包含：

```text
TARDIS_SOTA/
├── weights/
│   ├── dataverse_best.pt
│   ├── openvid_best.pt
│   └── seedance_best.pt
├── code/
│   ├── tardis/
│   └── 与权重对应的必要项目代码
├── scripts/
│   ├── train.sh
│   ├── infer.sh
│   ├── apply.sh
│   ├── run_sota_queue.sh
│   ├── verify_delivery.py
│   └── 必要 wrapper
├── configs/
│   ├── dataverse.json
│   ├── openvid.json
│   ├── seedance.json
│   ├── selection_scale_lock.json
│   ├── split_manifest_lock.json
│   └── delivery_manifest.json
├── reports/
│   ├── 三个数据集正式测试指标汇总.xlsx
│   ├── validation_pareto与seed复核.xlsx
│   ├── sota_decision.md
│   ├── source_audit.md
│   ├── leakage_audit.json
│   ├── metric_provenance.json
│   ├── resources_and_latency.xlsx
│   └── delivery_verification.json
├── infer_outputs/
│   ├── dataverse/
│   ├── openvid/
│   └── seedance/
├── apply_outputs/
│   ├── dataverse/
│   ├── openvid/
│   └── seedance/
├── references/
│   ├── handoff.md
│   ├── literature_review.md
│   └── prior_provenance.json
├── README.md
├── MODEL_CARD.md
├── requirements.txt
└── SHA256SUMS
```

不包含：训练数据、完整 SD-Turbo prior 文件、失败 checkpoint、latest optimizer 快照、大量可再生缓存、外部 baseline 代码或不存在的 baseline 指标表。

README 必须让没有聊天历史的 Agent 可以完成：环境准备、prior 准备、数据路径检查、单数据集 Train、resume、指定权重 Infer、Apply、三数据集 queue、正式评测和交付验证。

MODEL_CARD 必须说明：TARDIS 机制、prompt-only 边界、三个数据集、split、六指标、SOTA 自决定义、三份权重、训练 seed、参数量、资源、实时状态、已知限制和“内部 protocol-best 而非公开排行榜 SOTA”的结论边界。

---

## 16. Manifest、strict load 和 SHA-256

`configs/delivery_manifest.json` 至少记录：

- TARDIS 代码版本和关键文件 SHA-256。
- 模型构造参数、AblationVariant、temporal state schema。
- frozen prior 模型 ID、revision 和哈希。
- 三个数据集 manifest/split 哈希和真实样本数。
- 512x512、FPS、帧数、duration、seed 和采样配置。
- 六指标实现、依赖版本、输入范围和 aggregation。
- 三份权重路径、SHA-256、tensor 数、参数量、EMA 选择和正式 test 指标。
- 每份权重的训练 seed、parent 链、validation TC/LPIPS、锁定阈值和 validation_sota 判定。
- Python、PyTorch、CUDA、GPU、ffmpeg 和关键依赖。

交付验证必须在干净进程中对三份权重逐一执行：

1. 按 manifest 构造完全相同的 TARDIS temporal 模型。
2. temporal checkpoint strict 100% key/shape 加载。
3. 确认不含 `priors.*`，并加载指定 frozen prior。
4. 512x512 prompt-only `generate()` 可用性验证。
5. 固定输入和 seed 的确定性/允许误差测试。
6. 小型 validation 子集六指标复算，与归档结果核对。
7. Apply MP4 编解码和 sidecar 检查。
8. Excel、JSON、manifest、代码和 checkpoint 哈希交叉校验。

最终生成覆盖权重、代码、脚本、配置、报告和说明的 `SHA256SUMS`。只有三份权重全部通过时，`delivery_verification.json` 才能写：

```json
{"sota_acceptance_all": true, "strict_temporal_load_all": true}
```

---

## 17. 失败恢复和停止规则

- OOM：停止当前进程，减 micro batch、启用 gradient checkpointing 或增加 accumulation；不得跳过样本。
- DataLoader 137：检查 worker、共享内存、归档读取和预取，降低 worker/prefetch 后复测。
- NaN/Inf：保存首个异常 batch 元信息，检查 AMP、loss normalizer、LR、flow 和 latent 范围。
- 训练过慢：profile 解码、I/O、attention、diffusion steps、同步和 validation，不得通过减少正式数据伪造速度。
- TC 降而 LPIPS 升：继续以“两项同时达标”为唯一目标调整训练。
- LPIPS 降而 TC 升：继续以“两项同时达标”为唯一目标调整训练。
- quotient energy 不降：修正 basis/regularization/rank，不先扩大 DiT。
- router calibration 差：修正 oracle-to-predicted curriculum、ECE/Brier 和 budget loss。
- 候选回退：保留受保护 current best；归档失败候选的事件、manifest、指标和裁决后再决定是否清理其可再生 checkpoint payload。
- 进程中断：只从同数据集、同 run_id 的完整 latest 状态恢复。
- test 协议错误：结果作废并记录；禁止把已看到的 test 指标用于下一轮搜索。
- 只有完整 validation 的 TC 和 LPIPS 同时达到锁定阈值，才能标记 validation protocol-best；
  最终 test 结果单独报告。

资源或时间预算耗尽不等于达标。任一 validation 目标未满足时必须如实标记
`target_pass=false`；其它指标和审计结果不得覆盖这两个阈值判断。

---

## 18. 中间产物和清理规则

1. 所有本轮中间产物仅位于 `TARDIS_SOTA/work/`。
2. 每个数据集只保留受保护 current best、latest 和当前临时候选。
3. 新 best 通过完整 validation、strict load 和哈希检查后原子替换旧 best。
4. 数据集完成后删除其失败 checkpoint、optimizer 临时快照和可再生 validation 视频，只保留搜索摘要与必要日志。
5. 三数据集完成后删除 `work/` 中不属于交付的内容。
6. 不得删除原仓库、数据盘 manifest、用户文件、现有测试或其他交付包。
7. 最终根目录只新增一个 `TARDIS_SOTA/` 业务包，不留下零散实验目录。

---

## 19. Agent 必须执行的顺序

1. 读取 `handoff.md`、TARDIS 文档、代码和测试。
2. 创建唯一 `TARDIS_SOTA/` 目录及内部 `work/`。
3. 记录原代码、入口、环境、prior 和数据 manifest 哈希。
4. 运行现有测试与静态检查，完成 source audit。
5. 固定三个数据集的 split，并完成 id/media/caption 泄漏审计。
6. 锁定 512x512、帧数、FPS、六指标和 selection scales。
7. 建立 metric sanity controls 和 TARDIS 机制诊断。
8. 按 `dataverse -> openvid -> seedance` queue 处理。
9. 对当前数据集从头训练 initial incumbent，不跨数据集加载 temporal 权重。
10. 执行候选调优，只使用 validation 的 TC 和 LPIPS 决策。
11. 只用完整 validation 的 `target_pass` 与固定加权分选出并冻结三份权重和 SHA-256。
12. 三个数据集全部锁定后，统一执行最终 test Infer 并记录六指标。
13. 三个数据集统一验收并清理交付产物。
14. 三数据集完成后生成统一六指标 Excel 和 SOTA decision。
15. 为三份权重运行 prompt-only Apply 可用性验收。
16. 整理代码、脚本、配置、报告、README 和 MODEL_CARD。
17. 在干净进程执行 strict temporal load 和 delivery verification。
18. 清理中间文件，生成最终 `SHA256SUMS`。
19. 最终只向用户返回交付包路径、三份权重、统一指标表和验收摘要。

---

## 20. 最终验收清单

- [ ] 没有创新表依赖，使用的是现有 TARDIS 框架和真实代码。
- [ ] 没有外部 baseline 表，也没有伪造公开 SOTA 对比。
- [ ] SOTA 明确限定为固定协议下的内部 protocol-best。
- [ ] dataverse、openvid、seedance 的 train/validation/test 完全隔离。
- [ ] `split_seed=3407`、validation 256、test 512 和 manifest hash 已锁定。
- [ ] 三个数据集均使用 512x512、同一六指标实现和一致采样口径。
- [ ] checkpoint 只由 validation 选择，test 未参与调参。
- [ ] 三个数据集 validation 均达到统一的 LPIPS `<=0.60` 与各自 TC 阈值。
- [ ] 三个数据集各有一份唯一 temporal 权重，namespace 不混用。
- [ ] 三份权重均对交付代码 strict 100% temporal load。
- [ ] 最终统一 Infer 的三个完整 test 结果位于同一 Excel，同一 sheet。
- [ ] FVD、FID、CLIPScore、SSIM 和 per-video details 已完整记录。
- [ ] 30 FPS、33.3 ms、显存和 GPU 资源均有真实测量或明确未通过状态。
- [ ] Train、Infer、Apply 仍通过正式 torchrun Shell 入口运行。
- [ ] 三份权重的 prompt-only Apply 可用性验收通过。
- [ ] README、MODEL_CARD、manifest、verification 和 SHA256SUMS 完整。
- [ ] 最终包不依赖聊天历史，可由下一个 Agent 开箱复用。
- [ ] 项目根目录没有散落本轮 stage、checkpoint、output 和日志。

任一必需项未通过，都不能声称三个数据集已经完成 SOTA 交付。
