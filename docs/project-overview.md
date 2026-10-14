# TARDIS 项目说明

本文面向首次了解或准备运行项目的读者，介绍项目用途、核心方法、工程组成和运行入口。说明以仓库当前代码为依据，详细参数见各专项文档。

## 项目定位

TARDIS 全称为 **Transport-Aligned Residual Diffusion in Innovation Subspaces**，是面向连续视频生成与时序一致性研究的框架。它尝试复用相邻帧中可以预测的内容，把生成计算集中在发生变化的区域，减少逐帧独立生成带来的重复计算和画面不稳定。

仓库包含 Python 模型与实验代码、Electron 桌面客户端、HarmonyOS 客户端以及 Java 运维服务。Python 包名为 `tardis-video`，当前版本为 `0.1.0`，在 [pyproject.toml](../pyproject.toml) 中标记为 Alpha。阅读展示结果时，应将设计目标与经过统一协议验证的性能结论区分开。

## 核心方法

可以将 TARDIS 理解为“先沿用上一帧，再生成需要变化的部分”：

1. **建立首帧和语义条件**：使用冻结的文本编码器、VAE 和图像先验，为后续时序生成提供基础。
2. **估计运动并传输历史内容**：根据运动与可见性，将上一帧 latent 和历史状态对齐到当前帧。
3. **分解残差**：通过传输轨道投影，将变化分成可由运动附近的校正解释的部分，以及需要生成的创新部分。
4. **分配生成预算**：风险路由器与事件时间模块选择需要更新的区域和 token。
5. **校正并生成当前帧**：轻量校正器处理切向残差，稀疏残差扩散处理法向创新，再组合为当前结果。
6. **更新因果状态**：保存当前状态、锚点和历史信息，继续生成后续帧。

```text
文本提示 → 首帧与冻结先验
                 ↓
历史状态 → 运动估计与内容传输 → 残差分解与风险路由
                                      ↓
                         轻量校正 + 稀疏残差扩散
                                      ↓
                              当前帧 → 状态更新
```

主流程位于 [tardis/models/tardis.py](../tardis/models/tardis.py)。相关实现包括 `priors.py`、`motion.py`、`transport.py`、`quotient.py`、`router.py`、`clock.py`、`residual.py` 和 `state.py`，均位于 `tardis/models/`。

## 工程组成与实际调用关系

| 组件 | 位置 | 主要职责 |
| --- | --- | --- |
| Python 研究引擎 | [tardis/](../tardis/) | 模型、数据处理、训练、推理、指标和实验 |
| 标准运行脚本 | [scripts/](../scripts/) | `train.sh` 训练、`infer.sh` 评测、`apply.sh` 生成 |
| 桌面与网页客户端 | [tardis-client/](../tardis-client/) | 提示词、参考图、异步任务、视频播放与桌面归档 |
| HarmonyOS 客户端 | [UI/](../UI/) | ArkUI 创作界面，通过 HTTP 提交与查询任务 |
| Java 运维服务 | [web-server/](../web-server/) | 静态页面、健康检查、服务信息、指标与管理接口 |
| 测试与文档 | [tests/](../tests/)、[docs/](./) | 自动化检查、专项说明与展示材料 |

当前有两类独立的运行流程：

```text
自有模型研究：数据与模型权重 → Python CLI → checkpoint / 指标 / MP4

桌面生成：React 界面 → Electron IPC → electron/api.mjs → 智谱视频接口
网页生成：React 界面 → Express /api/generations → 智谱视频接口
HarmonyOS：ArkUI 界面 → 配置的 HTTP 服务 /api/generations
```

**当前客户端的实际后端需要特别区分。** [electron/api.mjs](../tardis-client/electron/api.mjs) 和 [server/index.mjs](../tardis-client/server/index.mjs) 均调用 `https://open.bigmodel.cn/api/paas/v4`，提交的模型为 `cogvideox-3`，读取 `ZHIPU_API_KEY`。桌面模式通过 IPC 调用主进程；网页开发模式才使用 Express 代理。入口选择见 [src/main.jsx](../tardis-client/src/main.jsx)。

现有 [API_CONTRACT.md](../tardis-client/server/API_CONTRACT.md) 描述的是 TARDIS 服务接口及 `TARDIS_INFERENCE_API_KEY`，与上述当前实现存在差异。部署时应以实际代码为准；若要让客户端调用本仓库的 TARDIS 模型，需要补齐 HTTP 服务与客户端适配，不能仅替换密钥变量就认为已经接通。

`web-server/` 提供 `/api/health`、`/api/info`、`/api/metrics` 等运维接口，并不实现上述视频生成 API，也不运行 Python GPU 模型。HarmonyOS 的 HTTP 调用见 [ApiClient.ets](../UI/entry/src/main/ets/service/ApiClient.ets)。

## 功能与输入输出

| 场景 | 输入 | 输出或行为 |
| --- | --- | --- |
| 模型训练 | 单个数据集的训练与验证划分、模型先验和训练配置 | `latest.pt`、`best.pt`、训练记录 |
| 推理评测 | 指定数据集的测试划分与对应 checkpoint | TC、LPIPS 等指标、逐视频明细、资源统计和展示视频 |
| Prompt-only 生成 | prompt、style、时长和匹配的 checkpoint | `video.mp4` 与记录参数的 `video.json` |
| 客户端创作 | 提示词、可选参考图和生成设置 | 提交异步任务、轮询、播放结果；桌面端支持本地归档 |

Python 流程使用 `dataverse`、`openvid`、`seedance` 三个数据集标识，单次训练或评测选择一个数据集。`apply` 不接收 source video，数据集标识用于选择对应权重。验证集用于选择 `best.pt`，测试集用于最终独立评测。

客户端当前校验提示词不超过 512 字符，参考图为 PNG/JPEG 且解码后不超过 5 MB。任务状态归一化为 `PROCESSING`、`SUCCESS`、`FAIL`。客户端请求与返回校验见上述 `electron/api.mjs` 和 `server/index.mjs`。

## 运行入口

### Python 模型训练与推理

推荐使用 Linux 与 NVIDIA CUDA GPU。依赖要求为 Python `>=3.12`、PyTorch `>=2.8,<2.9`，完整依赖见 [pyproject.toml](../pyproject.toml)。以下为 Bash 命令，在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

运行前需要准备基础模型、数据或已训练的 TARDIS checkpoint，并将下面的占位路径替换为本机实际路径。仅克隆源码不会自动获得可用的训练数据和项目权重。

```bash
export TARDIS_STORAGE_ROOT=/path/to/tardis-storage
export TARDIS_DATASETS_FILE=/path/to/datasets.local.txt

# 训练：需要已准备的数据集及路径文件
TARDIS_DATASET=dataverse bash scripts/train.sh

# 评测：需要对应数据集与 checkpoint
TARDIS_DATASET=dataverse \
TARDIS_CHECKPOINT=/path/to/best.pt bash scripts/infer.sh

# 生成：使用对应 checkpoint，不读取 source video
TARDIS_DATASET=dataverse TARDIS_CHECKPOINT=/path/to/best.pt \
TARDIS_PROMPT="A robot walking through a bamboo forest" \
TARDIS_DURATION=2 bash scripts/apply.sh
```

默认脚本把缓存、checkpoint 和输出放在 `TARDIS_STORAGE_ROOT` 下，默认值为 `/root/autodl-tmp/TARDIS`。模型结构、精度等参数应与 checkpoint 匹配；不要直接套用其他机器的数据路径。参数及输出布局见 [训练说明](train.md)、[评测说明](infer.md)、[生成说明](apply.md) 和 [数据集说明](datasets.md)。

### 桌面与网页客户端

准备 Node.js/npm，在仓库根目录执行：

```bash
cd tardis-client
npm install
npm run desktop:dev
```

实际生成需要可用的智谱凭据，可通过桌面设置界面配置，或在启动进程前设置 `ZHIPU_API_KEY`。桌面端使用 Electron `safeStorage` 在系统支持时加密保存本机凭据。不要把真实密钥提交到 Git。

同目录运行 `npm run dev` 可启动网页开发模式及 Express 代理，代理默认端口为 `8787`，凭据由服务端环境或本机 `.env.local` 提供。Windows 桌面目录包使用 `npm run desktop:dist` 构建。脚本定义见 [package.json](../tardis-client/package.json)。

### HarmonyOS 与运维服务

使用 DevEco Studio 打开 `UI/` 进行 HarmonyOS 构建，并将客户端服务地址配置为设备可访问、实现 `/api/generations` 的 HTTP 服务。真机上的 `127.0.0.1` 指向真机自身，不是开发电脑。

Java 运维服务要求 JDK 17 和 Maven，在仓库根目录执行：

```bash
cd web-server
mvn package
java -jar target/tardis-webserver.jar
```

默认应用端口为 `8080`；Nginx 和 Docker 配置位于 `web-server/`。具体配置入口为 [application.properties](../web-server/src/main/resources/application.properties)，对外部署前应调整管理凭据与访问范围。

## 验证、结果与文档导航

Python 开发依赖通过 `python -m pip install -e '.[dev]'` 安装。根目录 CI 配置在 [.github/workflows/ci.yml](../.github/workflows/ci.yml)，执行 Ruff、mypy 以及排除 `gpu`、`remote`、`slow` 标记的 pytest 测试；客户端测试入口为在 `tardis-client/` 中执行 `npm test`。

本说明文档依据静态代码与现有说明整理，不代表已经在当前机器完成训练、生成或客户端服务联调。演示图片与 GIF 用于观察视觉效果；速度、显存和画质结论需要在相同硬件、数据划分、权重与采样配置下实测。

- [README](../README.md)：项目展示与综合说明。
- [train.md](train.md)、[infer.md](infer.md)、[apply.md](apply.md)：三个 Python 入口的详细配置。
- [datasets.md](datasets.md)：数据准备、manifest 与划分。
- [实验方案.pdf](../实验方案.pdf)、[技术文档.pdf](../技术文档.pdf)：仓库中的实验和技术材料。
- [LICENSE](../LICENSE)：项目代码采用 Apache License 2.0；第三方模型、数据集与服务遵循各自条款。
