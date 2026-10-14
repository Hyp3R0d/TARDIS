# TARDIS 项目说明

本文介绍项目用途、目录分工和运行入口，适合首次阅读仓库的开发者。详细实验参数请查阅各专项文档。

## 项目做什么

TARDIS（Transport-Aligned Residual Diffusion in Innovation Subspaces）研究连续视频生成：先把历史帧中可复用的内容对齐到当前帧，再对需要更新的部分进行残差生成。目标是减少重复计算，并改善视频的时序一致性。

核心实现位于 [tardis/models/tardis.py](../tardis/models/tardis.py)。帧间处理包含运动估计、内容传输、残差分解、风险路由、轻量校正、稀疏残差生成和因果状态更新。实际启用的模块取决于模型配置与消融选项。

[Python 包配置](../pyproject.toml)将项目标记为 Alpha，包名为 `tardis-video`。运行性能与生成质量需要结合具体权重、数据、硬件和配置评估。

## 先选择运行目标

| 目标 | 入口 | 需要准备 |
| --- | --- | --- |
| 训练 TARDIS 模型 | [scripts/train.sh](../scripts/train.sh) | Python/CUDA 环境、基础模型、数据集和路径配置 |
| 评测 TARDIS checkpoint | [scripts/infer.sh](../scripts/infer.sh) | 对应数据集、checkpoint 和匹配的模型配置 |
| 根据提示词生成视频 | [scripts/apply.sh](../scripts/apply.sh) | TARDIS checkpoint、基础模型和提示词 |
| 运行桌面或网页创作界面 | [tardis-client/](../tardis-client/) | Node.js/npm、智谱 API 凭据 |
| 开发 HarmonyOS 界面 | [UI/](../UI/) | HarmonyOS 开发环境和可访问的配套服务 |
| 查看 Java 服务工程 | [web-server/](../web-server/) | Java/Maven 环境，配置见该目录 |

**Python 模型与当前桌面、网页客户端使用不同的生成入口。** 客户端的 [Electron 实现](../tardis-client/electron/api.mjs)和 [Express 实现](../tardis-client/server/index.mjs)调用智谱接口，模型为 `cogvideox-3`，凭据变量为 `ZHIPU_API_KEY`。启动客户端并不等于运行了本仓库的 TARDIS 模型。

## Python 运行入门

依赖要求为 Python `>=3.12`、PyTorch `>=2.8,<2.9`。模型运行建议准备 Linux 和 NVIDIA CUDA GPU；完整依赖以 [pyproject.toml](../pyproject.toml) 为准。

以下 Bash 命令在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

将以下占位路径换为实际路径，并先按照[数据集说明](datasets.md)准备数据。脚本支持 `dataverse`、`openvid`、`seedance` 数据集标识，单次选择一个。

```bash
export TARDIS_STORAGE_ROOT=/path/to/tardis-storage
export TARDIS_DATASETS_FILE=/path/to/datasets.local.txt

# 训练
TARDIS_DATASET=dataverse bash scripts/train.sh

# 评测：使用与数据集和模型配置匹配的权重
TARDIS_DATASET=dataverse TARDIS_CHECKPOINT=/path/to/best.pt \
  bash scripts/infer.sh

# 提示词生成：无需提供 source video
TARDIS_DATASET=dataverse TARDIS_CHECKPOINT=/path/to/best.pt \
  TARDIS_PROMPT="A robot walking through a bamboo forest" \
  TARDIS_DURATION=2 bash scripts/apply.sh
```

脚本默认存储根目录为 `/root/autodl-tmp/TARDIS`；通过 `TARDIS_STORAGE_ROOT` 指定本机目录。克隆源码不会自动获得训练数据和可用的 TARDIS checkpoint。

训练、评测和生成的配置及输出分别见 [train.md](train.md)、[infer.md](infer.md) 和 [apply.md](apply.md)。生成入口输出 MP4 视频及对应 JSON 元数据。

## 桌面与网页客户端

在仓库根目录执行：

```bash
cd tardis-client
npm install
npm run desktop:dev
```

桌面生成可通过设置界面配置 API Key，或在启动前设置 `ZHIPU_API_KEY`。网页开发模式使用同目录下的 `npm run dev`，同时启动 Vite 界面和 Express 代理；代理默认端口为 `8787`，从环境变量读取 `ZHIPU_API_KEY`。不要把真实密钥提交到仓库。

完整命令见 [package.json](../tardis-client/package.json)。若要让界面调用 TARDIS 自有模型，需要实现或适配相应服务接口。

## 验证与继续阅读

本说明依据源码和配置整理；文档检查不等同于完成 GPU 训练、视频生成或客户端联调。

- [README](../README.md)：项目介绍与效果展示。
- [docs/](./)：数据准备、训练、评测和生成说明。
- [tests/](../tests/)：Python 测试；开发依赖安装命令为 `python -m pip install -e '.[dev]'`。
- [CI 配置](../.github/workflows/ci.yml)：Ruff、格式检查、mypy，以及排除 `gpu`、`remote`、`slow` 标记的 pytest 检查。
- 客户端测试：在 `tardis-client/` 中执行 `npm test`。
- [LICENSE](../LICENSE)：项目许可证；第三方模型、数据与服务另遵循各自条款。
