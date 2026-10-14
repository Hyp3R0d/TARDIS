# TARDIS 协作与文档提交指南

本文说明如何关联远程仓库、创建分支、提交文档，并向 `Hyp3R0d/TARDIS` 发起 Pull Request（PR，合并请求）。项目部署和运行方式见 [README](README.md)。

## 1. 获取仓库并检查 remote

首次参与项目时，在准备存放代码的目录执行：

```bash
git clone https://github.com/Hyp3R0d/TARDIS.git
cd TARDIS
git remote -v
```

`clone` 会下载项目并自动将 `origin` 指向 `https://github.com/Hyp3R0d/TARDIS.git`。`remote` 是本地 Git 仓库关联的远程地址，不需要在克隆后重复添加。

如果已经有 TARDIS 的 Git 工作副本，先进入该目录并运行 `git status`、`git remote -v`。仅在缺少 `origin` 时添加：

```bash
git remote add origin https://github.com/Hyp3R0d/TARDIS.git
```

如果 `origin` 已存在，先确认它的用途。只有确实需要改为本仓库时才执行：

```bash
git remote set-url origin https://github.com/Hyp3R0d/TARDIS.git
```

如果本地只是从 ZIP 解压的普通文件夹，建议先克隆仓库，再复制需要修改的文件，以保留项目已有的提交历史。

## 2. 从 main 创建工作分支

先用 `git status` 确认没有未提交的改动；如有改动，先在原分支提交，或使用 `git stash` 临时保存，再进行下面的操作：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c docs/update-guide
```

`docs/update-guide` 是示例分支名，可按本次工作内容替换。`--ff-only` 会在本地与远程历史分叉时停止，此时应先检查差异。若分支已存在，用 `git switch docs/update-guide` 切换即可，不要再次创建。

## 3. 修改并检查文档

根据内容选择对应文件：

| 内容 | 文档入口 |
| --- | --- |
| 项目介绍、组件与部署概览 | [README.md](README.md) |
| 训练、验证和 checkpoint | [docs/train.md](docs/train.md) |
| 全量推理评测与结果输出 | [docs/infer.md](docs/infer.md) |
| Prompt-only 视频生成 | [docs/apply.md](docs/apply.md) |
| 数据集、manifest 和数据划分 | [docs/datasets.md](docs/datasets.md) |
| Git 协作与 PR 流程 | [CONTRIBUTING.md](CONTRIBUTING.md) |

编辑后检查改动内容和空白问题：

```bash
git status
git diff
git diff --check
```

新增且尚未跟踪的文件不会出现在普通 `git diff` 中，需要直接打开检查；暂存后可以通过下一步的 `git diff --cached` 查看。确认相对链接、代码块和命令与实际文件一致；只有实际运行过的验证才写为“已通过”。不要提交 API Key、SSH 私钥、`.env` 文件、数据集或模型权重。

## 4. 提交并推送分支

下面以修改 `CONTRIBUTING.md` 为例；上传其他文档时，将文件名替换为实际路径：

```bash
git add CONTRIBUTING.md
git diff --cached
git diff --cached --check
git commit -m "docs: update contribution guide"
git push -u origin docs/update-guide
```

提交前确认暂存区只包含本次需要的文件。`commit` 将改动记录在本地；`push` 将提交上传到远程分支。首次推送的 `-u` 会建立跟踪关系，后续在同一分支提交后可直接运行 `git push`。

## 5. 发起 Pull Request

1. 打开 [TARDIS 仓库](https://github.com/Hyp3R0d/TARDIS)，进入 **Pull requests → New pull request**，或点击推送后出现的 **Compare & pull request**。
2. 确认目标分支（base）为 `main`，来源分支（compare）为 `docs/update-guide`。
3. 检查提交和文件差异，填写本次解决的问题、具体改动和验证结果。
4. 点击 **Create pull request**，提交给维护者审阅。

PR 创建后仍可在同一分支修改文档、提交并推送；新的提交会自动出现在该 PR 中。发起 PR 不代表已经合并，合并由有权限的维护者在审阅后处理。

仅修改说明文字时，可在 PR 中记录 Markdown 预览、链接检查和 `git diff --check` 的结果。涉及 Python 代码时，仓库当前 CI 还会执行 `ruff check .`、`ruff format --check .`、`mypy tardis` 和 `pytest -m "not gpu and not remote and not slow"`；配置见 [.github/workflows/ci.yml](.github/workflows/ci.yml)。

## 推送权限与账号

直接向本仓库推送分支需要写入权限。接受 Collaborator（协作者）邀请、浏览器登录和本机 Git 认证是不同环节；浏览器登录成功并不表示终端或第三方连接器使用了同一账号。

遇到 `403`、`Permission denied` 或 `Repository not found` 时，检查远程地址、实际认证账号、协作邀请状态以及令牌或连接器对本仓库的授权范围。没有上游写入权限时，可以在 GitHub Fork 仓库，将分支推送到自己的 Fork，再向 `Hyp3R0d/TARDIS` 的 `main` 发起 PR。
