# 本地数据集说明

TARDIS 的训练、验证和测试只读取数据盘本体。项目路径：

```text
/home/TARDIS/data -> /root/autodl-tmp/TARDIS/datasets
```

三源均按同一数据契约整理为 **8,000 个唯一 prompt-video 对**，有效媒体预算为
**44-46 GB/源**。有效媒体体积按正式 manifest 实际引用的视频字节计算，不能通过重复记录、
未引用归档、空文件或重编码膨胀凑数。

## 一键准备

```bash
bash scripts/download_datasets.sh
```

脚本固定 Hugging Face revision，并默认使用 `https://hf-mirror.com`。下载支持断点续传和最多
30 次退避重试。三源严格按 Seedance、DataVerse、OpenVid 的顺序处理，避免 200 GB 数据盘
在候选包与正式分片并存时超过容量。

脚本可重复执行：每个来源先运行正式验收，已经满足条目数、字节预算、固定划分和媒体闭包的
来源会直接跳过。单独验收全部来源：

```bash
PYTHONPATH=. python -m tardis.data.curate_local all --verify-only
```

## 固定目标

| 项目 | 每源目标 |
|---|---:|
| 唯一 prompt-video 对 | 8,000 |
| 有效媒体体积 | 44-46 GB |
| Train | 7,232 |
| Validation | 256 |
| Test | 512 |
| 划分种子 | 3407 |
| 筛选策略版本 | `tardis-balanced-45gb-v1` |

DataVerse 和 OpenVid 使用 `record_identity_v1`：基于 `revision + source + video ID` 的稳定
哈希划分。Seedance 使用 `caption_group_v1`：先对 caption 执行 NFC 规范化和首尾空白清理，
再把完全相同的 caption 作为不可拆分组，通过确定性 exact subset-sum 得到精确的
7,232/256/512。这样同一 prompt 的多个视频不会跨 split。manifest 中每条记录均保存
`curation_split`；`TARDIS_SOTA/configs/split_manifest_lock.json` 还锁定 manifest SHA、划分
策略和三个 split 的 ID SHA-256。

## 来源策略

### Seedance

- 仓库：`GokuScraper/seedance-2-prompts-datasets`
- Revision：`515aa5bd59123fb489914ce9cd21419badb08be4`
- 封装：独立 MP4/WebM/MOV
- 策略：从有完整 prompt 和本地媒体的候选中确定性选择 8,000 条，随后删除所有未被正式
  manifest 引用的媒体
- 当前已验收有效媒体：`45,590,612,736` 字节
- 划分策略：`caption_group_v1`；train/validation/test caption 交集均为 0
- 当前 manifest SHA-256：`a82ebf193a099b93d428b656ed4e19d411078f71b7b3286eec7cc5a2d12109df`

### DataVerse

- 仓库：`Vchitect/Vchitect_T2V_DataVerse`
- Revision：`e068be25f4d06a837992a1e9096fd00105c83f2c`
- 封装：保留原始 TAR，不解压
- 策略：从 1,906 个远端 TAR 的固定 revision 文件树中选择 8 个完整分片；每包 1,000 条，
  在不重复、不重编码的条件下同时满足 8,000 条和约 45 GB

固定 TAR：

```text
00000/000127.tar
00001/000397.tar
00002/000599.tar
00003/000720.tar
00004/000807.tar
00005/001038.tar
00006/001353.tar
00007/001496.tar
```

八个归档的仓库字节总量为 `45,000,734,720`。当前已验收的 8,000 个视频成员总量为
`44,994,529,600` 字节；归档头和填充字节不计入有效媒体体积。

### OpenVid-1M

- 仓库：`nkp37/OpenVid-1M`
- Revision：`d8a63bd22989c80b5734ec2bb989f4e1b61a5807`
- 候选包：`OpenVid_part100/84/108/114/85.zip`
- 策略：只通过 HTTP Range 读取五个 ZIP 的中央目录，与固定 revision 的 CSV 连接得到
  10,000 个候选；以美学、时域一致性和适中运动量作为质量项，在 45 GB 约束下选出 8,000 条
- 选择计划有效媒体：`45,001,385,093` 字节

OpenVid 不会把约 100 GB 候选 ZIP 长期留在数据盘。每个源 ZIP 按顺序断点下载，选中的成员
被重打包为无压缩 TAR；该 TAR 通过成员数量和字节验收后，源 ZIP 立即删除。五个来源包约各
贡献 1,600 条，避免由单一分片主导。

## Manifest 契约

每个来源根目录都有：

```text
tardis_manifest.jsonl
curation_report.json
```

每条 manifest 记录至少包含：

```text
id
caption
media_locator
source
metadata.revision
metadata.media_bytes
metadata.quality_score
metadata.curation_policy
metadata.curation_split
```

DataVerse 和 OpenVid 还记录原始归档及成员名。训练通过本地字节范围直接读取 TAR 成员，不会
为训练再次解压出一份视频。

## 验收规则

`--verify-only` 会同时检查：

1. 每源恰好 8,000 个唯一记录；
2. manifest 引用的视频总量处于 44-46 GB；
3. Train/Validation/Test 恰好为 7,232/256/512；
4. 所有媒体路径均存在，成员大小与 manifest 一致；
5. 不存在未引用视频、未引用 TAR 或遗留 OpenVid 源 ZIP；
6. `curation_report.json` 与 manifest 的条目数、字节数和划分一致。
7. Seedance 相同 NFC-normalized caption 不得跨 split，持久化 split 必须与运行时划分一致。

Train 和 Infer 每个进程仍只装配 `TARDIS_DATASET` 指定的一个来源。三个来源分别训练、分别
管理权重，不会在同一个进程中混合。
