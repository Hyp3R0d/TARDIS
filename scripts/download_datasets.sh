#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STORAGE_ROOT="${TARDIS_STORAGE_ROOT:-/root/autodl-tmp/TARDIS}"
DATA_ROOT="${STORAGE_ROOT}/datasets"
LOG_ROOT="${STORAGE_ROOT}/logs/datasets"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache/xdg}"

SEEDANCE_REPO="GokuScraper/seedance-2-prompts-datasets"
SEEDANCE_REVISION="515aa5bd59123fb489914ce9cd21419badb08be4"
OPENVID_REPO="nkp37/OpenVid-1M"
OPENVID_REVISION="d8a63bd22989c80b5734ec2bb989f4e1b61a5807"
MAX_DOWNLOAD_ATTEMPTS="${TARDIS_DOWNLOAD_ATTEMPTS:-30}"

for command in hf aria2c python; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "缺少数据准备命令：${command}" >&2
    exit 127
  fi
done
if [[ ! "${MAX_DOWNLOAD_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARDIS_DOWNLOAD_ATTEMPTS 必须是正整数。" >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}" "${LOG_ROOT}" "${HF_HUB_CACHE}"

retry() {
  local label="$1"
  shift
  local attempt
  local delay
  for ((attempt = 1; attempt <= MAX_DOWNLOAD_ATTEMPTS; attempt++)); do
    echo "[${label}] 尝试 ${attempt}/${MAX_DOWNLOAD_ATTEMPTS}"
    if "$@"; then
      return 0
    fi
    if ((attempt == MAX_DOWNLOAD_ATTEMPTS)); then
      echo "[${label}] 已耗尽全部重试次数。" >&2
      return 1
    fi
    delay=$((attempt * 10))
    if ((delay > 120)); then
      delay=120
    fi
    echo "[${label}] 镜像瞬时失败，${delay} 秒后从断点续传。" >&2
    sleep "${delay}"
  done
}

verify_source() {
  local source="$1"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m tardis.data.curate_local "${source}" \
      --data-root "${DATA_ROOT}" \
      --verify-only
}

prepare_seedance() {
  if verify_source seedance >/dev/null 2>&1; then
    echo "Seedance 已通过正式验收，跳过下载。"
    return 0
  fi
  retry seedance hf download "${SEEDANCE_REPO}" \
    --repo-type dataset \
    --revision "${SEEDANCE_REVISION}" \
    --local-dir "${DATA_ROOT}/seedance-2-prompts-datasets" \
    --max-workers 4
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m tardis.data.curate_local seedance --data-root "${DATA_ROOT}"
}

prepare_dataverse() {
  if verify_source dataverse >/dev/null 2>&1; then
    echo "DataVerse 已通过正式验收，跳过下载。"
    return 0
  fi
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m tardis.data.curate_local dataverse \
      --data-root "${DATA_ROOT}" \
      --download-missing
}

prepare_openvid() {
  if verify_source openvid >/dev/null 2>&1; then
    echo "OpenVid 已通过正式验收，跳过下载。"
    return 0
  fi
  retry openvid-metadata hf download "${OPENVID_REPO}" \
    .gitattributes \
    README.md \
    OpenVid-1M.png \
    data/train/OpenVid-1M.csv \
    --repo-type dataset \
    --revision "${OPENVID_REVISION}" \
    --local-dir "${DATA_ROOT}/OpenVid-1M" \
    --max-workers 2
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m tardis.data.curate_local openvid \
      --data-root "${DATA_ROOT}" \
      --download-missing
}

prepare_seedance
prepare_dataverse
prepare_openvid

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m tardis.data.curate_local all \
    --data-root "${DATA_ROOT}" \
    --verify-only

du -sh "${DATA_ROOT}"/*
echo "三个平衡数据集均已通过 8,000 条、44-46 GB 和固定划分验收。"
