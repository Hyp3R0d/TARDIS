#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

IMAGE_NAME="${IMAGE_NAME:-tardis}"
MODE="${1:-train}"

case "${MODE}" in
  train|infer|apply) ;;
  *)
    echo "usage: $0 [train|infer|apply]" >&2
    exit 2
    ;;
esac

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not reachable" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warning: nvidia-smi not found on host, --gpus all may fail" >&2
fi

mkdir -p "${REPO_ROOT}/checkpoints" "${REPO_ROOT}/outputs" "${REPO_ROOT}/data"

docker run --gpus all --rm -it \
  --name "tardis-${MODE}" \
  --ipc host \
  --shm-size 16g \
  -v "${REPO_ROOT}/checkpoints:/workspace/checkpoints" \
  -v "${REPO_ROOT}/outputs:/workspace/outputs" \
  -v "${REPO_ROOT}/data:/workspace/data" \
  -e "TARDIS_NPROC=${TARDIS_NPROC:-1}" \
  "${IMAGE_NAME}:${MODE}"
