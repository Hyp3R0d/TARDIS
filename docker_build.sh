#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

IMAGE_NAME="${IMAGE_NAME:-tardis}"
TARGET="${TARGET:-latest}"
TAG="${TAG:-${TARGET}}"

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not reachable" >&2
  exit 1
fi

echo "building ${IMAGE_NAME}:${TAG} (target=${TARGET})"
docker build --target "${TARGET}" -t "${IMAGE_NAME}:${TAG}" .
