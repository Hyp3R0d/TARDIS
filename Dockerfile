# syntax=docker/dockerfile:1
# TARDIS GPU 服务镜像：训练 / 推理 / 应用。
# 构建：  docker build -t tardis:latest .          (默认 train 行为)
#         docker build --target infer -t tardis:infer .
#         docker build --target apply -t tardis:apply .
# 运行：  docker run --gpus all --rm -it tardis:latest

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

FROM ${CUDA_IMAGE} AS base

ARG TORCH_INDEX
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 先装与 CUDA 匹配的 torch/torchvision，避免 pip install . 回落到 CPU 版
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --index-url ${TORCH_INDEX} "torch>=2.8,<2.9" "torchvision>=0.23,<0.24"

WORKDIR /workspace

COPY pyproject.toml README.md LICENSE datasets.txt ./
COPY tardis ./tardis
COPY scripts ./scripts

RUN python3 -m pip install .

ENV TARDIS_STORAGE_ROOT=/workspace \
    TARDIS_DATASETS_FILE=/workspace/datasets.txt \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/workspace/cache/huggingface

RUN mkdir -p /workspace/cache /workspace/checkpoints /workspace/outputs

FROM base AS train
ENTRYPOINT ["bash", "scripts/train.sh"]

FROM base AS infer
ENTRYPOINT ["bash", "scripts/infer.sh"]

FROM base AS apply
ENTRYPOINT ["bash", "scripts/apply.sh"]

FROM train AS latest
