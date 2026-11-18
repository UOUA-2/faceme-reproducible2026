#!/usr/bin/env bash
# ICVIP/DaChuang 的 F 盘环境配置。
# 使用：source /mnt/f/ICVIP/DaChuang/scripts/env_f_drive.sh
# 目录创建、写入测试或路径检查失败时直接退出，不回退到 C 盘。

set -euo pipefail

export ICVIP_ROOT=/mnt/f/ICVIP
export ICVIP_PROJECT=/mnt/f/ICVIP/DaChuang
export ICVIP_MODELS=/mnt/f/ICVIP/DaChuang/models
export ICVIP_WEIGHTS=/mnt/f/ICVIP/DaChuang/weights
export ICVIP_ENV=/mnt/f/ICVIP/envs/dachuang

export XDG_CACHE_HOME=/mnt/f/ICVIP/cache/xdg
export HF_HOME=/mnt/f/ICVIP/cache/huggingface
export HF_HUB_CACHE=/mnt/f/ICVIP/cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/mnt/f/ICVIP/cache/huggingface/hub
export HF_DATASETS_CACHE=/mnt/f/ICVIP/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/f/ICVIP/cache/huggingface/transformers
export TORCH_HOME=/mnt/f/ICVIP/cache/torch
export PIP_CACHE_DIR=/mnt/f/ICVIP/cache/pip
export CONDA_PKGS_DIRS=/mnt/f/ICVIP/cache/conda/pkgs
export CONDA_ENVS_PATH=/mnt/f/ICVIP/envs
export GRADIO_TEMP_DIR=/mnt/f/ICVIP/cache/gradio
export TMPDIR=/mnt/f/ICVIP/cache/tmp

required_dirs=(
  "$XDG_CACHE_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
  "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR"
  "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$GRADIO_TEMP_DIR" "$TMPDIR"
)

for dir_path in "${required_dirs[@]}"; do
  mkdir -p "$dir_path"
done

printf '%s\n' \
  "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
  "HF_HOME=$HF_HOME" \
  "HF_HUB_CACHE=$HF_HUB_CACHE" \
  "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" \
  "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE" \
  "TORCH_HOME=$TORCH_HOME" \
  "PIP_CACHE_DIR=$PIP_CACHE_DIR" \
  "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS" \
  "CONDA_ENVS_PATH=$CONDA_ENVS_PATH" \
  "GRADIO_TEMP_DIR=$GRADIO_TEMP_DIR" \
  "TMPDIR=$TMPDIR"

for dir_path in "${required_dirs[@]}"; do
  if [[ ! -d "$dir_path" || ! -w "$dir_path" ]]; then
    printf 'ERROR: F-drive directory is missing or not writable: %s\n' "$dir_path" >&2
    return 1 2>/dev/null || exit 1
  fi
  printf 'WRITABLE: %s\n' "$dir_path"
done

write_probe="$TMPDIR/.env_f_drive_write_test.$$"
printf 'icvip-f-drive-write-test\n' > "$write_probe"
rm -f "$write_probe"
printf 'F_DRIVE_WRITE_TEST=PASS\n'
