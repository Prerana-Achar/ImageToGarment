#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
TRELLIS2_DIR="${TRELLIS2_DIR:-${PROJECT_DIR}/third_party/TRELLIS.2}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
CUDA_HOME="${CUDA_HOME:-}"
SETUP_ROOT="${SETUP_ROOT:-/is/cluster/fast/pachar/Data/ImageToGarment/trellis2_setup}"
JOB_TAG="${LOG_DIR:-trellis2_setup_${CONDOR_CLUSTER:-manual}_${CONDOR_PROCESS:-0}}"

mkdir -p "${SETUP_ROOT}/tmp" "${SETUP_ROOT}/extensions" "${SETUP_ROOT}/hf_home" "${SETUP_ROOT}/pip_cache"

export PROJECT_DIR
export TRELLIS2_DIR
export VENV_DIR

if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    for candidate in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.1 /opt/cuda /opt/cuda-13.0 /opt/cuda-12.4; do
      if [ -x "$candidate/bin/nvcc" ]; then
        CUDA_HOME="$candidate"
        break
      fi
    done
  fi
fi

if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: Could not find nvcc. Load a CUDA toolkit module or set CUDA_HOME to a path containing bin/nvcc." >&2
  echo "Checked PATH and common /usr/local/cuda* /opt/cuda* locations." >&2
  exit 2
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SETUP_ROOT}/tmp/${JOB_TAG}"
export TRELLIS2_EXT_DIR="${SETUP_ROOT}/extensions"
export HF_HOME="${HF_HOME:-${SETUP_ROOT}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SETUP_ROOT}/pip_cache}"
export REBUILD_EXTENSIONS="${REBUILD_EXTENSIONS:-0}"

mkdir -p "$TMPDIR" "$TRELLIS2_EXT_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$PIP_CACHE_DIR"

cd "$PROJECT_DIR"

echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "TRELLIS2_DIR: $TRELLIS2_DIR"
echo "VENV_DIR: $VENV_DIR"
echo "CUDA_HOME: $CUDA_HOME"
echo "TMPDIR: $TMPDIR"
echo "TRELLIS2_EXT_DIR: $TRELLIS2_EXT_DIR"
echo "HF_HOME: $HF_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

bash scripts/setup_trellis2_cluster.sh