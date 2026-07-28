#!/bin/bash
set -euo pipefail

if [ -z "${IMAGE_PATH:-}" ]; then
  echo "ERROR: IMAGE_PATH must be set by the submit wrapper" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/runs/person_garment_trellis_gnn}"
SETUP_ROOT="${SETUP_ROOT:-/is/cluster/fast/pachar/Data/ImageToGarment/trellis2_setup}"
JOB_TAG="${LOG_DIR:-person_garment_trellis_gnn_${CONDOR_CLUSTER:-manual}_${CONDOR_PROCESS:-0}}"
CUDA_HOME="${CUDA_HOME:-}"

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
  echo "ERROR: Could not find nvcc. Load a CUDA toolkit module or set CUDA_HOME." >&2
  exit 3
fi

export PROJECT_DIR
export VENV_DIR
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SETUP_ROOT}/tmp/${JOB_TAG}"
export HF_HOME="${HF_HOME:-${SETUP_ROOT}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SETUP_ROOT}/pip_cache}"
export TORCH_HOME="${TORCH_HOME:-${SETUP_ROOT}/torch_home}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$OUT_ROOT"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

echo "Host: $(hostname)"
echo "Image: $IMAGE_PATH"
echo "Out root: $OUT_ROOT"
echo "CUDA_HOME: $CUDA_HOME"
echo "HF_HOME: $HF_HOME"
echo "Confidence thresholds: root=${MIN_ROOT_CONFIDENCE:-0.75}, categorical=${MIN_ACTIVE_CATEGORICAL_CONFIDENCE:-0.65}, numeric_prob=${MIN_ACTIVE_NUMERIC_PROBABILITY:-0.55}, numeric_uncertainty=${MAX_ACTIVE_NUMERIC_UNCERTAINTY:-0.35}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

cmd=(
  python -u scripts/person_garment_trellis_gnn_pipeline.py
  --image "$IMAGE_PATH"
  --out-dir "$OUT_ROOT"
  --gnn-epochs "${GNN_EPOCHS:-200}"
  --gnn-surface-samples "${GNN_SURFACE_SAMPLES:-8192}"
  --min-root-confidence "${MIN_ROOT_CONFIDENCE:-0.75}"
  --min-active-categorical-confidence "${MIN_ACTIVE_CATEGORICAL_CONFIDENCE:-0.65}"
  --min-active-numeric-probability "${MIN_ACTIVE_NUMERIC_PROBABILITY:-0.55}"
  --max-active-numeric-uncertainty "${MAX_ACTIVE_NUMERIC_UNCERTAINTY:-0.35}"
)

if [ -n "${CHECKPOINT:-}" ]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "${FORCE_PIPELINE:-}" ]; then
  cmd+=(--force)
fi
if [ -n "${ALLOW_LOW_CONFIDENCE_GARMENTCODE:-}" ]; then
  cmd+=(--allow-low-confidence-garmentcode)
fi

"${cmd[@]}"