#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
IMAGE_ROOT="${IMAGE_ROOT:-/is/cluster/fast/pachar/Data/GarmentImage}"
CHECKPOINT="${CHECKPOINT:-}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/garmentimage_comparisons/garment_tree_ensemble}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
RENDER_RESOLUTION_SCALE="${RENDER_RESOLUTION_SCALE:-2.0}"
LIMIT="${LIMIT:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_SIM_STEPS="${MAX_SIM_STEPS:-}"
MAX_SIM_TIME="${MAX_SIM_TIME:-}"

if [ -z "${CHECKPOINT}" ]; then echo "ERROR: CHECKPOINT is required" >&2; exit 2; fi
if [ ! -x "${VENV}/bin/python" ]; then echo "ERROR: venv missing: ${VENV}" >&2; exit 3; fi
if [ ! -f "${CHECKPOINT}" ]; then echo "ERROR: checkpoint missing: ${CHECKPOINT}" >&2; exit 4; fi
if [ ! -d "${IMAGE_ROOT}" ]; then echo "ERROR: image root missing: ${IMAGE_ROOT}" >&2; exit 5; fi

mkdir -p "${OUT_DIR}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.cache/torch}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
echo "Host: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Images: ${IMAGE_ROOT}"
echo "Outputs: ${OUT_DIR}"
nvidia-smi

args=(
  "${PROJECT_DIR}/scripts/garmentimages_batch_infer_garment_tree.py"
  --checkpoint "${CHECKPOINT}"
  --image-root "${IMAGE_ROOT}"
  --out-dir "${OUT_DIR}"
  --device cuda
  --render-resolution-scale "${RENDER_RESOLUTION_SCALE}"
)
if [ -n "${LIMIT}" ]; then args+=(--limit "${LIMIT}"); fi
if [ "${SKIP_EXISTING}" = "1" ]; then args+=(--skip-existing); fi
if [ -n "${MAX_SIM_STEPS}" ]; then args+=(--max-sim-steps "${MAX_SIM_STEPS}"); fi
if [ -n "${MAX_SIM_TIME}" ]; then args+=(--max-sim-time "${MAX_SIM_TIME}"); fi

"${VENV}/bin/python" -u "${args[@]}"