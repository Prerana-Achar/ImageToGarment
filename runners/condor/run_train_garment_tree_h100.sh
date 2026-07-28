#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_smplx}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/garment_tree}"
VENV="${VENV:-${PROJECT_DIR}/venv}"

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-288}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"
ALLOW_UNBALANCED_DATA="${ALLOW_UNBALANCED_DATA:-1}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"

if [ ! -f "${PREPARED_DIR}/schema.json" ]; then
  echo "ERROR: prepared data missing at ${PREPARED_DIR}" >&2
  echo "Run: bash scripts/prepare_smplx_data.sh" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

EXTRA_ARGS=()
if [ "${ALLOW_UNBALANCED_DATA}" = "1" ]; then
  EXTRA_ARGS+=(--allow-unbalanced-data)
fi

nvidia-smi || true
python -u "${PROJECT_DIR}/train_garment_tree.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --device cuda \
  --amp \
  "${EXTRA_ARGS[@]}"

