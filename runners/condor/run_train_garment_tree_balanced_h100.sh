#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_smplx_balanced}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/garment_tree_balanced_v1}"
VENV="${VENV:-${PROJECT_DIR}/venv}"

EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-80}"
PATIENCE="${PATIENCE:-60}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-288}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"

if [ ! -f "${PREPARED_DIR}/schema.json" ] || [ ! -f "${PREPARED_DIR}/balance_report.json" ]; then
  echo "ERROR: balanced prepared dataset is missing at ${PREPARED_DIR}" >&2
  echo "Run: bash scripts/compile_balanced_smplx_data.sh" >&2
  exit 3
fi

READY="$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["ready"])).lower())' "${PREPARED_DIR}/balance_report.json")"
if [ "${READY}" != "true" ]; then
  echo "ERROR: ${PREPARED_DIR}/balance_report.json is not ready" >&2
  exit 4
fi

mkdir -p "${OUT_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

echo "Host:          $(hostname)"
echo "Prepared data: ${PREPARED_DIR}"
echo "Output:        ${OUT_DIR}"
echo "Epochs:        ${EPOCHS}"
echo "Batch size:    ${BATCH_SIZE}"
nvidia-smi

python -u "${PROJECT_DIR}/train_garment_tree.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --min-epochs "${MIN_EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --device cuda \
  --amp

