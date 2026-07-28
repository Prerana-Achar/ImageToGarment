#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_smplx_output_balanced}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/garment_tree_output_balanced_v1}"
VENV="${VENV:-${PROJECT_DIR}/venv}"

EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-80}"
PATIENCE="${PATIENCE:-60}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-288}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"

for REQUIRED in schema.json balance_report.json sample_weights.json; do
  if [ ! -f "${PREPARED_DIR}/${REQUIRED}" ]; then
    echo "ERROR: missing ${PREPARED_DIR}/${REQUIRED}" >&2
    echo "Run: bash scripts/compile_output_balanced_smplx_data.sh" >&2
    exit 3
  fi
done

READY="$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["ready"])).lower())' "${PREPARED_DIR}/balance_report.json")"
if [ "${READY}" != "true" ]; then
  echo "ERROR: output-balanced data report is not ready" >&2
  exit 4
fi

mkdir -p "${OUT_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
nvidia-smi

python -u "${PROJECT_DIR}/train_garment_tree_output_balanced.py" \
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

