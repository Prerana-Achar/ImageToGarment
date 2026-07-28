#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_smplx_output_balanced}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/garment_tree_foundation_small_dd_v1}"
VENV="${VENV:-${PROJECT_DIR}/venv}"

EPOCHS="${EPOCHS:-500}"
MIN_EPOCHS="${MIN_EPOCHS:-500}"
PATIENCE="${PATIENCE:-500}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
LAMBDA_ORDINAL="${LAMBDA_ORDINAL:-0.20}"
LAMBDA_CONSISTENCY="${LAMBDA_CONSISTENCY:-0.10}"
TEACHER_FORCE_EPOCHS="${TEACHER_FORCE_EPOCHS:-100}"
DROPOUT="${DROPOUT:-0.10}"
DROP_PATH="${DROP_PATH:-0.10}"
WANDB_PROJECT="${WANDB_PROJECT:-ImageToGarment}"
WANDB_ENTITY="${WANDB_ENTITY:-draping}"
WANDB_MODE="${WANDB_MODE:-online}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"

for REQUIRED in schema.json balance_report.json balanced_sampling.json; do
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
export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
nvidia-smi

TRAIN_ARGS=(
  --prepared-dir "$PREPARED_DIR"
  --out-dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --min-epochs "$MIN_EPOCHS"
  --patience "$PATIENCE"
  --no-early-stopping
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --warmup-epochs "$WARMUP_EPOCHS"
  --weight-decay "$WEIGHT_DECAY"
  --label-smoothing "$LABEL_SMOOTHING"
  --lambda-ordinal "$LAMBDA_ORDINAL"
  --lambda-consistency "$LAMBDA_CONSISTENCY"
  --teacher-force-epochs "$TEACHER_FORCE_EPOCHS"
  --dropout "$DROPOUT"
  --drop-path "$DROP_PATH"
  --dimension 128
  --query-layers 2
  --attention-heads 4
  --train-eval-every 5
  --save-every 50
  --augmentation domain
  --aspect-pad
  --encoder-kind foundation
  --backbone-name dinov2_vits14
  --backbone-repo DINOv2
  --freeze-backbone
  --hierarchical-categoricals
  --device cuda
  --amp
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --wandb-mode "$WANDB_MODE"
  --wandb-tags garment-tree foundation-pretrained smaller-decoder double-descent hierarchical output-balanced h100
)
printf 'trainer arguments:'
printf ' <%q>' "${TRAIN_ARGS[@]}"
printf '\n'
python -u "$PROJECT_DIR/train_garment_tree_output_balanced.py" "${TRAIN_ARGS[@]}"