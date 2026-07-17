#!/bin/bash

# Condor payload for training either DINOv2 baseline or model.py multihead.
# This script is intentionally small: all heavy lifting stays in
# dinov2_pipeline.py.

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_all}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
ARCHITECTURE="${ARCHITECTURE:-modelpy}"
BACKBONE="${BACKBONE:-dinov2_vitl14}"
MODE="${MODE:-all_images}"
AUGMENTATION="${AUGMENTATION:-light}"
EPOCHS="${EPOCHS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-3}"
LAMBDA_CAT="${LAMBDA_CAT:-1.0}"
DROPOUT="${DROPOUT:-0.4}"
HEAD_HIDDEN_DIMS="${HEAD_HIDDEN_DIMS:-64 32}"
HEAD_LAYER_NORM="${HEAD_LAYER_NORM:-1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-effective}"
CLASS_WEIGHT_BETA="${CLASS_WEIGHT_BETA:-0.999}"
CLASS_WEIGHT_MAX="${CLASS_WEIGHT_MAX:-5.0}"
REG_LOSS="${REG_LOSS:-smooth_l1}"
SMOOTH_L1_BETA="${SMOOTH_L1_BETA:-0.05}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
LR_PLATEAU_PATIENCE="${LR_PLATEAU_PATIENCE:-5}"
LR_PLATEAU_FACTOR="${LR_PLATEAU_FACTOR:-0.5}"
MIN_LR="${MIN_LR:-1e-6}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-5e-4}"
SHARED_HIDDEN_DIM="${SHARED_HIDDEN_DIM:-}"
MODEL_SCHEMA="${MODEL_SCHEMA:-${PROJECT_DIR}/GarmentCodeRC/assets/design_params/default_new.yaml}"
DINOV2_DIR="${DINOV2_DIR:-${PROJECT_DIR}/DINOv2}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/${ARCHITECTURE}_${BACKBONE}_h100}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ImageToGarment}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-condor h100}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --architecture)
      ARCHITECTURE="$2"
      shift 2
      ;;
    --prepared-dir)
      PREPARED_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Host:          $(hostname)"
echo "Project:       ${PROJECT_DIR}"
echo "Architecture:  ${ARCHITECTURE}"
echo "Prepared data: ${PREPARED_DIR}"
echo "Output dir:    ${OUT_DIR}"
echo "Backbone:      ${BACKBONE}"
echo "Batch size:    ${BATCH_SIZE}"

cd "${PROJECT_DIR}" || exit 2

if [ -f "${PROJECT_DIR}/runners/condor/secrets.sh" ]; then
  . "${PROJECT_DIR}/runners/condor/secrets.sh"
fi

if [ ! -d "${VENV}" ]; then
  echo "ERROR: venv not found at ${VENV}" >&2
  exit 2
fi
. "${VENV}/bin/activate"

if [ ! -f "${PREPARED_DIR}/schema.json" ]; then
  echo "ERROR: prepared data missing at ${PREPARED_DIR}" >&2
  echo "Run: bash scripts/fetch_all_fast_data.sh" >&2
  exit 3
fi

if [ "${ARCHITECTURE}" = "modelpy" ] && [ ! -f "${MODEL_SCHEMA}" ]; then
  echo "ERROR: MODEL_SCHEMA missing: ${MODEL_SCHEMA}" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${PROJECT_DIR}/.cache/torch"
mkdir -p "${DATA_ROOT}/.cache/huggingface"

export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.cache/torch}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${PROJECT_DIR}/.cache/torch/kernels}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

WANDB_ARGS=()
if [ "${WANDB}" = "1" ]; then
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename "${OUT_DIR}")}"
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}" --wandb-mode "${WANDB_MODE}")
  if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [ -n "${WANDB_TAGS}" ]; then
    WANDB_ARGS+=(--wandb-tags ${WANDB_TAGS})
  fi
fi

REGULARIZATION_ARGS=(
  --augmentation "${AUGMENTATION}"
  --label-smoothing "${LABEL_SMOOTHING}"
  --class-weighting "${CLASS_WEIGHTING}"
  --class-weight-beta "${CLASS_WEIGHT_BETA}"
  --class-weight-max "${CLASS_WEIGHT_MAX}"
  --reg-loss "${REG_LOSS}"
  --smooth-l1-beta "${SMOOTH_L1_BETA}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --lr-plateau-patience "${LR_PLATEAU_PATIENCE}"
  --lr-plateau-factor "${LR_PLATEAU_FACTOR}"
  --min-lr "${MIN_LR}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --min-delta "${MIN_DELTA}"
)
if [ "${HEAD_LAYER_NORM}" = "1" ]; then
  REGULARIZATION_ARGS+=(--head-layer-norm)
fi

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

if [ "${ARCHITECTURE}" = "baseline" ]; then
  python -u "${PROJECT_DIR}/dinov2_pipeline.py" train \
    --architecture baseline \
    --prepared-dir "${PREPARED_DIR}" \
    --out-dir "${OUT_DIR}" \
    --backbone "${BACKBONE}" \
    --mode "${MODE}" \
    --epochs "${EPOCHS}" \
    --save-every "${SAVE_EVERY}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lambda-cat "${LAMBDA_CAT}" \
    --dropout "${DROPOUT}" \
    "${REGULARIZATION_ARGS[@]}" \
    --device cuda \
    --amp \
    "${WANDB_ARGS[@]}"
elif [ "${ARCHITECTURE}" = "modelpy" ]; then
  if [ -n "${SHARED_HIDDEN_DIM}" ]; then
    python -u "${PROJECT_DIR}/dinov2_pipeline.py" train \
      --architecture modelpy \
      --prepared-dir "${PREPARED_DIR}" \
      --out-dir "${OUT_DIR}" \
      --backbone "${BACKBONE}" \
      --mode "${MODE}" \
      --epochs "${EPOCHS}" \
      --save-every "${SAVE_EVERY}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --lambda-cat "${LAMBDA_CAT}" \
      "${REGULARIZATION_ARGS[@]}" \
      --dropout "${DROPOUT}" \
      --shared-hidden-dim "${SHARED_HIDDEN_DIM}" \
      --head-hidden-dims ${HEAD_HIDDEN_DIMS} \
      --model-schema "${MODEL_SCHEMA}" \
      --dinov2-dir "${DINOV2_DIR}" \
      --device cuda \
      --amp \
      "${WANDB_ARGS[@]}"
  else
    python -u "${PROJECT_DIR}/dinov2_pipeline.py" train \
      --architecture modelpy \
      --prepared-dir "${PREPARED_DIR}" \
      --out-dir "${OUT_DIR}" \
      --backbone "${BACKBONE}" \
      --mode "${MODE}" \
      --epochs "${EPOCHS}" \
      --save-every "${SAVE_EVERY}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --lambda-cat "${LAMBDA_CAT}" \
      "${REGULARIZATION_ARGS[@]}" \
      --dropout "${DROPOUT}" \
      --head-hidden-dims ${HEAD_HIDDEN_DIMS} \
      --model-schema "${MODEL_SCHEMA}" \
      --dinov2-dir "${DINOV2_DIR}" \
      --device cuda \
      --amp \
      "${WANDB_ARGS[@]}"
  fi
else
  echo "ERROR: ARCHITECTURE must be baseline or modelpy, got ${ARCHITECTURE}" >&2
  exit 2
fi
