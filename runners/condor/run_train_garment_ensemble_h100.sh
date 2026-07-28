#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_smplx_output_balanced}"
AUX_PREPARED_DIR="${AUX_PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_chatgarment_full_current}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/garment_tree_ensemble_pose_ema_v4}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
MEMBERS="${MEMBERS:-5}"
FOLDS="${FOLDS:-5}"
OOF_EPOCHS="${OOF_EPOCHS:-30}"
EPOCHS="${EPOCHS:-180}"
ROUTE_DIMENSION="${ROUTE_DIMENSION:-64}"
ROUTE_CLASS_TOKEN_SCALE="${ROUTE_CLASS_TOKEN_SCALE:-1.0}"
ROUTE_FREEZE_EPOCH="${ROUTE_FREEZE_EPOCH:-30}"
SAMPLES_PER_GARMENT="${SAMPLES_PER_GARMENT:-4}"
ROOT_BALANCE_STRENGTH="${ROOT_BALANCE_STRENGTH:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
FEATURE_CACHE_AUGMENTATIONS="${FEATURE_CACHE_AUGMENTATIONS:-3}"
VALIDATION_EVERY="${VALIDATION_EVERY:-5}"
EARLY_VALIDATION_EPOCHS="${EARLY_VALIDATION_EPOCHS:-20}"
TEACHER_FORCE_EPOCHS="${TEACHER_FORCE_EPOCHS:-10}"
TEACHER_FORCE_START="${TEACHER_FORCE_START:-0.5}"
AUX_PRETRAIN_EPOCHS="${AUX_PRETRAIN_EPOCHS:-15}"
AUX_BATCH_FRACTION="${AUX_BATCH_FRACTION:-0.25}"
AUX_FEATURE_CACHE_AUGMENTATIONS="${AUX_FEATURE_CACHE_AUGMENTATIONS:-1}"
AUX_ROUTER_WEIGHT="${AUX_ROUTER_WEIGHT:-0.50}"
WANDB_PROJECT="${WANDB_PROJECT:-ImageToGarment}"
WANDB_ENTITY="${WANDB_ENTITY:-draping}"
WANDB_MODE="${WANDB_MODE:-online}"
BACKBONE_NAME="${BACKBONE_NAME:-dinov2_vitb14}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"
for required in schema.json targets.npz images.json splits.json balance_report.json balanced_sampling.json; do
  if [ ! -f "${PREPARED_DIR}/${required}" ]; then
    echo "ERROR: missing ${PREPARED_DIR}/${required}" >&2
    exit 3
  fi
done
for required in schema.json targets.npz images.json splits.json compatibility_report.json; do
  if [ ! -f "${AUX_PREPARED_DIR}/${required}" ]; then
    echo "ERROR: missing ${AUX_PREPARED_DIR}/${required}" >&2
    echo "Run: bash scripts/compile_chatgarment_auxiliary.sh" >&2
    exit 4
  fi
done
mkdir -p "${OUT_DIR}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
nvidia-smi

python -u "${PROJECT_DIR}/train_garment_ensemble.py" \
  --prepared-dir "${PREPARED_DIR}" \
  --aux-prepared-dir "${AUX_PREPARED_DIR}" \
  --aux-pretrain-epochs "${AUX_PRETRAIN_EPOCHS}" \
  --aux-batch-fraction "${AUX_BATCH_FRACTION}" \
  --aux-feature-cache-augmentations "${AUX_FEATURE_CACHE_AUGMENTATIONS}" \
  --out-dir "${OUT_DIR}" \
  --members "${MEMBERS}" \
  --folds "${FOLDS}" \
  --oof-epochs "${OOF_EPOCHS}" \
  --epochs "${EPOCHS}" \
  --router-warmup-epochs 0 \
  --samples-per-garment "${SAMPLES_PER_GARMENT}" \
  --root-balance-strength "${ROOT_BALANCE_STRENGTH}" \
  --bagging-fraction 0.85 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --lr 3e-4 \
  --min-lr 2e-6 \
  --weight-decay 0.05 \
  --dimension 64 \
  --route-dimension "${ROUTE_DIMENSION}" \
  --route-class-token-scale "${ROUTE_CLASS_TOKEN_SCALE}" \
  --route-freeze-epoch "${ROUTE_FREEZE_EPOCH}" \
  --query-layers 1 \
  --attention-heads 4 \
  --dropout 0.25 \
  --feature-dropout 0.20 \
  --feature-noise 0.03 \
  --feature-cache-augmentations "${FEATURE_CACHE_AUGMENTATIONS}" \
  --class-token-scale 0.0 \
  --label-smoothing 0.10 \
  --router-label-smoothing 0.05 \
  --lambda-router 1.0 \
  --aux-router-weight "${AUX_ROUTER_WEIGHT}" \
  --lambda-pose-consistency 0.15 \
  --ema-decay 0.995 \
  --validation-every "${VALIDATION_EVERY}" \
  --early-validation-epochs "${EARLY_VALIDATION_EPOCHS}" \
  --teacher-force-epochs "${TEACHER_FORCE_EPOCHS}" \
  --teacher-force-start "${TEACHER_FORCE_START}" \
  --stacker-steps 400 \
  --stacker-lr 0.05 \
  --backbone-name "${BACKBONE_NAME}" \
  --backbone-repo DINOv2 \
  --aspect-pad \
  --amp \
  --device cuda \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-tags garment-tree isolated-route
