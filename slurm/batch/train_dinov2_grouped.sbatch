#!/bin/bash
#SBATCH --job-name=dino-grouped
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="/mnt/beegfs/home/stud136/3dv_project"
CONDA_SH="/home/stud136/miniconda3/etc/profile.d/conda.sh"

cd "${PROJECT_DIR}"
source "${CONDA_SH}"
conda activate project

export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PREPARED_DIR="${PREPARED_DIR:-prepared_v2}"
OUT_DIR="${OUT_DIR:-runs/dinov2_grouped}"
BACKBONE="${BACKBONE:-dinov2_vits14}"
MODE="${MODE:-single}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.01}"
LAMBDA_CAT="${LAMBDA_CAT:-0.5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
REG_LOSS="${REG_LOSS:-huber}"
HUBER_DELTA="${HUBER_DELTA:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
CAT_FREEZE_PATIENCE="${CAT_FREEZE_PATIENCE:-3}"
CAT_FREEZE_MIN_EPOCHS="${CAT_FREEZE_MIN_EPOCHS:-5}"
CAT_FREEZE_MIN_DELTA="${CAT_FREEZE_MIN_DELTA:-1e-4}"

# This is a no-op for manifests already prepared on the cluster, but remaps
# older manifests containing the original local-machine image prefix.
OLD_IMAGE_PREFIX="${OLD_IMAGE_PREFIX:-/Users/siddharth/Study/3dv_project/data/chatgarment_data/garments_imgs_v2_3}"
NEW_IMAGE_PREFIX="${NEW_IMAGE_PREFIX:-${PROJECT_DIR}/data/chatgarment_data/garments_imgs_v2_3}"

DINOV2_CACHE="${TORCH_HOME}/hub/facebookresearch_dinov2_main"
if [[ ! -f "${DINOV2_CACHE}/hubconf.py" ]]; then
  echo "cached DINOv2 repository not found: ${DINOV2_CACHE}" >&2
  exit 1
fi

echo "host=$(hostname)"
echo "date=$(date)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "architecture=grouped"
echo "mode=${MODE}"
echo "prepared_dir=${PREPARED_DIR}"
echo "out_dir=${OUT_DIR}"
echo "reg_loss=${REG_LOSS}"
echo "cat_freeze=patience:${CAT_FREEZE_PATIENCE},min_epochs:${CAT_FREEZE_MIN_EPOCHS},min_delta:${CAT_FREEZE_MIN_DELTA}"

python - <<'PY'
import torch
import torchvision

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm allocation")
print("gpu", torch.cuda.get_device_name(0))
PY

srun python -u train_dinov2_grouped.py \
  --prepared-dir "${PREPARED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --backbone "${BACKBONE}" \
  --mode "${MODE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --lr "${LR}" \
  --backbone-lr-scale "${BACKBONE_LR_SCALE}" \
  --lambda-cat "${LAMBDA_CAT}" \
  --label-smoothing "${LABEL_SMOOTHING}" \
  --reg-loss "${REG_LOSS}" \
  --huber-delta "${HUBER_DELTA}" \
  --grad-clip "${GRAD_CLIP}" \
  --cat-freeze-patience "${CAT_FREEZE_PATIENCE}" \
  --cat-freeze-min-epochs "${CAT_FREEZE_MIN_EPOCHS}" \
  --cat-freeze-min-delta "${CAT_FREEZE_MIN_DELTA}" \
  --device cuda \
  --amp \
  --image-path-prefix "${OLD_IMAGE_PREFIX}" "${NEW_IMAGE_PREFIX}"
