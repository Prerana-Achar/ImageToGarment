#!/bin/bash
#SBATCH --job-name=dinov2-garment
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
OUT_DIR="${OUT_DIR:-runs/dinov2_vits14}"
BACKBONE="${BACKBONE:-dinov2_vits14}"
MODE="${MODE:-single}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
LAMBDA_CAT="${LAMBDA_CAT:-1.0}"

# Kept for the current prepared_v2 manifest, which may contain absolute paths
# from a local Mac. If you regenerate prepared_v2 on the cluster, this rewrite is
# simply a no-op.
OLD_IMAGE_PREFIX="${OLD_IMAGE_PREFIX:-/Users/siddharth/Study/3dv_project/data/chatgarment_data/garments_imgs_v2_3}"
NEW_IMAGE_PREFIX="${NEW_IMAGE_PREFIX:-${PROJECT_DIR}/data/chatgarment_data/garments_imgs_v2_3}"

echo "host=$(hostname)"
echo "date=$(date)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "prepared_dir=${PREPARED_DIR}"
echo "out_dir=${OUT_DIR}"

python - <<'PY'
import torch
import torchvision
import numpy
import PIL

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("PIL", PIL.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

srun python train_dinov2.py \
  --prepared-dir "${PREPARED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --backbone "${BACKBONE}" \
  --mode "${MODE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --lr "${LR}" \
  --lambda-cat "${LAMBDA_CAT}" \
  --device cuda \
  --amp \
  --image-path-prefix "${OLD_IMAGE_PREFIX}" "${NEW_IMAGE_PREFIX}"
