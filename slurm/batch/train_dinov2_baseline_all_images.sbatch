#!/bin/bash
#SBATCH --job-name=dino-base-allimg
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
OUT_DIR="${OUT_DIR:-runs/dinov2_baseline_all_images}"
BACKBONE="${BACKBONE:-dinov2_vits14}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
LAMBDA_CAT="${LAMBDA_CAT:-1.0}"

echo "host=$(hostname)"
echo "date=$(date)"
echo "architecture=baseline"
echo "mode=all_images"
echo "prepared_dir=${PREPARED_DIR}"
echo "out_dir=${OUT_DIR}"

python - <<'PY'
import torch
import torchvision

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm allocation")
print("gpu", torch.cuda.get_device_name(0))
PY

srun python -u dinov2_pipeline.py train \
  --architecture baseline \
  --prepared-dir "${PREPARED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --backbone "${BACKBONE}" \
  --mode all_images \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --lr "${LR}" \
  --lambda-cat "${LAMBDA_CAT}" \
  --device cuda \
  --amp
