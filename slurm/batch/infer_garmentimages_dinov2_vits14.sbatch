#!/bin/bash
#SBATCH --job-name=infer-gi-vits14
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

python scripts/infer_garmentimages_dinov2_vits14.py \
  --image-root GarmentImages \
  --out-dir runs/garmentimage_comparisons/dinov2_vits14 \
  --device cuda \
  --skip-existing
