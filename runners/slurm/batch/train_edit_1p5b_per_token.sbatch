#!/bin/bash
#SBATCH --job-name=edit-1p5b-per
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

# Garment editor, variant B (per_token): one <VAL> token per active float, a
# shared MLP regresses one scalar from each <VAL> hidden state.
# Backbone: Qwen2.5-1.5B-Instruct, 4-bit QLoRA. Ready to submit as-is:
#
#   sbatch slurm/batch/train_edit_1p5b_per_token.sbatch
#
# Pairs with train_edit_1p5b_single_token.sbatch -- identical in every respect
# except the readout variant, which is what makes the A/B comparison valid.
# If you change a hyperparameter here, change it there too.

set -euo pipefail

PROJECT_DIR="/mnt/beegfs/home/stud136/3dv_project"
CONDA_SH="/home/stud136/miniconda3/etc/profile.d/conda.sh"

cd "${PROJECT_DIR}"
source "${CONDA_SH}"
conda activate project

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VARIANT="per_token"
BACKBONE="${BACKBONE:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-prepared_edit}"
OUT_DIR="${OUT_DIR:-runs/edit_1p5b_per_token}"

EPOCHS="${EPOCHS:-2}"
# 1.5B leaves more headroom than 3B, so batch 4 x accum 4 keeps the same
# effective batch (16) with fewer optimiser steps.
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
LAMBDA_NUM="${LAMBDA_NUM:-0.1}"
MAX_LEN="${MAX_LEN:-1500}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
HEAD_HIDDEN="${HEAD_HIDDEN:-512}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-20}"
VAL_EVERY="${VAL_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SEED="${SEED:-42}"

if [[ ! -f "${DATA_DIR}/schema_edit.json" ]]; then
  echo "prepared editing data not found in ${DATA_DIR}" >&2
  echo "run: python prepare_edit_data.py --out ${DATA_DIR}" >&2
  exit 1
fi

# HF_HUB_OFFLINE=1 turns a missing download into a confusing mid-run error, so
# check the weights are really cached. Prefetch from a login node with:
#   python -c "from huggingface_hub import snapshot_download as d; \
#              d('${BACKBONE}', allow_patterns=['*.safetensors','*.json','*.txt'])"
BACKBONE_CACHE="${HF_HOME}/hub/models--${BACKBONE//\//--}"
if ! ls "${BACKBONE_CACHE}"/snapshots/*/*.safetensors >/dev/null 2>&1; then
  echo "backbone weights not cached: ${BACKBONE_CACHE}" >&2
  echo "prefetch it from a login node before submitting (see comment above)" >&2
  exit 1
fi

mkdir -p slurm/logs "${OUT_DIR}"

echo "host=$(hostname)"
echo "date=$(date)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "variant=${VARIANT}"
echo "backbone=${BACKBONE}"
echo "out_dir=${OUT_DIR}"
echo "batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} max_len=${MAX_LEN}"
echo "lr=${LR} head_lr=${HEAD_LR} lambda_num=${LAMBDA_NUM}"

python - <<'PY'
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__,
      "bitsandbytes", bitsandbytes.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm allocation")
print("gpu", torch.cuda.get_device_name(0))
PY

srun python -u train_edit_model.py \
  --variant "${VARIANT}" \
  --backbone "${BACKBONE}" \
  --data-dir "${DATA_DIR}" \
  --out "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --head-lr "${HEAD_LR}" \
  --lambda-num "${LAMBDA_NUM}" \
  --max-len "${MAX_LEN}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --head-hidden "${HEAD_HIDDEN}" \
  --grad-clip "${GRAD_CLIP}" \
  --log-every "${LOG_EVERY}" \
  --val-every "${VAL_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --seed "${SEED}"
