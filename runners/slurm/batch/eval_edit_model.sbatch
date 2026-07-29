#!/bin/bash
#SBATCH --job-name=edit-eval
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

# Run two-pass inference for BOTH trained variants on the 134-pair edit
# benchmark, then print the head-to-head table. Submit after both training jobs
# have written a checkpoint:
#
#   sbatch slurm/batch/eval_edit_model.sbatch
#
# Two passes per variant:
#   * teacher-forced -- heads read off the reference target text. Isolates the
#     float *readout* (the actual A/B question) from JSON-generation quality;
#     both variants are scored on identical ground-truth structure.
#   * generated      -- full two-pass inference. Adds parse rate and
#                       discrete-field accuracy, i.e. the end-to-end system.

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

BACKBONE="${BACKBONE:-Qwen/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-prepared_edit}"
CKPT_A="${CKPT_A:-runs/edit_single_token/final}"
CKPT_B="${CKPT_B:-runs/edit_per_token/final}"
OUT_DIR="${OUT_DIR:-runs/edit_eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"

for c in "${CKPT_A}" "${CKPT_B}"; do
  if [[ ! -f "${c}/heads.pt" ]]; then
    echo "checkpoint missing (no heads.pt): ${c}" >&2
    exit 1
  fi
done

mkdir -p slurm/logs "${OUT_DIR}"

echo "host=$(hostname)"; echo "date=$(date)"
echo "ckpt_A(single_token)=${CKPT_A}"
echo "ckpt_B(per_token)=${CKPT_B}"

run_infer () {  # $1=ckpt  $2=tag  $3=extra flag
  srun python -u infer_edit.py \
    --ckpt "$1" \
    --backbone "${BACKBONE}" \
    --data "${DATA_DIR}/edit_eval.jsonl" \
    --schema "${DATA_DIR}/schema_edit.json" \
    --out "${OUT_DIR}/preds_$2.jsonl" \
    --max-new-tokens "${MAX_NEW_TOKENS}" $3
}

echo "=== teacher-forced (isolates the float readout) ==="
run_infer "${CKPT_A}" "single_tf" "--teacher-forced"
run_infer "${CKPT_B}" "per_tf"    "--teacher-forced"

echo "=== generated (end-to-end two-pass) ==="
run_infer "${CKPT_A}" "single_gen" ""
run_infer "${CKPT_B}" "per_gen"    ""

echo
echo "################ A/B: teacher-forced readout ################"
python -u eval_edit.py \
  --schema "${DATA_DIR}/schema_edit.json" \
  --ref "${DATA_DIR}/edit_eval.jsonl" \
  --preds "single_token:${OUT_DIR}/preds_single_tf.jsonl" \
          "per_token:${OUT_DIR}/preds_per_tf.jsonl" \
  --out "${OUT_DIR}/metrics_teacher_forced.json"

echo
echo "################ A/B: generated (end-to-end) ################"
python -u eval_edit.py \
  --schema "${DATA_DIR}/schema_edit.json" \
  --ref "${DATA_DIR}/edit_eval.jsonl" \
  --preds "single_token:${OUT_DIR}/preds_single_gen.jsonl" \
          "per_token:${OUT_DIR}/preds_per_gen.jsonl" \
  --out "${OUT_DIR}/metrics_generated.json"

echo "[done] metrics in ${OUT_DIR}"
