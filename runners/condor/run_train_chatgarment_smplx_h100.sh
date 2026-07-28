#!/bin/bash
set -eu
PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_DIR="${DATA_DIR:-/is/cluster/fast/pachar/Data/ChatGarmentSMPLXVLM}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/chatgarment_smplx}"
RUN_NAME="${RUN_NAME:-chatgarment_smplx_lora}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/${RUN_NAME}}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
HF_HOME="${HF_HOME:-/is/cluster/fast/pachar/Data/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-/is/cluster/fast/pachar/Data/.cache/torch}"
EPOCHS="${EPOCHS:-20}"; BATCH_SIZE="${BATCH_SIZE:-2}"; EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"; NUM_WORKERS="${NUM_WORKERS:-8}"; LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"; LORA_R="${LORA_R:-64}"; LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
WANDB_PROJECT="${WANDB_PROJECT:-ImageToGarment}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
if [ -f "${PROJECT_DIR}/runners/condor/secrets.sh" ]; then
  . "${PROJECT_DIR}/runners/condor/secrets.sh"
fi
if [ ! -x "${VENV}/bin/python" ]; then echo "ERROR: venv missing: ${VENV}" >&2; exit 2; fi
if [ ! -s "${DATA_DIR}/train.json" ] || [ ! -s "${DATA_DIR}/val.json" ]; then
  echo "ERROR: run bash scripts/prepare_chatgarment_smplx_vlm.sh first" >&2; exit 3
fi
"${VENV}/bin/python" -c "import json,sys; r=json.load(open('${DATA_DIR}/report.json')); sys.exit(0 if r.get('ready') else 'Dataset report is not training-ready')"
mkdir -p "${OUT_DIR}" "${HF_HOME}" "${TORCH_HOME}/kernels"
export HF_HOME TORCH_HOME WANDB_PROJECT WANDB_RUN_NAME WANDB_MODE
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${TORCH_HOME}/kernels}"
export HOME="${HOME:-${PROJECT_DIR}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_WORKERS}}"
cd "${PROJECT_DIR}/ChatGarment"
echo "Host: $(hostname)"; echo "Data: ${DATA_DIR}"; echo "Output: ${OUT_DIR}"
nvidia-smi
"${VENV}/bin/python" -c "import torch; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available())"
"${VENV}/bin/deepspeed" llava/train/train_mem_garmentcode_outfit.py   --lora_enable True --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}"   --mm_projector_lr 2e-5 --deepspeed ./scripts/zero2.json   --model_name_or_path liuhaotian/llava-v1.5-7b --version v1   --data_path "${DATA_DIR}/train.json" --data_path_eval "${DATA_DIR}/val.json" --image_folder /   --vision_tower openai/clip-vit-large-patch14-336 --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2   --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad   --group_by_modality_length True --bf16 True --tf32 True --output_dir "${OUT_DIR}"   --num_train_epochs "${EPOCHS}" --per_device_train_batch_size "${BATCH_SIZE}"   --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" --gradient_accumulation_steps "${GRAD_ACCUM}"   --learning_rate "${LR}" --weight_decay "${WEIGHT_DECAY}" --warmup_ratio 0.03   --logging_steps 1 --model_max_length 3072 --gradient_checkpointing True   --dataloader_num_workers "${NUM_WORKERS}" --lazy_preprocess True --report_to wandb
