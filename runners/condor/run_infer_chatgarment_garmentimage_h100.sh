#!/bin/bash
set -eu
PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
RUN_NAME="${RUN_NAME:-smplx_vlm_lora}"
IMAGE_ROOT="${IMAGE_ROOT:-/is/cluster/fast/pachar/Data/GarmentImage}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/ChatGarment/ChatGarmentResults}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/runs/chatgarment_smplx/${RUN_NAME}/ckpt_model}"
CONVERTED_CKPT="${CONVERTED_CKPT:-${PROJECT_DIR}/runs/chatgarment_smplx/${RUN_NAME}/pytorch_model.bin}"
HF_HOME="${HF_HOME:-/is/cluster/fast/pachar/Data/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-/is/cluster/fast/pachar/Data/.cache/torch}"

if [ ! -x "${VENV}/bin/python" ]; then echo "ERROR: venv missing: ${VENV}" >&2; exit 2; fi
if [ ! -d "${IMAGE_ROOT}" ]; then echo "ERROR: image root missing: ${IMAGE_ROOT}" >&2; exit 3; fi
if [ ! -d "${CKPT_DIR}" ]; then echo "ERROR: DeepSpeed checkpoint dir missing: ${CKPT_DIR}" >&2; exit 4; fi

mkdir -p "$(dirname "${CONVERTED_CKPT}")" "${RESULTS_DIR}" "${HF_HOME}" "${TORCH_HOME}/kernels"
export HF_HOME TORCH_HOME
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${TORCH_HOME}/kernels}"
export HOME="${HOME:-${PROJECT_DIR}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
echo "Host: $(hostname)"
echo "Run: ${RUN_NAME}"
echo "Images: ${IMAGE_ROOT}"
echo "Results: ${RESULTS_DIR}"
nvidia-smi
"${VENV}/bin/python" -c "import torch; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available())"

if [ ! -s "${CONVERTED_CKPT}" ]; then
  echo "Converting DeepSpeed checkpoint: ${CKPT_DIR} -> ${CONVERTED_CKPT}"
  "${VENV}/bin/python" "${CKPT_DIR}/zero_to_fp32.py" "${CKPT_DIR}" "${CONVERTED_CKPT}"
else
  echo "Using existing converted checkpoint: ${CONVERTED_CKPT}"
fi

cd "${PROJECT_DIR}/ChatGarment"
export CHATGARMENT_RESUME_PATH="${CONVERTED_CKPT}"
export CHATGARMENT_LOG_BASE_DIR="${RESULTS_DIR}"
export CHATGARMENT_EXP_NAME="${RUN_NAME}"

for DIR in "${IMAGE_ROOT}"/*; do
  [ -d "${DIR}" ] || continue
  echo "Running inference on ${DIR}"
  "${VENV}/bin/python" scripts/evaluate_garment_v2_imggen_1float.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path_eval "${DIR}" \
    --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --model_max_length 3072 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1
done

echo "Done. Results are under ${RESULTS_DIR}/${RUN_NAME}/"