#!/bin/bash
set -eu
PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATASET_ROOT="${DATASET_ROOT:-/is/cluster/fast/pachar/Data/GarmentCodeSMPLX}"
OUT_DIR="${OUT_DIR:-/is/cluster/fast/pachar/Data/ChatGarmentSMPLXVLM}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
if [ ! -x "${VENV}/bin/python" ]; then echo "ERROR: venv missing: ${VENV}" >&2; exit 2; fi
cd "${PROJECT_DIR}"
PATCH="${PROJECT_DIR}/patches/chatgarment_smplx_vlm.patch"
TRAINER="${PROJECT_DIR}/ChatGarment/llava/train/train_garmentcode_outfit.py"
FLOAT_MODEL="${PROJECT_DIR}/ChatGarment/llava/model/language_model/llava_garment_float50.py"
if ! grep -q "sample_randomly=True" "${TRAINER}" || ! grep -q "clamp_min(1.0)" "${FLOAT_MODEL}"; then
  git -C "${PROJECT_DIR}/ChatGarment" apply "${PATCH}"
  echo "Applied ChatGarment SMPLX trainer patch."
fi
"${VENV}/bin/python" scripts/prepare_chatgarment_smplx_vlm.py --dataset-root "${DATASET_ROOT}" --out "${OUT_DIR}" "$@"
