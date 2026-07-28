#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
CHAT_PREPARED_DIR="${CHAT_PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_all}"
TARGET_PREPARED_DIR="${TARGET_PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_smplx_output_balanced}"
CHAT_DESIGN_SCHEMA="${CHAT_DESIGN_SCHEMA:-${PROJECT_DIR}/GarmentCodeRC/assets/design_params/design_used.yaml}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/ImageToGarment/prepared_chatgarment_full_current}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
MINIMUM_MAPPED_FRACTION="${MINIMUM_MAPPED_FRACTION:-0.90}"
ALLOW_UNSEEN_ROUTES="${ALLOW_UNSEEN_ROUTES:-1}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"
EXTRA_ARGS=()
if [ "${ALLOW_UNSEEN_ROUTES}" = "1" ]; then
  EXTRA_ARGS+=(--allow-unseen-routes)
fi
python -u compile_chatgarment_auxiliary.py \
  --chat-prepared-dir "${CHAT_PREPARED_DIR}" \
  --target-prepared-dir "${TARGET_PREPARED_DIR}" \
  --chat-design-schema "${CHAT_DESIGN_SCHEMA}" \
  --out-dir "${OUT_DIR}" \
  --minimum-mapped-fraction "${MINIMUM_MAPPED_FRACTION}" \
  --numeric-select-tolerance 1e-4 \
  "${EXTRA_ARGS[@]}"
cat "${OUT_DIR}/compatibility_report.json"