#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATASET_ROOT="${DATASET_ROOT:-/is/cluster/fast/pachar/Data/GarmentCodeSMPLX}"
PREPARED_DIR="${PREPARED_DIR:-/is/cluster/fast/pachar/Data/ImageToGarment/prepared_smplx_balanced}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
VAL_BODY_FRACTION="${VAL_BODY_FRACTION:-0.15}"
NUMERIC_BINS="${NUMERIC_BINS:-10}"
SEED="${SEED:-42}"

cd "${PROJECT_DIR}"
. "${VENV}/bin/activate"

python -u "${PROJECT_DIR}/compile_balanced_garmentcode_smplx.py" \
  --dataset-root "${DATASET_ROOT}" \
  --out "${PREPARED_DIR}" \
  --schema "${PROJECT_DIR}/GarmentCodeRC/assets/design_params/default_new.yaml" \
  --val-body-fraction "${VAL_BODY_FRACTION}" \
  --numeric-bins "${NUMERIC_BINS}" \
  --seed "${SEED}"

