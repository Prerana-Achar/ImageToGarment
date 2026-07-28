#!/bin/bash

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATASET_ROOT="${DATASET_ROOT:-/is/cluster/fast/pachar/Data/GarmentCodeSMPLX}"
PREPARED_DIR="${PREPARED_DIR:-/is/cluster/fast/pachar/Data/ImageToGarment/prepared_smplx}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
SEED="${SEED:-42}"
VAL_FRACTION_OF_TRAIN_BODIES="${VAL_FRACTION_OF_TRAIN_BODIES:-0.1}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-1}"
TEST_FRACTION_OF_VAL_BODIES="${TEST_FRACTION_OF_VAL_BODIES:-}"

cd "${PROJECT_DIR}" || exit 2
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV}" >&2
  exit 2
fi
. "${VENV}/bin/activate"

EXTRA_ARGS=()
if [ "${ALLOW_INCOMPLETE}" = "1" ]; then
  EXTRA_ARGS+=(--allow-incomplete)
fi
if [ -n "${TEST_FRACTION_OF_VAL_BODIES}" ]; then
  EXTRA_ARGS+=(--test-fraction-of-val-bodies "${TEST_FRACTION_OF_VAL_BODIES}")
fi

python -u "${PROJECT_DIR}/prepare_garmentcode_smplx.py" \
  --dataset-root "${DATASET_ROOT}" \
  --out "${PREPARED_DIR}" \
  --schema "${PROJECT_DIR}/GarmentCodeRC/assets/design_params/default_new.yaml" \
  --val-fraction-of-train-bodies "${VAL_FRACTION_OF_TRAIN_BODIES}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
