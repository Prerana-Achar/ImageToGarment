#!/bin/bash

# Fetch all ChatGarment dataset versions supported by prepare_data.py into
# /is/cluster/fast/pachar/Data and build a combined prepared dataset.
#
# Run on the cluster:
#   cd /is/cluster/pachar/Projects/ImageToGarment
#   bash scripts/fetch_all_fast_data.sh
#
# Outputs:
#   Raw files: /is/cluster/fast/pachar/Data/ChatGarmentDataset
#   Prepared:  /is/cluster/fast/pachar/Data/ImageToGarment/prepared_all
#
# Version toggles:
#   INCLUDE_V1=1 INCLUDE_V2=1 INCLUDE_V3=1 INCLUDE_V4=1
#   INCLUDE_V1=0 bash scripts/fetch_all_fast_data.sh   # skip huge rest-pose v1

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/ChatGarmentDataset}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_all}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
REPO_ID="${REPO_ID:-sy000/ChatGarmentDataset}"
FRAMES="${FRAMES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
SEED="${SEED:-42}"

INCLUDE_V1="${INCLUDE_V1:-1}"
INCLUDE_V2="${INCLUDE_V2:-1}"
INCLUDE_V3="${INCLUDE_V3:-1}"
INCLUDE_V4="${INCLUDE_V4:-1}"

V1_JSON="${V1_JSON:-training/synthetic/data_restpose_img_v1.json}"
V2_JSON="${V2_JSON:-training/synthetic/data_img_v2.json}"
V3_JSON="${V3_JSON:-training/synthetic/data_img_v3.json}"
V4_JSON="${V4_JSON:-training/synthetic/data_img_v4.json}"

V1_SHARDS="${V1_SHARDS:-garments_imgs_v1_1.zip garments_imgs_v1_2.zip garments_imgs_v1_3.zip garments_imgs_v1_4.zip garments_imgs_v1_5.zip}"
V2_SHARDS="${V2_SHARDS:-garments_imgs_v2_1.zip garments_imgs_v2_2.zip garments_imgs_v2_3.zip}"
V3_SHARDS="${V3_SHARDS:-garments_imgs_v3.zip}"
V4_SHARDS="${V4_SHARDS:-garments_imgs_v4.zip}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 2
fi

cd "${PROJECT_DIR}" || exit 2

if [ -d "${VENV}" ]; then
  . "${VENV}/bin/activate"
else
  echo "ERROR: venv not found at ${VENV}" >&2
  exit 2
fi

mkdir -p "${DATASET_DIR}"
mkdir -p "${PREPARED_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found; installing huggingface_hub into the active venv..."
  python -m pip install -U huggingface_hub
fi

download_file() {
  relpath="$1"
  if [ -f "${DATASET_DIR}/${relpath}" ]; then
    echo "Already downloaded: ${relpath}"
    return 0
  fi
  echo "Downloading: ${relpath}"
  huggingface-cli download "${REPO_ID}" "${relpath}" \
    --repo-type dataset \
    --local-dir "${DATASET_DIR}" \
    --local-dir-use-symlinks False
}

extract_shard() {
  shard="$1"
  shard_root="${DATASET_DIR}/${shard%.zip}"
  mkdir -p "${shard_root}"

  marker="${shard_root}/.extract_complete"
  if [ ! -f "${marker}" ]; then
    echo "Extracting ${shard} -> ${shard_root}"
    unzip -q -n "${DATASET_DIR}/${shard}" -d "${shard_root}"
    touch "${marker}"
  else
    echo "Already extracted: ${shard_root}"
  fi

  nested_root="${shard_root}/${shard%.zip}"
  if [ -d "${nested_root}" ]; then
    shard_root="${nested_root}"
  fi
  echo "${shard_root}"
}

prepare_set_args=""

add_version() {
  tag="$1"
  json_rel="$2"
  shards="$3"

  download_file "${json_rel}"
  json_path="${DATASET_DIR}/${json_rel}"
  if [ ! -f "${json_path}" ]; then
    echo "ERROR: missing JSON after download: ${json_path}" >&2
    exit 3
  fi

  roots=""
  for shard in ${shards}; do
    download_file "${shard}"
    if [ ! -f "${DATASET_DIR}/${shard}" ]; then
      echo "ERROR: missing shard after download: ${DATASET_DIR}/${shard}" >&2
      exit 3
    fi
    root_path="$(extract_shard "${shard}")"
    roots="${roots} ${root_path}"
  done

  prepare_set_args="${prepare_set_args} --set ${tag} ${json_path}${roots}"
}

echo "Project:      ${PROJECT_DIR}"
echo "Raw dataset:  ${DATASET_DIR}"
echo "Prepared out: ${PREPARED_DIR}"
echo "Frames:       ${FRAMES}"

if [ "${INCLUDE_V1}" = "1" ]; then
  add_version "v1" "${V1_JSON}" "${V1_SHARDS}"
fi
if [ "${INCLUDE_V2}" = "1" ]; then
  add_version "v2" "${V2_JSON}" "${V2_SHARDS}"
fi
if [ "${INCLUDE_V3}" = "1" ]; then
  add_version "v3" "${V3_JSON}" "${V3_SHARDS}"
fi
if [ "${INCLUDE_V4}" = "1" ]; then
  add_version "v4" "${V4_JSON}" "${V4_SHARDS}"
fi

if [ -z "${prepare_set_args}" ]; then
  echo "ERROR: no dataset versions selected" >&2
  exit 4
fi

echo "Preparing combined dataset manifest..."
# shellcheck disable=SC2086
python "${PROJECT_DIR}/prepare_data.py" \
  ${prepare_set_args} \
  --out "${PREPARED_DIR}" \
  --frames ${FRAMES} \
  --val "${VAL_FRACTION}" \
  --test "${TEST_FRACTION}" \
  --seed "${SEED}"

echo
echo "Done. Combined prepared data:"
echo "  ${PREPARED_DIR}"