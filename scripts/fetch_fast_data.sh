#!/bin/bash

# Fetch the ChatGarment training data into the shared fast storage and prepare
# the compact training manifest used by dinov2_pipeline.py.
#
# Run on the cluster:
#   cd /is/cluster/pachar/Projects/ImageToGarment
#   bash scripts/fetch_fast_data.sh
#
# Useful overrides:
#   DATA_ROOT=/is/cluster/fast/pachar/Data
#   V2_SHARDS="garments_imgs_v2_1.zip garments_imgs_v2_2.zip garments_imgs_v2_3.zip"
#   FRAMES=all

PROJECT_DIR="${PROJECT_DIR:-/is/cluster/pachar/Projects/ImageToGarment}"
DATA_ROOT="${DATA_ROOT:-/is/cluster/fast/pachar/Data}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/ChatGarmentDataset}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_ROOT}/ImageToGarment/prepared_v2}"
VENV="${VENV:-${PROJECT_DIR}/venv}"
REPO_ID="${REPO_ID:-sy000/ChatGarmentDataset}"
V2_JSON="${V2_JSON:-training/synthetic/data_img_v2.json}"
V2_SHARDS="${V2_SHARDS:-garments_imgs_v2_1.zip garments_imgs_v2_2.zip garments_imgs_v2_3.zip}"
FRAMES="${FRAMES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
SEED="${SEED:-42}"

echo "Project:      ${PROJECT_DIR}"
echo "Data root:    ${DATA_ROOT}"
echo "Raw dataset:  ${DATASET_DIR}"
echo "Prepared out: ${PREPARED_DIR}"

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

echo "Downloading metadata JSON..."
huggingface-cli download "${REPO_ID}" "${V2_JSON}" \
  --repo-type dataset \
  --local-dir "${DATASET_DIR}" \
  --local-dir-use-symlinks False

IMAGE_ROOTS=""
for shard in ${V2_SHARDS}; do
  echo "Downloading image shard: ${shard}"
  huggingface-cli download "${REPO_ID}" "${shard}" \
    --repo-type dataset \
    --local-dir "${DATASET_DIR}" \
    --local-dir-use-symlinks False

  shard_root="${DATASET_DIR}/${shard%.zip}"
  if [ ! -d "${shard_root}" ]; then
    mkdir -p "${shard_root}"
  fi

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

  IMAGE_ROOTS="${IMAGE_ROOTS} ${shard_root}"
done

JSON_PATH="${DATASET_DIR}/${V2_JSON}"
if [ ! -f "${JSON_PATH}" ]; then
  echo "ERROR: expected metadata JSON is missing: ${JSON_PATH}" >&2
  exit 3
fi

echo "Preparing compact dataset manifest..."
python "${PROJECT_DIR}/prepare_data.py" \
  --set v2 "${JSON_PATH}" ${IMAGE_ROOTS} \
  --out "${PREPARED_DIR}" \
  --frames ${FRAMES} \
  --val "${VAL_FRACTION}" \
  --test "${TEST_FRACTION}" \
  --seed "${SEED}"

echo
echo "Done."
echo "Prepared data:"
echo "  ${PREPARED_DIR}"
