#!/bin/bash
set -euo pipefail
if [ -z "${1:-}" ]; then
  echo "Usage: bash runners/condor/submit_garment_tree_ensemble_garmentimage_h100.sh <checkpoint> [run_name] [bid] [image_root] [limit]"
  echo "Example: bash runners/condor/submit_garment_tree_ensemble_garmentimage_h100.sh /is/cluster/pachar/Projects/ImageToGarment/runs/.../best.pt pose_v6_garmentimage 50 /is/cluster/fast/pachar/Data/GarmentImage 12"
  exit 1
fi
PROJECT_DIR="/is/cluster/pachar/Projects/ImageToGarment"
CHECKPOINT="$1"
RUN_NAME="${2:-garment_tree_ensemble_garmentimage}"
BID="${3:-50}"
IMAGE_ROOT="${4:-/is/cluster/fast/pachar/Data/GarmentImage}"
LIMIT="${5:-}"
LOG_DIR="$(date '+%Y-%m-%d_%H-%M-%S')_${RUN_NAME}"
OUT_DIR="${PROJECT_DIR}/runs/garmentimage_comparisons/${LOG_DIR}"
ABS_LOG_DIR="${PROJECT_DIR}/runners/condor/logs/${LOG_DIR}"
cd "${PROJECT_DIR}"
mkdir -p "${ABS_LOG_DIR}"
condor_submit_bid "${BID}" runners/condor/infer_garment_tree_ensemble_garmentimage_h100.sub \
  log_dir="${LOG_DIR}" checkpoint="${CHECKPOINT}" image_root="${IMAGE_ROOT}" out_dir="${OUT_DIR}" limit="${LIMIT}"
echo "Submitted GarmentImage ensemble inference with bid ${BID}"
echo "Logs: ${ABS_LOG_DIR}"
echo "  event log: ${ABS_LOG_DIR}/job.log"
echo "  stdout:    ${ABS_LOG_DIR}/job.out"
echo "  stderr:    ${ABS_LOG_DIR}/job.err"
echo "Output: ${OUT_DIR}"