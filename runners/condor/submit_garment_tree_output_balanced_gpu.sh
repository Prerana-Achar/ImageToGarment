#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: bash runners/condor/submit_garment_tree_output_balanced_gpu.sh <run_name> [bid]"
  exit 1
fi

PROJECT_DIR="/is/cluster/pachar/Projects/ImageToGarment"
RUN_NAME="$1"
BID="${2:-50}"
LOG_DIR="$(date '+%Y-%m-%d_%H-%M-%S')_${RUN_NAME}"
ABS_LOG_DIR="${PROJECT_DIR}/runners/condor/logs/${LOG_DIR}"

cd "${PROJECT_DIR}"
mkdir -p "${ABS_LOG_DIR}"
condor_submit_bid "${BID}" runners/condor/train_garment_tree_output_balanced_gpu.sub \
  log_dir="${LOG_DIR}"

echo "Submitted ${RUN_NAME} with one cluster-assigned GPU and bid ${BID}"
echo "Logs: ${ABS_LOG_DIR}"
echo "  event log: ${ABS_LOG_DIR}/job.log"
echo "  stdout:    ${ABS_LOG_DIR}/job.out"
echo "  stderr:    ${ABS_LOG_DIR}/job.err"
echo "Output: ${PROJECT_DIR}/runs/${LOG_DIR}_garment_tree_output_balanced"