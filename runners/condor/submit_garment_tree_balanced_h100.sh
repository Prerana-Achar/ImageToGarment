#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: bash runners/condor/submit_garment_tree_balanced_h100.sh <run_name> [bid]"
  echo "Example: bash runners/condor/submit_garment_tree_balanced_h100.sh garment_tree_v1 150"
  exit 1
fi

RUN_NAME="$1"
BID="${2:-100}"
LOG_DIR="$(date '+%Y-%m-%d_%H-%M-%S')_${RUN_NAME}"

cd /is/cluster/pachar/Projects/ImageToGarment
mkdir -p "runners/condor/logs/${LOG_DIR}"

condor_submit_bid "${BID}" runners/condor/train_garment_tree_balanced_h100.sub \
  log_dir="${LOG_DIR}"

echo "Submitted ${RUN_NAME} with H100 bid ${BID}"
echo "Logs: runners/condor/logs/${LOG_DIR}"
echo "Output: runs/${LOG_DIR}_garment_tree"

