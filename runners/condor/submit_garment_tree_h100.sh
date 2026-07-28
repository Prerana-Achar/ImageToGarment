#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: bash runners/condor/submit_garment_tree_h100.sh <run_name> [bid]"
  exit 1
fi

run_name="$1"
bid="${2:-100}"
log_dir="$(date '+%Y-%m-%d_%H-%M-%S')_${run_name}"
mkdir -p "runners/condor/logs/${log_dir}"

condor_submit_bid "${bid}" runners/condor/train_garment_tree_h100.sub \
  log_dir="${log_dir}"

