#!/bin/bash
set -eu
if [ -z "${1:-}" ]; then echo "Usage: bash runners/condor/submit_chatgarment_smplx_h100.sh <run_name> [bid]"; exit 1; fi
run_name="$1"; bid="${2:-150}"
log_dir="$(date '+%Y-%m-%d_%H-%M-%S')_${run_name}"
mkdir -p "runners/condor/logs/${log_dir}"
condor_submit_bid "${bid}" runners/condor/train_chatgarment_smplx_h100.sub log_dir="${log_dir}" run_name="${run_name}"
