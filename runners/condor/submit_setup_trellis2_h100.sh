#!/bin/bash
set -euo pipefail

bid="${1:-200}"
run_name="trellis2_setup"
log_dir="$(date '+%Y-%m-%d_%H-%M-%S')_${run_name}"

mkdir -p "runners/condor/logs/${log_dir}"

condor_submit_bid "${bid}" runners/condor/setup_trellis2_h100.sub \
  log_dir="${log_dir}"