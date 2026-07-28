#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: bash runners/condor/submit_h100.sh <run_name> [modelpy|baseline|grouped] [bid]"
  echo "  e.g. bash runners/condor/submit_h100.sh garment_grouped grouped"
  exit 1
fi

run_name="$1"
architecture="${2:-modelpy}"
bid="${3:-100}"

if [ "${architecture}" != "modelpy" ] && [ "${architecture}" != "baseline" ] && [ "${architecture}" != "grouped" ]; then
  echo "ERROR: architecture must be modelpy, baseline, or grouped, got: ${architecture}" >&2
  exit 2
fi

log_dir="$(date '+%Y-%m-%d_%H-%M-%S')_${run_name}"
mkdir -p "runners/condor/logs/${log_dir}"

condor_submit_bid "${bid}" runners/condor/train_dinov2_h100.sub \
  log_dir="${log_dir}" architecture="${architecture}"