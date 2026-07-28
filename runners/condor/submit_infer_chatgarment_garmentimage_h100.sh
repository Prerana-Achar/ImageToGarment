#!/bin/bash
set -eu
if [ -z "${1:-}" ]; then
  echo "Usage: bash runners/condor/submit_infer_chatgarment_garmentimage_h100.sh <run_name> [image_root] [bid]"
  echo "  e.g. bash runners/condor/submit_infer_chatgarment_garmentimage_h100.sh smplx_vlm_lora /is/cluster/fast/pachar/Data/GarmentImage 150"
  exit 1
fi
run_name="$1"
image_root="${2:-/is/cluster/fast/pachar/Data/GarmentImage}"
bid="${3:-150}"
log_dir="$(date '+%Y-%m-%d_%H-%M-%S')_${run_name}_infer_garmentimage"
mkdir -p "runners/condor/logs/${log_dir}"
condor_submit_bid "${bid}" runners/condor/infer_chatgarment_garmentimage_h100.sub \
  log_dir="${log_dir}" run_name="${run_name}" image_root="${image_root}"