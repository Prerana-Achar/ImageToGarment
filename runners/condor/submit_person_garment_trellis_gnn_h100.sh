#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: bash runners/condor/submit_person_garment_trellis_gnn_h100.sh <image_path> [bid] [checkpoint] [gnn_epochs] [gnn_surface_samples] [min_root_conf] [min_cat_conf] [min_num_prob] [max_num_uncertainty]" >&2
  exit 1
fi

image_path="$1"
bid="${2:-200}"
checkpoint="${3:-}"
gnn_epochs="${4:-200}"
gnn_surface_samples="${5:-8192}"
min_root_confidence="${6:-0.75}"
min_active_categorical_confidence="${7:-0.65}"
min_active_numeric_probability="${8:-0.55}"
max_active_numeric_uncertainty="${9:-0.35}"
run_name="person_garment_trellis_gnn"
log_dir="$(date '+%Y-%m-%d_%H-%M-%S')_${run_name}"

mkdir -p "runners/condor/logs/${log_dir}"

condor_submit_bid "${bid}" runners/condor/person_garment_trellis_gnn_h100.sub \
  log_dir="${log_dir}" \
  image_path="${image_path}" \
  checkpoint="${checkpoint}" \
  gnn_epochs="${gnn_epochs}" \
  gnn_surface_samples="${gnn_surface_samples}" \
  min_root_confidence="${min_root_confidence}" \
  min_active_categorical_confidence="${min_active_categorical_confidence}" \
  min_active_numeric_probability="${min_active_numeric_probability}" \
  max_active_numeric_uncertainty="${max_active_numeric_uncertainty}"

echo "Submitted person->garment->TRELLIS->GarmentCode->GNN pipeline with bid ${bid}"
echo "Logs: runners/condor/logs/${log_dir}"
echo "Confidence gate: root>=${min_root_confidence}, categorical>=${min_active_categorical_confidence}, numeric_prob>=${min_active_numeric_probability}, numeric_uncertainty<=${max_active_numeric_uncertainty}"