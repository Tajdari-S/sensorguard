#!/usr/bin/env bash
set -euo pipefail

python_bin="$1"
verifier_url="$2"
node_url="$3"
output_dir="$4"
import_dir="$5"

mkdir -p "$import_dir" "$output_dir"
while true; do
  if curl --fail --silent --max-time 10 "$verifier_url/ready_verifier.json" >/dev/null \
     && curl --fail --silent --max-time 10 "$node_url/ready_node.json" >/dev/null; then
    break
  fi
  sleep 60
done

curl --fail --silent --show-error "$verifier_url/plan.json" -o "$import_dir/plan.json"
curl --fail --silent --show-error "$verifier_url/electrical_features.csv" -o "$import_dir/electrical_features.csv"
curl --fail --silent --show-error "$node_url/nvml_features.csv" -o "$import_dir/nvml_features.csv"

"$python_bin" scripts/analysis/post_collection_architecture.py \
  --plan "$import_dir/plan.json" \
  --nvml-features "$import_dir/nvml_features.csv" \
  --electrical-features "$import_dir/electrical_features.csv" \
  --development-window-predictions results/current_paper_physical_20260831/window_predictions.csv \
  --frozen-dir results/current_paper_physical_20260831 \
  --output-dir "$output_dir"
