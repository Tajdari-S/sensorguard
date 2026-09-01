#!/usr/bin/env bash
set -euo pipefail

role="$1"
plan="$2"
root="$3"
output="$4"
python_bin="$5"

expected="$($python_bin -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["runs"]))' "$plan")"
status="$root/status_${role}.csv"
while true; do
  if [[ -f "$status" ]]; then
    completed="$(($(wc -l < "$status") - 1))"
    if [[ "$completed" -ge "$expected" ]]; then
      break
    fi
  fi
  sleep 30
done

"$python_bin" scripts/analysis/extract_hard_family_features.py \
  --role "$role" --plan "$plan" --root "$root" --output "$output"
