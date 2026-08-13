#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-results/roofline/ncu}"
gpu="${2:-0}"
mkdir -p "$out_dir/benchmarks"

command -v ncu >/dev/null 2>&1 || { echo "ERROR: ncu is not on PATH" >&2; exit 2; }
python3 -c 'import torch; assert torch.cuda.is_available()' || {
  echo "ERROR: CUDA PyTorch is unavailable" >&2; exit 2;
}

cases=(gemm_1024 gemm_2048 gemm_4096 gemm_tall gemv copy elementwise)
for case_name in "${cases[@]}"; do
  echo "Profiling ${case_name} on cuda:${gpu}"
  ncu --target-processes all \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --csv --log-file "$out_dir/${case_name}.ncu.csv" \
    python3 scripts/roofline/benchmark_kernels.py \
      --case "$case_name" --device "cuda:${gpu}" --warmup 3 --iterations 5 \
      --output "$out_dir/benchmarks/${case_name}.json"
done
