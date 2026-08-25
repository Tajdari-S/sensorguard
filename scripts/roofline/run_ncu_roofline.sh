#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-results/roofline/ncu}"
gpu="${2:-0}"
mkdir -p "$out_dir/benchmarks" "$out_dir/profiled-timing-discard"

command -v ncu >/dev/null 2>&1 || { echo "ERROR: ncu is not on PATH" >&2; exit 2; }
python3 -c 'import torch; assert torch.cuda.is_available()' || {
  echo "ERROR: CUDA PyTorch is unavailable" >&2; exit 2;
}

cases=(gemm_1024 gemm_2048 gemm_4096 gemm_tall gemv copy elementwise)
for case_name in "${cases[@]}"; do
  # Timing comes from an UNPROFILED run: NCU replay serializes kernels and
  # poisons wall-clock, so the profiled process's tflops are meaningless.
  echo "Timing ${case_name} (unprofiled) on cuda:${gpu}"
  python3 scripts/roofline/benchmark_kernels.py \
    --case "$case_name" --device "cuda:${gpu}" --dtype float32 \
    --warmup 3 --iterations 5 \
    --output "$out_dir/benchmarks/${case_name}.json"
  # Counter pass: request the raw DRAM byte counters explicitly — section
  # CSVs export only human-named summary metrics, not dram__bytes_*.sum.
  echo "Profiling ${case_name} on cuda:${gpu}"
  ncu --target-processes all \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv --log-file "$out_dir/${case_name}.ncu.csv" \
    python3 scripts/roofline/benchmark_kernels.py \
      --case "$case_name" --device "cuda:${gpu}" --dtype float32 \
      --warmup 3 --iterations 5 \
      --output "$out_dir/profiled-timing-discard/${case_name}.json"
done
