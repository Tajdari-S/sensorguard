#!/usr/bin/env python3
"""CUDA microbenchmarks for constructing an empirical roofline."""

import argparse
import json
import statistics
from pathlib import Path

CASES = {
    "gemm_1024": ("gemm", {"m": 1024, "n": 1024, "k": 1024}),
    "gemm_2048": ("gemm", {"m": 2048, "n": 2048, "k": 2048}),
    "gemm_4096": ("gemm", {"m": 4096, "n": 4096, "k": 4096}),
    "gemm_tall": ("gemm", {"m": 16384, "n": 512, "k": 1024}),
    "gemv": ("gemv", {"m": 16384, "k": 4096}),
    "copy": ("copy", {"elements": 32 * 1024 * 1024}),
    "elementwise": ("elementwise", {"elements": 32 * 1024 * 1024}),
}


def case_metadata(name, bytes_per_element):
    op, dims = CASES[name]
    if op == "gemm":
        m, n, k = dims["m"], dims["n"], dims["k"]
        flops = 2 * m * n * k
        minimum_bytes = bytes_per_element * (m * k + k * n + m * n)
    elif op == "gemv":
        m, k = dims["m"], dims["k"]
        flops = 2 * m * k
        minimum_bytes = bytes_per_element * (m * k + k + m)
    elif op == "copy":
        flops = 0
        minimum_bytes = 2 * bytes_per_element * dims["elements"]
    else:
        flops = 2 * dims["elements"]
        minimum_bytes = 2 * bytes_per_element * dims["elements"]
    return {"case": name, "operation": op, "dimensions": dims, "flops": flops,
            "minimum_bytes": minimum_bytes,
            "arithmetic_intensity_min": flops / minimum_bytes if minimum_bytes else 0.0}


def build_operation(torch, name, device, dtype):
    op, dims = CASES[name]
    if op == "gemm":
        a = torch.randn((dims["m"], dims["k"]), device=device, dtype=dtype)
        b = torch.randn((dims["k"], dims["n"]), device=device, dtype=dtype)
        return lambda: torch.mm(a, b)
    if op == "gemv":
        a = torch.randn((dims["m"], dims["k"]), device=device, dtype=dtype)
        x = torch.randn((dims["k"],), device=device, dtype=dtype)
        return lambda: torch.mv(a, x)
    x = torch.randn((dims["elements"],), device=device, dtype=dtype)
    if op == "copy":
        y = torch.empty_like(x)
        return lambda: y.copy_(x)
    return lambda: x.mul(1.1).add(2.0)


def benchmark(torch, name, device, dtype, warmup, iterations):
    fn = build_operation(torch, name, device, dtype)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    meta = case_metadata(name, torch.tensor([], dtype=dtype).element_size())
    median_ms = statistics.median(samples)
    seconds = median_ms / 1000.0
    meta.update({"dtype": str(dtype).replace("torch.", ""), "device": str(device),
                 "warmup": warmup, "iterations": iterations, "median_ms": median_ms,
                 "tflops": meta["flops"] / seconds / 1e12 if seconds else 0.0,
                 "minimum_gbps": meta["minimum_bytes"] / seconds / 1e9 if seconds else 0.0})
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(CASES)))
        return 0
    if args.output is None:
        parser.error("--output is required unless --list is used")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required for roofline execution")
    dtype = getattr(torch, args.dtype)
    cases = args.case or list(CASES)
    results = [benchmark(torch, case, args.device, dtype, args.warmup, args.iterations)
               for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
