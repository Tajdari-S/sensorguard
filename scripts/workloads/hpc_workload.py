#!/usr/bin/env python3
"""Non-ML P0 workloads: GEMM, FFT, memcpy (adapter for hpc_gemm/hpc_fft/hpc_memcpy).

Each mode reports useful work (iterations and derived FLOP/byte counts) on
stdout at exit so the supervisor can record useful_work_value.
"""

import argparse
import json
import time

import torch


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gemm", "fft", "memcpy"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=300)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    end = raw_now() + args.duration_s
    iters = 0

    if args.mode == "gemm":
        a = torch.randn(args.size, args.size, device=device)
        b = torch.randn(args.size, args.size, device=device)
        torch.cuda.synchronize(device)
        while raw_now() < end:
            a @ b
            iters += 1
            if iters % 50 == 0:
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        useful = {"iterations": iters, "flops": iters * 2 * args.size**3}
    elif args.mode == "fft":
        x = torch.randn(64, args.size, args.size // 8, device=device, dtype=torch.complex64)
        torch.cuda.synchronize(device)
        while raw_now() < end:
            torch.fft.fft2(x)
            iters += 1
            if iters % 20 == 0:
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        useful = {"iterations": iters}
    else:  # memcpy
        host = torch.randn(args.size, args.size, pin_memory=True)
        dev = torch.empty(args.size, args.size, device=device)
        while raw_now() < end:
            dev.copy_(host, non_blocking=True)
            host.copy_(dev, non_blocking=True)
            torch.cuda.synchronize(device)
            iters += 1
        useful = {"iterations": iters, "bytes": iters * 2 * host.numel() * 4}

    print("useful_work " + json.dumps({"mode": args.mode, **useful}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
