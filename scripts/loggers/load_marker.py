#!/usr/bin/env python3
"""Scripted load marker: N GEMM bursts that produce an unambiguous power
edge on the target GPU. Prints one line per burst with CLOCK_MONOTONIC_RAW
start/end so the supervisor can measure (not assume) per-channel alignment.
"""

import argparse
import sys
import time

import torch


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bursts", type=int, default=3)
    parser.add_argument("--burst-s", type=float, default=2.0)
    parser.add_argument("--gap-s", type=float, default=2.0)
    parser.add_argument("--size", type=int, default=8192)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)
    torch.cuda.synchronize(device)

    for i in range(args.bursts):
        start = raw_now()
        # Synchronize every iteration: unsynced enqueue races ahead of a
        # slow/throttled GPU and the drain can exceed any timeout.
        while raw_now() - start < args.burst_s:
            a @ b
            torch.cuda.synchronize(device)
        end = raw_now()
        print(f"marker_burst {i} start_raw_s={start:.6f} end_raw_s={end:.6f} device={args.device}", flush=True)
        time.sleep(args.gap_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
