#!/usr/bin/env python3
"""E1 calibration workloads: idle, fixed GEMM, memory copy, bursty load."""

import argparse
import time

import torch


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["idle", "gemm", "memcpy", "bursty"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=60)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--burst-period-s", type=float, default=10, help="bursty: on/off half-period")
    args = parser.parse_args()

    end = raw_now() + args.duration_s
    if args.mode == "idle":
        while raw_now() < end:
            time.sleep(0.25)
        return 0

    device = torch.device(args.device)
    if args.mode == "memcpy":
        host = torch.randn(args.size, args.size, pin_memory=True)
        dev = torch.empty(args.size, args.size, device=device)
        while raw_now() < end:
            dev.copy_(host, non_blocking=True)
            host.copy_(dev, non_blocking=True)
            torch.cuda.synchronize(device)
        return 0

    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)
    torch.cuda.synchronize(device)
    if args.mode == "gemm":
        while raw_now() < end:
            a @ b
            torch.cuda.synchronize(device)
        return 0

    # bursty: alternate GEMM load and sleep at the given half-period
    while raw_now() < end:
        t = raw_now() + args.burst_period_s
        while raw_now() < min(t, end):
            a @ b
        torch.cuda.synchronize(device)
        time.sleep(max(0, min(args.burst_period_s, end - raw_now())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
