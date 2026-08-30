#!/usr/bin/env python3
"""Scheduled matched non-training controls for physical-sensor collection."""

import argparse
import json
import time
from pathlib import Path

import torch


def wait_for_epoch(target: float) -> None:
    if time.time() > target + 2.0:
        raise RuntimeError(f"scheduled start {target} is already more than 2 s late")
    while time.time() < target:
        time.sleep(min(0.05, target - time.time()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["idle", "gemm", "fft", "memcpy"])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--start-epoch-s", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # Allocate before the scheduled edge so initialization is excluded.
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    c = torch.empty_like(a)
    torch.cuda.synchronize(device)
    wait_for_epoch(args.start_epoch_s)
    start_epoch = time.time()
    deadline = time.monotonic() + args.duration_s
    iterations = 0
    if args.mode == "idle":
        time.sleep(args.duration_s)
    else:
        while time.monotonic() < deadline:
            if args.mode == "gemm":
                torch.mm(a, b, out=c)
            elif args.mode == "fft":
                c.copy_(torch.fft.fft(a, dim=1).real)
            else:
                c.copy_(a)
                a.copy_(b)
                b.copy_(c)
            torch.cuda.synchronize(device)
            iterations += 1
    result = {
        "mode": args.mode,
        "target": 0,
        "device": str(device),
        "seed": args.seed,
        "iterations": iterations,
        "scheduled_start_epoch_s": args.start_epoch_s,
        "start_epoch_s": start_epoch,
        "end_epoch_s": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
