#!/usr/bin/env python3
"""Run a UUID-pinned GEMM marker at a shared wall-clock epoch."""

import argparse
import json
import os
import time
from pathlib import Path


def wait_until(epoch_s: float) -> None:
    while True:
        remaining = epoch_s - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--start-epoch-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--cycles", type=int, default=0,
                        help="if positive, alternate heavy-load and idle intervals")
    parser.add_argument("--on-s", type=float, default=6.0)
    parser.add_argument("--off-s", type=float, default=6.0)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch

    device = torch.device("cuda:0")
    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)
    torch.cuda.synchronize(device)
    ready_epoch_s = time.time()
    if ready_epoch_s > args.start_epoch_s:
        raise RuntimeError(
            f"{args.label} missed scheduled start by {ready_epoch_s - args.start_epoch_s:.3f}s"
        )

    wait_until(args.start_epoch_s)
    actual_start_epoch_s = time.time()
    iterations = 0
    active_intervals = []
    inactive_intervals = []
    if args.cycles > 0:
        for cycle in range(args.cycles):
            target_on_start = actual_start_epoch_s + cycle * (args.on_s + args.off_s)
            wait_until(target_on_start)
            on_start = time.time()
            target_on_end = target_on_start + args.on_s
            while time.time() < target_on_end:
                a @ b
                torch.cuda.synchronize(device)
                iterations += 1
            on_end = time.time()
            active_intervals.append([on_start, on_end])
            target_off_end = target_on_end + args.off_s
            off_start = time.time()
            wait_until(target_off_end)
            inactive_intervals.append([off_start, time.time()])
    else:
        end_epoch_s = actual_start_epoch_s + args.duration_s
        while time.time() < end_epoch_s:
            a @ b
            torch.cuda.synchronize(device)
            iterations += 1
    actual_end_epoch_s = time.time()

    payload = {
        "label": args.label,
        "gpu_uuid": args.gpu_uuid,
        "planned_start_epoch_s": args.start_epoch_s,
        "actual_start_epoch_s": actual_start_epoch_s,
        "actual_end_epoch_s": actual_end_epoch_s,
        "duration_s": actual_end_epoch_s - actual_start_epoch_s,
        "iterations": iterations,
        "cycles": args.cycles,
        "on_s": args.on_s if args.cycles else None,
        "off_s": args.off_s if args.cycles else None,
        "active_intervals_epoch_s": active_intervals,
        "inactive_intervals_epoch_s": inactive_intervals,
        "hostname": os.uname().nodename,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
