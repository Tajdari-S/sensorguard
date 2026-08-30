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
    end_epoch_s = actual_start_epoch_s + args.duration_s
    iterations = 0
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
        "hostname": os.uname().nodename,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
