#!/usr/bin/env python3
"""Build a synchronized physical negative-exposure campaign on mapped GPU1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


MODES = ("gemm", "fft", "memcpy", "idle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--duration-s", type=int, default=300)
    parser.add_argument("--cadence-s", type=int, default=345)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--start-delay-s", type=float, default=90)
    parser.add_argument("--start-epoch-s", type=float)
    args = parser.parse_args()

    start = (
        args.start_epoch_s
        if args.start_epoch_s is not None
        else time.time() + args.start_delay_s
    )
    runs = []
    cell = 0
    for repetition in range(1, args.repetitions + 1):
        for mode in MODES:
            runs.append({
                "run_id": (
                    f"physical_negative_{cell + 1:02d}_{mode}_r{repetition}"
                ),
                "kind": "control",
                "mode": mode,
                "target": 0,
                "repetition": repetition,
                "duration_s": args.duration_s,
                "start_epoch_s": start + cell * args.cadence_s,
                "seed": 13000 + cell,
                # The runner isolates the physical UUID, so it becomes CUDA
                # ordinal zero even when an unhealthy lower-index GPU is
                # omitted from CUDA enumeration.
                "cuda_device": "cuda:0",
            })
            cell += 1
    plan = {
        "campaign": "current-paper-physical-negative-exposure",
        "protocol": (
            "rep-major matched non-training controls; synchronized 1 Hz NVML "
            "and 10 kS/s GPU current; frozen detector scored once"
        ),
        "gpu_index": args.gpu_index,
        "expected_cuda_uuid": args.gpu_uuid,
        "cuda_visible_devices": args.gpu_uuid,
        "scope_serial": "12789/2929",
        "scope_channel": "A",
        "sample_interval_us": 100,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "runs": len(runs),
        "negative_gpu_hours": len(runs) * args.duration_s / 3600,
        "first_start": start,
        "last_end": runs[-1]["start_epoch_s"] + args.duration_s,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
