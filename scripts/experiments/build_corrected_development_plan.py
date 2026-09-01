#!/usr/bin/env python3
"""Rebase the 30-run physical development plan onto UUID-isolated GPUs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--secondary-gpu-uuid", required=True)
    parser.add_argument("--start-delay-s", type=float, default=90)
    parser.add_argument("--start-epoch-s", type=float)
    parser.add_argument("--cadence-s", type=int, default=150)
    parser.add_argument("--seed-offset", type=int, default=20000)
    args = parser.parse_args()

    template = json.loads(args.template.read_text())
    start = (
        args.start_epoch_s
        if args.start_epoch_s is not None
        else time.time() + args.start_delay_s
    )
    runs = []
    for index, old in enumerate(template["runs"]):
        run = dict(old)
        run["run_id"] = "corrected_" + str(old["run_id"])
        run["start_epoch_s"] = start + index * args.cadence_s
        run["seed"] = int(old["seed"]) + args.seed_offset
        run["cuda_device"] = "cuda:0"
        run["secondary_cuda_device"] = "cuda:1"
        runs.append(run)

    plan = {
        "campaign": "current-paper-physical-development-corrected-uuid",
        "protocol": (
            "three repetitions per matched family; rep-major; UUID-isolated "
            "RTX 3090 GPU1; migration secondary GPU2; 10 kS/s current and 1 Hz NVML"
        ),
        "scope_serial": "12789/2929",
        "scope_channel": "A",
        "sample_interval_us": 100,
        "gpu_index": args.gpu_index,
        "expected_cuda_uuid": args.gpu_uuid,
        "cuda_visible_devices": (
            f"{args.gpu_uuid},{args.secondary_gpu_uuid}"
        ),
        "secondary_gpu_uuid": args.secondary_gpu_uuid,
        "run_duration_s": int(runs[0]["duration_s"]),
        "cadence_s": args.cadence_s,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "runs": len(runs),
        "first_start": start,
        "last_end": runs[-1]["start_epoch_s"] + runs[-1]["duration_s"],
        "cuda_visible_devices": plan["cuda_visible_devices"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
