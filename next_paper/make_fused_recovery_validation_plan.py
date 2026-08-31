#!/usr/bin/env python3
"""Materialize the fresh post-recovery validation plan with absolute times."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


VARIANTS = (
    {"variant": "smaller_deeper", "batch_size": 256, "size": 1536, "depth": 4,
     "dtype": "float32", "learning_rate": 0.001, "seed": 23100},
    {"variant": "baseline_shape_new_seed", "batch_size": 256, "size": 2048, "depth": 3,
     "dtype": "float32", "learning_rate": 0.001, "seed": 24100},
    {"variant": "wider_shallower", "batch_size": 128, "size": 2560, "depth": 2,
     "dtype": "float32", "learning_rate": 0.001, "seed": 25100},
)
MODES = (
    ("fused_update", 1), ("forward3", 0), ("adamw", 1),
    ("dummy_write", 0), ("forward", 0),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-epoch-s", type=float)
    parser.add_argument("--lead-s", type=int, default=180)
    parser.add_argument("--duration-s", type=int, default=90)
    parser.add_argument("--cadence-s", type=int, default=120)
    args = parser.parse_args()
    start = args.start_epoch_s
    if start is None:
        start = math.ceil((time.time() + args.lead_s) / 60) * 60
    runs = []
    cell = 0
    for repetition, variant in enumerate(VARIANTS, start=1):
        for mode, target in MODES:
            cell += 1
            run = {
                "run_id": f"recovery_{cell:02d}_{mode}_{variant['variant']}_r{repetition}",
                "kind": "fused", "mode": mode, "target": target,
                "duration_s": args.duration_s,
                "start_epoch_s": start + (cell - 1) * args.cadence_s,
                **variant,
            }
            run["seed"] = variant["seed"] + cell
            runs.append(run)
    plan = {
        "campaign": "post-sealed-fused-update-recovery-validation",
        "protocol": (
            "one-time evaluation of frozen recovery artifact on three fresh "
            "workload configurations; rep-major matched controls"
        ),
        "frozen_artifact_sha256":
            "2f922014b1a132b20cf605746584dea074d6a72bcbc91607589bf3389a9db648",
        "scope_serial": "12789/2929", "scope_channel": "A",
        "gpu_index": 1, "sample_interval_us": 100,
        "start_epoch_s": start, "cadence_s": args.cadence_s,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "start_epoch_s": start,
                      "runs": len(runs),
                      "finish_epoch_s": runs[-1]["start_epoch_s"] + args.duration_s}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
