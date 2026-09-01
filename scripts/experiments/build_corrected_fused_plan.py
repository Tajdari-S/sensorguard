#!/usr/bin/env python3
"""Build a UUID-isolated fused-update confirmation after corrected freezing."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--start-delay-s", type=float, default=90)
    parser.add_argument("--start-epoch-s", type=float)
    parser.add_argument("--cadence-s", type=int, default=135)
    parser.add_argument("--seed-offset", type=int, default=30000)
    args = parser.parse_args()

    template = json.loads(args.template.read_text())
    frozen = json.loads(args.freeze_manifest.read_text())
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
        runs.append(run)

    plan = {
        "campaign": "corrected-fused-update-confirmation",
        "protocol": (
            "corrected models frozen before collection; rep-major; UUID-isolated "
            "GPU1; fused family was known from an invalid earlier attempt and is "
            "therefore confirmation rather than an untouched attack family"
        ),
        "gpu_index": args.gpu_index,
        "expected_cuda_uuid": args.gpu_uuid,
        "cuda_visible_devices": args.gpu_uuid,
        "scope_serial": "12789/2929",
        "scope_channel": "A",
        "sample_interval_us": 100,
        "freeze_manifest_sha256": digest(args.freeze_manifest),
        "frozen_models": [
            {
                "modality": record["modality"],
                "sha256": record["sha256"],
            }
            for record in frozen["models"]
        ],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "runs": len(runs),
        "first_start": start,
        "last_end": runs[-1]["start_epoch_s"] + runs[-1]["duration_s"],
        "freeze_manifest_sha256": plan["freeze_manifest_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
