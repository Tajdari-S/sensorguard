#!/usr/bin/env python3
"""Build freeze-separated development and sealed physical red-team plans."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEVELOPMENT_FAMILIES = (
    ("honest_full", 1),
    ("chunked_optimizer", 1),
    ("throttled", 1),
    ("inference_dilution", 1),
    ("inference_control", 0),
)
SEALED_FAMILIES = (
    ("lora_dilution", 1),
    ("inference_control", 0),
)


def make_runs(
    families: tuple[tuple[str, int], ...],
    repetitions: int,
    start: float,
    cadence_s: int,
    duration_s: int,
    prefix: str,
    seed_base: int,
) -> list[dict[str, object]]:
    runs = []
    cell = 0
    for repetition in range(1, repetitions + 1):
        for mode, target in families:
            runs.append({
                "run_id": f"{prefix}_{cell + 1:03d}_{mode}_r{repetition:02d}",
                "kind": "redteam",
                "mode": mode,
                "family": mode,
                "target": target,
                "repetition": repetition,
                "duration_s": duration_s,
                "start_epoch_s": start + cell * cadence_s,
                "seed": seed_base + cell,
                "cuda_device": "cuda:0",
                "batch_size": 128,
                "size": 2048,
                "depth": 3,
                "dtype": "float32",
                "learning_rate": 1e-3,
                "optimizer_chunks": 8,
                "dilution": 20,
                "throttle_s": 0.05,
                "lora_rank": 8,
            })
            cell += 1
    return runs


def plan(
    campaign: str,
    protocol: str,
    runs: list[dict[str, object]],
    gpu_index: int,
    gpu_uuid: str,
    scope_serial: str,
    sealed: bool,
) -> dict[str, object]:
    return {
        "campaign": campaign,
        "protocol": protocol,
        "status": "code-ready; not yet collected",
        "sealed": sealed,
        "requires_frozen_detector_manifest": sealed,
        "gpu_index": gpu_index,
        "expected_cuda_uuid": gpu_uuid,
        "cuda_visible_devices": gpu_uuid,
        "scope_serial": scope_serial,
        "scope_channel": "A",
        "sample_interval_us": 100,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--scope-serial", default="12789/2929")
    parser.add_argument("--start-delay-s", type=float, default=120)
    parser.add_argument("--start-epoch-s", type=float)
    parser.add_argument("--duration-s", type=int, default=300)
    parser.add_argument("--cadence-s", type=int, default=345)
    parser.add_argument("--development-repetitions", type=int, default=10)
    parser.add_argument("--sealed-repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.development_repetitions < 1 or args.sealed_repetitions < 1:
        parser.error("repetitions must be positive")

    development_start = (
        args.start_epoch_s if args.start_epoch_s is not None
        else time.time() + args.start_delay_s
    )
    development_runs = make_runs(
        DEVELOPMENT_FAMILIES,
        args.development_repetitions,
        development_start,
        args.cadence_s,
        args.duration_s,
        "dev",
        91000,
    )
    # These timestamps are only a template. The sealed plan remains blocked by
    # run_synchronized_physical.py until a detector manifest is frozen, and
    # should then be regenerated with a fresh --start-epoch-s.
    sealed_start = development_runs[-1]["start_epoch_s"] + args.cadence_s
    sealed_runs = make_runs(
        SEALED_FAMILIES,
        args.sealed_repetitions,
        float(sealed_start),
        args.cadence_s,
        args.duration_s,
        "sealed",
        92000,
    )
    development = plan(
        "stronger-paper-redteam-development",
        "rep-major matched RTX 3090 development; paired 1 Hz NVML and 10 kS/s rail current",
        development_runs,
        args.gpu_index,
        args.gpu_uuid,
        args.scope_serial,
        False,
    )
    sealed = plan(
        "stronger-paper-redteam-sealed-lora",
        "opened only after feature, threshold, run rule, and rescue rule are frozen",
        sealed_runs,
        args.gpu_index,
        args.gpu_uuid,
        args.scope_serial,
        True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_path = args.output_dir / "development_plan.json"
    sealed_path = args.output_dir / "sealed_lora_plan.json"
    development_path.write_text(json.dumps(development, indent=2) + "\n")
    sealed_path.write_text(json.dumps(sealed, indent=2) + "\n")
    print(json.dumps({
        "development_plan": str(development_path),
        "development_runs": len(development_runs),
        "sealed_plan": str(sealed_path),
        "sealed_runs": len(sealed_runs),
        "instruction": "collect development, freeze manifest, regenerate timestamps, then collect sealed plan with --frozen-manifest",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
