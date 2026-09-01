#!/usr/bin/env python3
"""Build the synchronized RTX 3090 replay of Rahman's hardest NVML families."""

import argparse
import json
import time
from pathlib import Path


TARGETS = (
    ("adversarial_B_low_util", 1, 0.606061, "adversarial_training.py", ["--strategy", "B"]),
    ("adversarial_D_temporal_disruption", 1, 0.678571, "adversarial_training.py", ["--strategy", "D"]),
    ("adversarial_F_interleave_30", 1, 0.642857, "adversarial_advanced.py", ["--strategy", "F", "--train-frac", "0.3"]),
    ("adversarial_composite_10", 1, 0.500000, "adversarial_composite.py", ["--idle-fraction", "0.1"]),
    ("whitebox_lora_N5", 1, 0.611111, "adversarial_whitebox.py", ["--strategy", "WB-L", "--dilution", "5", "--grad-ckpt", "--batch-size", "2", "--seq-len", "256"]),
    ("whitebox_lora_N10", 1, 0.666667, "adversarial_whitebox.py", ["--strategy", "WB-L", "--dilution", "10", "--grad-ckpt", "--batch-size", "2", "--seq-len", "256"]),
    ("whitebox_lora_N20", 1, 0.111111, "adversarial_whitebox.py", ["--strategy", "WB-L", "--dilution", "20", "--grad-ckpt", "--batch-size", "2", "--seq-len", "256"]),
    ("whitebox_inference_control", 0, None, "adversarial_whitebox.py", ["--strategy", "WB-I", "--batch-size", "2", "--seq-len", "256"]),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-delay-s", type=float, default=90.0)
    parser.add_argument("--start-epoch-s", type=float,
                        help="explicit shared start time; overrides --start-delay-s")
    parser.add_argument("--duration-s", type=int, default=300)
    parser.add_argument("--cadence-s", type=int, default=345)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--python", default="/home/felkru/spar-venv/bin/python")
    parser.add_argument("--source-root", type=Path, default=Path("/home/felkru/gpumon-repo"))
    parser.add_argument("--workdir", type=Path, default=Path("/home/felkru/SPAR-GPU-monitoring"))
    args = parser.parse_args()

    start = args.start_epoch_s if args.start_epoch_s is not None else time.time() + args.start_delay_s
    runs = []
    cell = 0
    # Repetition-major order balances sensor drift across attack families.
    for repetition in range(1, args.repetitions + 1):
        for label, target, prior_rate, script, extra in TARGETS:
            run_id = f"hard_{cell + 1:02d}_{label}_r{repetition}"
            command = [
                args.python,
                str(args.source_root / "workloads" / script),
                *extra,
                "--duration",
                str(args.duration_s),
                "--device",
                f"cuda:{args.gpu_index}",
            ]
            runs.append({
                "run_id": run_id,
                "kind": "external",
                "mode": label,
                "target": target,
                "repetition": repetition,
                "prior_nvml_leave_family_out_detection": prior_rate,
                "duration_s": args.duration_s,
                "start_epoch_s": start + cell * args.cadence_s,
                "seed": 7600 + cell,
                "cuda_device": f"cuda:{args.gpu_index}",
                "workdir": str(args.workdir),
                "command": command,
            })
            cell += 1

    plan = {
        "campaign": "paired-hard-nvml-families-rtx3090",
        "protocol": "exact Rahman attack implementations; repetition-major; synchronized 1 Hz NVML and 10 kS/s GPU current",
        "gpu_index": args.gpu_index,
        "expected_cuda_uuid": args.gpu_uuid,
        "scope_serial": "12789/2929",
        "scope_channel": "A",
        "sample_interval_us": 100,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(runs), "first_start": start}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
