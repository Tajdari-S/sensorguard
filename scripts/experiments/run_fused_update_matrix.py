#!/usr/bin/env python3
"""Run the validation-only fused-update matrix on a confirmed idle GPU."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MODES = ("fused_update", "forward3", "adamw", "dummy_write", "forward")


def gpu_uuid(index: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    )
    return output.strip()


def compute_processes(index: int):
    target = gpu_uuid(index)
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if row and row[0] == target:
            rows.append(row)
    return rows


def command_for(args, mode: str, rep: int, run_id: str):
    workload = [
        args.python,
        "scripts/workloads/fused_update_workload.py",
        "--mode",
        mode,
        "--device",
        f"cuda:{args.gpu_index}",
        "--duration-s",
        str(args.duration_s),
        "--batch-size",
        str(args.batch_size),
        "--size",
        str(args.size),
        "--depth",
        str(args.depth),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed + rep),
    ]
    supervisor = [
        args.python,
        "scripts/loggers/supervisor.py",
        "--run-id",
        run_id,
        "--workload-cmd",
        shlex.join(workload),
        "--workload-family",
        "training_evasion" if mode == "fused_update" else "matched_control",
        "--workload-name",
        mode,
        "--gpu-index",
        str(args.gpu_index),
        "--gpus",
        str(args.gpu_index),
        "--sensors",
        args.sensors,
        "--out-root",
        str(args.out_root),
        "--seed",
        str(args.seed + rep),
    ]
    return workload, supervisor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sensors", default="nvml,dcgm")
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-root", type=Path, default=Path("data/validation_runs"))
    parser.add_argument("--summary", type=Path, default=Path("results/attacks/fused_update_runs.csv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--final-test-authorized",
        action="store_true",
        help="confirm R15 is frozen and opening the preregistered held-out family is authorized",
    )
    parser.add_argument(
        "--allow-held-out-gpu4",
        action="store_true",
        help="explicit protocol override; do not use before the final-test freeze",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.final_test_authorized:
        parser.error(
            "fused_update_kernel is the preregistered held-out family; "
            "execution is blocked until R15 is frozen"
        )
    if args.gpu_index == 4 and not args.allow_held_out_gpu4:
        parser.error("GPU 4 is held out; use a validation GPU or explicitly record the override")
    if not args.dry_run:
        occupants = compute_processes(args.gpu_index)
        if occupants:
            print(f"REFUSING: GPU {args.gpu_index} has compute processes: {occupants}", file=sys.stderr)
            return 3

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = []
    for rep in range(args.repetitions):
        # Rep-major order spreads thermal drift across modes.
        for mode in MODES:
            run_id = f"{stamp}_fused-{mode}_gpu{args.gpu_index}_r{rep + 1}"
            workload, supervisor = command_for(args, mode, rep, run_id)
            plan.append((run_id, mode, rep + 1, workload, supervisor))

    if args.dry_run:
        for _, _, _, _, supervisor in plan:
            print(shlex.join(supervisor))
        return 0

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, mode, rep, _, supervisor in plan:
        # Re-check between cells: never start over a newly arrived job.
        occupants = compute_processes(args.gpu_index)
        if occupants:
            print(f"STOPPING QUEUE SAFELY: GPU became occupied: {occupants}", file=sys.stderr)
            break
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(supervisor).returncode
        manifest = args.out_root / run_id / "manifest.yaml"
        rows.append(
            {
                "run_id": run_id,
                "mode": mode,
                "repetition": rep,
                "gpu_index": args.gpu_index,
                "sensors": args.sensors,
                "started_utc": started,
                "return_code": completed,
                "manifest_path": str(manifest),
            }
        )
        with args.summary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if completed != 0:
            print(f"cell {run_id} failed; continuing so failures remain auditable", file=sys.stderr)

    print(json.dumps({"planned": len(plan), "completed": len(rows), "summary": str(args.summary)}))
    return 0 if len(rows) == len(plan) and all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
