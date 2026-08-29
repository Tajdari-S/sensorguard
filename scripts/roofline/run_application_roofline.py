#!/usr/bin/env python3
"""Plan or execute application-level NCU roofline collection on one idle GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def gpu_uuid(index: int) -> str:
    return subprocess.check_output([
        "nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).strip()


def compute_processes(index: int) -> list[list[str]]:
    target = gpu_uuid(index)
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [row for row in csv.reader(output.splitlines(), skipinitialspace=True)
            if row and row[0] == target]


def workload_command(args, case: dict, repetition: int, output: Path,
                     profile_range: bool) -> list[str]:
    command = [
        args.python, "scripts/roofline/application_workload.py",
        "--case-id", case["case_id"], "--suite", case["suite"],
        "--platform", args.platform, "--repetition", str(repetition),
        "--mode", case["mode"], "--device", f"cuda:{args.gpu_index}",
        "--dtype", case["dtype"], "--batch-size", str(case["batch_size"]),
        "--seq-len", str(case["seq_len"]),
        "--decode-tokens", str(case["decode_tokens"]),
        "--gap-ms", str(case["gap_ms"]), "--warmup", str(args.warmup),
        "--iterations", str(case["iterations"]),
        "--seed", str(args.seed + repetition - 1), "--output", str(output),
    ]
    if profile_range:
        command.append("--profile-range")
    return command


def plan(args, matrix: dict) -> list[dict]:
    repetitions = args.repetitions or int(matrix.get("repetitions", 1))
    cases = [case for case in matrix["cases"]
             if args.suite == "all" or case["suite"] == args.suite]
    if not cases:
        raise ValueError(f"no cases selected for suite {args.suite}")
    rows = []
    for case in cases:
        for repetition in range(1, repetitions + 1):
            stem = f"{case['case_id']}-r{repetition:02d}"
            timing = args.output_root / "timing" / f"{stem}.json"
            ncu_csv = args.output_root / "ncu" / f"{stem}.ncu.csv"
            profiled = args.output_root / "profiled-timing-discard" / f"{stem}.json"
            point = args.output_root / "points" / f"{stem}.json"
            rows.append({
                "case": case,
                "repetition": repetition,
                "timing": timing,
                "ncu_csv": ncu_csv,
                "point": point,
                "timing_command": workload_command(args, case, repetition, timing, False),
                "ncu_command": [
                    args.ncu, "--target-processes", "all", "--profile-from-start", "off",
                    "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum",
                    "--csv", "--log-file", str(ncu_csv),
                    *workload_command(args, case, repetition, profiled, True),
                ],
            })
    return rows


def parser_command(args, item: dict) -> list[str]:
    command = [
        args.python, "scripts/roofline/parse_application_ncu.py",
        "--timing", str(item["timing"]), "--ncu-csv", str(item["ncu_csv"]),
        "--output", str(item["point"]),
        "--append-csv", str(args.output_root / "application-roofline-points.csv"),
    ]
    if args.peak_tflops is not None:
        command.extend(["--peak-tflops", str(args.peak_tflops)])
    if args.peak_gbps is not None:
        command.extend(["--peak-gbps", str(args.peak_gbps)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--platform", required=True,
                        help="stable label such as rtx3090-node2 or h200-nvl")
    parser.add_argument("--suite", choices=["rtx3090_application", "cross_gpu_bridge", "all"],
                        default="rtx3090_application")
    parser.add_argument("--matrix", type=Path,
                        default=Path("configs/application_roofline_matrix.json"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--peak-tflops", type=float,
                        help="measured sustained peak at the same dtype and clocks")
    parser.add_argument("--peak-gbps", type=float,
                        help="measured sustained DRAM bandwidth at the same clocks")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--execute", action="store_true",
                        help="actually use the GPU; otherwise print an auditable plan")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("repetitions must be positive")
    if (args.peak_tflops is None) != (args.peak_gbps is None):
        parser.error("provide both --peak-tflops and --peak-gbps, or neither")
    if args.output_root is None:
        args.output_root = Path("results/roofline/applications") / args.platform / args.suite

    matrix = json.loads(args.matrix.read_text())
    items = plan(args, matrix)
    if not args.execute:
        for item in items:
            print(shlex.join(item["timing_command"]))
            print(shlex.join(item["ncu_command"]))
            print(shlex.join(parser_command(args, item)))
        print(json.dumps({"status": "planned_only", "runs": len(items),
                          "sealed_data_opened": False}))
        return 0

    if shutil.which(args.ncu) is None:
        parser.error(f"NCU executable not found: {args.ncu}")
    points_csv = args.output_root / "application-roofline-points.csv"
    if points_csv.exists() and not args.resume:
        parser.error(
            f"output already exists: {points_csv}; use --resume or a new output root"
        )
    occupants = compute_processes(args.gpu_index)
    if occupants:
        print(f"REFUSING: GPU {args.gpu_index} is occupied: {occupants}", file=sys.stderr)
        return 3
    for directory in ["timing", "ncu", "profiled-timing-discard", "points"]:
        (args.output_root / directory).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    completed = 0
    for item in items:
        if args.resume and item["point"].exists():
            completed += 1
            continue
        occupants = compute_processes(args.gpu_index)
        if occupants:
            print(f"STOPPING SAFELY: GPU became occupied: {occupants}", file=sys.stderr)
            break
        for command in [item["timing_command"], item["ncu_command"], parser_command(args, item)]:
            print(shlex.join(command), flush=True)
            return_code = subprocess.run(command).returncode
            if return_code != 0:
                print(f"FAILED rc={return_code}; leaving artifacts for audit", file=sys.stderr)
                return 1
        completed += 1
    print(json.dumps({"planned": len(items), "completed": completed,
                      "points_csv": str(points_csv)}))
    return 0 if completed == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
