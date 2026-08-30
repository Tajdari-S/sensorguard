#!/usr/bin/env python3
"""Run one UUID-pinned node1 development workload with NVML/DCGM logging."""

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MODES = (
    "ordinary_training",
    "duty_shaping",
    "interleaving",
    "memory_minimal",
    "migration",
    "inference_control",
)


def gpu_uuid(index: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    )
    return output.strip()


def compute_processes(uuids: set[str]):
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
        if row and row[0] in uuids and len(row) > 1 and row[1] not in {"[N/A]", "N/A"}:
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--secondary-gpu-index", type=int)
    parser.add_argument("--secondary-gpu-uuid")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--training-fraction", type=float, default=0.35)
    parser.add_argument("--sensors", default="nvml,dcgm")
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-root", type=Path, default=Path("data/development_runs"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.gpu_index == 0:
        parser.error("node1 GPU 0 is quarantined and may not be used")
    if gpu_uuid(args.gpu_index) != args.gpu_uuid:
        parser.error("primary GPU index/UUID mapping does not match live nvidia-smi")
    if args.mode == "migration":
        if args.secondary_gpu_index is None or not args.secondary_gpu_uuid:
            parser.error("migration requires --secondary-gpu-index and --secondary-gpu-uuid")
        if args.secondary_gpu_index == 0:
            parser.error("node1 GPU 0 is quarantined and may not be used")
        if gpu_uuid(args.secondary_gpu_index) != args.secondary_gpu_uuid:
            parser.error("secondary GPU index/UUID mapping does not match live nvidia-smi")

    indices = [args.gpu_index]
    uuids = [args.gpu_uuid]
    if args.mode == "migration":
        indices.append(args.secondary_gpu_index)
        uuids.append(args.secondary_gpu_uuid)
    visible = ",".join(uuids)
    logger_gpus = ",".join(str(index) for index in indices)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = args.summary or Path(f"results/development_tests/{stamp}_{args.mode}_runs.csv")

    plans = []
    for repetition in range(1, args.repetitions + 1):
        seed = args.seed + repetition - 1
        run_id = f"{stamp}_dev-{args.mode}_node1-gpu{'-'.join(map(str, indices))}_r{repetition:02d}"
        useful_work = args.out_root / run_id / "useful_work.json"
        workload = [
            "env",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID",
            f"CUDA_VISIBLE_DEVICES={visible}",
            args.python,
            "scripts/workloads/development_evasion_workload.py",
            "--mode",
            args.mode,
            "--device",
            "cuda:0",
            "--duration-s",
            str(args.duration_s),
            "--batch-size",
            str(args.batch_size),
            "--size",
            str(args.size),
            "--depth",
            str(args.depth),
            "--training-fraction",
            str(args.training_fraction),
            "--seed",
            str(seed),
            "--output",
            str(useful_work),
        ]
        if args.mode == "migration":
            workload.extend(["--secondary-device", "cuda:1"])
        supervisor = [
            args.python,
            "scripts/loggers/supervisor.py",
            "--run-id",
            run_id,
            "--workload-cmd",
            shlex.join(workload),
            "--workload-family",
            "development_training_evasion",
            "--workload-name",
            args.mode,
            "--gpu-index",
            str(args.gpu_index),
            "--gpu-uuid",
            args.gpu_uuid,
            "--gpus",
            logger_gpus,
            "--sensors",
            args.sensors,
            "--out-root",
            str(args.out_root),
            "--python",
            args.python,
            "--seed",
            str(seed),
        ]
        plans.append((run_id, repetition, seed, supervisor))

    if args.dry_run:
        for _, _, _, command in plans:
            print(shlex.join(command))
        return 0

    occupants = compute_processes(set(uuids))
    if occupants:
        print(f"REFUSING: requested GPU has compute processes: {occupants}", file=sys.stderr)
        return 3

    rows = []
    for run_id, repetition, seed, command in plans:
        occupants = compute_processes(set(uuids))
        if occupants:
            print(f"STOPPING QUEUE SAFELY: GPU became occupied: {occupants}", file=sys.stderr)
            break
        started = datetime.now(timezone.utc).isoformat()
        return_code = subprocess.run(command).returncode
        rows.append(
            {
                "run_id": run_id,
                "mode": args.mode,
                "repetition": repetition,
                "seed": seed,
                "gpu_indices": logger_gpus,
                "gpu_uuids": visible,
                "sensors": args.sensors,
                "started_utc": started,
                "return_code": return_code,
                "manifest_path": str(args.out_root / run_id / "manifest.yaml"),
            }
        )
        write_rows(summary, rows)
        if return_code != 0:
            print(f"cell {run_id} failed; later cells remain auditable", file=sys.stderr)

    result = {"planned": len(plans), "completed": len(rows), "summary": str(summary)}
    print(json.dumps(result, sort_keys=True))
    return 0 if len(rows) == len(plans) and all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
