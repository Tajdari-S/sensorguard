#!/usr/bin/env python3
"""Inspect or materialize SensorGuard run queues without launching GPU code."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def select(rows, group, priority, status):
    for row in rows:
        if group and row["category"] != group:
            continue
        if priority and row["priority"] != priority:
            continue
        if status and row["status"] != status:
            continue
        yield row


def main() -> int:
    parser = argparse.ArgumentParser(description="List workload coverage or create an auditable CSV queue")
    parser.add_argument("--manifest", default="configs/workloads.json")
    parser.add_argument("--group", choices=["training", "inference", "non_ml", "evasion", "reviewer_extension"])
    parser.add_argument("--priority", choices=["P0", "P1", "P2"])
    parser.add_argument("--status")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--gpu-set", default="0", help="logical GPU set, e.g. 0 or 0,1,2,3")
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--write-queue", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.duration_s < 1:
        parser.error("repetitions and duration must be positive")
    rows = json.loads(Path(args.manifest).read_text())["workloads"]
    chosen = list(select(rows, args.group, args.priority, args.status))
    if args.write_queue:
        args.write_queue.parent.mkdir(parents=True, exist_ok=True)
        with args.write_queue.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "workload_id", "variant", "gpu_set", "duration_s", "status", "created_utc"])
            writer.writeheader()
            created = datetime.now(timezone.utc).isoformat()
            for row in chosen:
                for variant in row["variants"]:
                    for rep in range(1, args.repetitions + 1):
                        writer.writerow({
                            "run_id": f"{row['id']}__r{rep:02d}", "workload_id": row["id"],
                            "variant": variant, "gpu_set": args.gpu_set, "duration_s": args.duration_s,
                            "status": "planned", "created_utc": created,
                        })
        print(f"Wrote {args.write_queue}")
    else:
        print("id\tpriority\tstatus\tapplication")
        for row in chosen:
            print(f"{row['id']}\t{row['priority']}\t{row['status']}\t{row['application']}")
    print(f"Selected {len(chosen)} records; no GPU workload was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
