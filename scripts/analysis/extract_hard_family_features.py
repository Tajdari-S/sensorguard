#!/usr/bin/env python3
"""Extract compact synchronized features on the node or verifier independently."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_synchronized_physical import (  # noqa: E402
    nvml_seconds,
    physical_seconds,
    windows,
)


NVML_COLUMNS = [
    "util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w", "temp_c",
    "clock_sm_mhz", "clock_mem_mhz", "pcie_tx_kbps", "pcie_rx_kbps",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["node", "verifier"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    status_path = args.root / f"status_{args.role}.csv"
    status = pd.read_csv(status_path)
    expected = {run["run_id"] for run in plan["runs"]}
    observed = set(status.run_id)
    if observed != expected or len(status) != len(expected):
        raise RuntimeError(
            f"incomplete {args.role} status: {len(observed)}/{len(expected)} run IDs"
        )
    if (status.return_code != 0).any():
        failed = status.loc[status.return_code != 0, ["run_id", "return_code"]]
        raise RuntimeError(f"{args.role} contains failed cells:\n{failed}")

    frames = []
    for run in plan["runs"]:
        run_dir = args.root / run["run_id"]
        start = float(run["start_epoch_s"])
        end = start + float(run["duration_s"])
        workload_path = run_dir / "workload.json"
        if args.role == "node" and workload_path.is_file():
            workload = json.loads(workload_path.read_text())
            start = float(workload["start_epoch_s"])
            end = float(workload["end_epoch_s"])
        if args.role == "node":
            seconds = nvml_seconds(run_dir / "nvml.csv", start, end)
            frame = windows(
                seconds, NVML_COLUMNS, "nvml_", run["run_id"], run["mode"],
                int(run["target"]),
            )
        else:
            seconds = physical_seconds(run_dir, start, end)
            frame = windows(
                seconds, [column for column in seconds if column != "second"],
                "elec_", run["run_id"], run["mode"], int(run["target"]),
            )
        if frame.empty:
            raise RuntimeError(f"no windows extracted for {run['run_id']}")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    ready = {
        "role": args.role,
        "runs": int(result.run_id.nunique()),
        "windows": len(result),
        "features": len(result.columns) - 5,
        "output": str(args.output),
    }
    (args.output.parent / f"ready_{args.role}.json").write_text(
        json.dumps(ready, indent=2) + "\n"
    )
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
