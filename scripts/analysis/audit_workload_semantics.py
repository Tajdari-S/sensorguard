#!/usr/bin/env python3
"""Audit labels, timing, device isolation, and useful work for a campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_workload(run_dir: Path) -> dict:
    path = run_dir / "workload.json"
    if path.is_file():
        return json.loads(path.read_text())
    stdout = run_dir / "workload.stdout"
    if stdout.is_file():
        for line in reversed(stdout.read_text().splitlines()):
            if line.startswith("useful_work "):
                return json.loads(line.removeprefix("useful_work "))
    raise RuntimeError(f"no workload record in {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    rows = []
    for run in plan["runs"]:
        record = load_workload(args.node_root / run["run_id"])
        target = int(run["target"])
        progress = record.get("meaningful_optimization_progress")
        if target == 1 and progress is not True:
            raise RuntimeError(
                f"positive run lacks meaningful optimization: {run['run_id']}"
            )
        mode_matches = record.get("mode") == run["mode"]
        if not mode_matches:
            raise RuntimeError(f"mode mismatch for {run['run_id']}")
        actual_start = float(record["start_epoch_s"])
        start_error = actual_start - float(run["start_epoch_s"])
        rows.append({
            "run_id": run["run_id"],
            "family": run["mode"],
            "target": target,
            "mode_matches_plan": mode_matches,
            "scheduled_start_epoch_s": float(run["start_epoch_s"]),
            "actual_start_epoch_s": actual_start,
            "start_error_s": start_error,
            "meaningful_optimization_progress": progress,
            "relative_loss_reduction": record.get("relative_loss_reduction"),
            "max_weight_change": record.get("max_weight_change"),
            "steps_or_iterations": record.get(
                "steps", record.get("iterations")
            ),
            "workload_cuda_visible_devices": record.get(
                "cuda_visible_devices"
            ),
            "plan_cuda_visible_devices": plan.get("cuda_visible_devices"),
            "expected_cuda_uuid": plan.get("expected_cuda_uuid"),
        })
    result = pd.DataFrame(rows)
    if result.start_error_s.abs().max() > 2.0:
        raise RuntimeError("at least one workload started over 2 s off schedule")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({
        "runs": len(result),
        "positive_runs": int(result.target.sum()),
        "positive_semantics_pass": bool(
            result.loc[result.target.eq(1),
                       "meaningful_optimization_progress"].eq(True).all()
        ),
        "maximum_absolute_start_error_s": float(
            result.start_error_s.abs().max()
        ),
        "expected_cuda_uuid": plan.get("expected_cuda_uuid"),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
