#!/usr/bin/env python3
"""Audit negative-exposure manifests and summarize eligible GPU-hours.

Directory counts are not exposure. This tool accepts manifest files or
directories, retains only completed runs with a positive measured workload
duration and healthy sensor channels, and records every exclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import yaml


RUN_LOCATION = re.compile(r"_(?P<host>[^_]+)-gpu(?P<gpu>\d+)_")


def manifest_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_file():
            paths.add(item)
        elif item.is_dir():
            paths.update(item.rglob("manifest.yaml"))
    return sorted(paths)


def parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def audit_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    run_id = str(data.get("run_id", path.parent.name))
    match = RUN_LOCATION.search(run_id)
    duration = (data.get("workload") or {}).get("duration_s")
    try:
        duration_s = float(duration)
    except (TypeError, ValueError):
        duration_s = 0.0
    channels = data.get("sensor_channels") or []
    unhealthy = [str(c.get("channel_id", "unknown")) for c in channels
                 if c.get("health") != "pass"]
    reasons = []
    if data.get("status") != "completed":
        reasons.append(f"status={data.get('status', 'missing')}")
    if duration_s <= 0:
        reasons.append("missing_or_nonpositive_workload_duration")
    if not channels:
        reasons.append("no_sensor_channels")
    if unhealthy:
        reasons.append("unhealthy_channels=" + ";".join(unhealthy))
    return {
        "path": str(path),
        "run_id": run_id,
        "host": match.group("host") if match else "unknown",
        "gpu": int(match.group("gpu")) if match else None,
        "start": parse_time(data.get("start_utc")),
        "end": parse_time(data.get("end_utc")),
        "duration_s": duration_s,
        "status": str(data.get("status", "missing")),
        "channels": [str(c.get("channel_id", "unknown")) for c in channels],
        "eligible": not reasons,
        "exclusion_reason": "|".join(reasons),
    }


def overlap_pairs(rows: list[dict]) -> list[dict]:
    overlaps = []
    by_device: dict[tuple[str, int | None], list[dict]] = {}
    for row in rows:
        if row["eligible"] and row["start"] and row["end"]:
            by_device.setdefault((row["host"], row["gpu"]), []).append(row)
    for (host, gpu), group in by_device.items():
        group.sort(key=lambda row: row["start"])
        for previous, current in zip(group, group[1:]):
            if current["start"] < previous["end"]:
                overlaps.append({
                    "host": host,
                    "gpu": gpu,
                    "first_run_id": previous["run_id"],
                    "second_run_id": current["run_id"],
                })
    return overlaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--prior-hours", type=float, default=0.0)
    parser.add_argument("--assume-zero-alerts", action="store_true",
                        help="Compute the zero-event bound only after detector evaluation confirms zero alerts")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--failure-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = [audit_manifest(path) for path in manifest_paths(args.inputs)]
    eligible = [row for row in rows if row["eligible"]]
    measured_hours = sum(row["duration_s"] for row in eligible) / 3600.0
    total_hours = args.prior_hours + measured_hours
    overlaps = overlap_pairs(rows)
    summary = {
        "manifest_count": len(rows),
        "eligible_manifest_count": len(eligible),
        "excluded_manifest_count": len(rows) - len(eligible),
        "prior_eligible_gpu_hours": round(args.prior_hours, 6),
        "new_eligible_gpu_hours": round(measured_hours, 6),
        "total_eligible_gpu_hours": round(total_hours, 6),
        "same_gpu_manifest_overlaps": overlaps,
        "zero_false_alerts_confirmed": bool(args.assume_zero_alerts),
        "one_sided_95pct_false_alert_upper_bound_per_gpu_hour": (
            round(-math.log(0.05) / total_hours, 9)
            if args.assume_zero_alerts and total_hours > 0 and not overlaps else None
        ),
        "eligible_runs": [
            {
                "run_id": row["run_id"], "host": row["host"], "gpu": row["gpu"],
                "duration_s": row["duration_s"], "channels": row["channels"],
            }
            for row in eligible
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.failure_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.failure_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "host", "gpu", "status", "duration_s", "exclusion_reason", "path"
        ])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames}
                         for row in rows if not row["eligible"])
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("eligible_runs", "same_gpu_manifest_overlaps")}, indent=2))
    return 1 if overlaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
