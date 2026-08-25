#!/usr/bin/env python3
"""Aggregate run manifests into the E1 calibration report (per-channel pass/fail)."""

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--output", type=Path, default=None, help="markdown output; default stdout")
    args = parser.parse_args()

    rows = []
    for manifest in sorted(args.runs_dir.glob("*/manifest.yaml")):
        doc = yaml.safe_load(manifest.read_text())
        for ch in doc.get("sensor_channels", []):
            rows.append({
                "run_id": doc["run_id"],
                "workload": doc.get("workload", {}).get("name", ""),
                "status": doc.get("status", ""),
                "channel": ch.get("channel_id"),
                "health": ch.get("health"),
                "samples": ch.get("samples"),
                "missing": ch.get("missing_fraction"),
                "align_ms": ch.get("alignment_error_ms"),
                "edge_ms": ch.get("edge_latency_ms"),
            })
    if not rows:
        print("No manifests found.", file=sys.stderr)
        return 1

    lines = ["# E1 calibration report", "",
             "| run_id | workload | run status | channel | health | samples | missing | align ms | edge ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run_id']} | {r['workload']} | {r['status']} | {r['channel']} "
                     f"| {r['health']} | {r['samples']} | {r['missing']} | {r['align_ms']} | {r['edge_ms']} |")
    n_fail = sum(1 for r in rows if r["health"] != "pass")
    lines += ["", f"Channels: {len(rows)}, failing: {n_fail}."]
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
