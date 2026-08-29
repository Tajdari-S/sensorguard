#!/usr/bin/env python3
"""Join unprofiled application timing with NCU DRAM-byte measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from parse_ncu import launch_dram_bytes, parse_launches


def summarize(timing: dict, ncu_csv: Path, peak_tflops=None, peak_gbps=None) -> dict:
    launches = parse_launches(ncu_csv)
    byte_values = [launch_dram_bytes(launch["metrics"]) for launch in launches]
    byte_values = [value for value in byte_values if value is not None]
    if not byte_values:
        raise ValueError(f"no DRAM byte counters found in {ncu_csv}")
    dram_bytes = float(sum(byte_values))
    total_flops = float(timing["total_flops"])
    point = dict(timing)
    point.update({
        "ncu_csv": str(ncu_csv),
        "ncu_launches": len(launches),
        "measured_dram_bytes": dram_bytes,
        "arithmetic_intensity": total_flops / dram_bytes,
        "peak_tflops": peak_tflops,
        "peak_gbps": peak_gbps,
        "ridge_arithmetic_intensity": None,
        "normalized_arithmetic_intensity": None,
        "normalized_wall_throughput": None,
    })
    if peak_tflops and peak_gbps:
        ridge = peak_tflops * 1000.0 / peak_gbps
        point["ridge_arithmetic_intensity"] = ridge
        point["normalized_arithmetic_intensity"] = point["arithmetic_intensity"] / ridge
        point["normalized_wall_throughput"] = point["wall_tflops"] / peak_tflops
    return point


def append_csv(path: Path, row: dict) -> None:
    fields = [
        "case_id", "suite", "platform", "repetition", "gpu_name", "mode", "dtype",
        "batch_size", "seq_len", "decode_tokens", "gap_ms", "iterations",
        "total_flops", "measured_dram_bytes", "arithmetic_intensity",
        "active_tflops", "wall_tflops", "peak_tflops", "peak_gbps",
        "ridge_arithmetic_intensity", "normalized_arithmetic_intensity",
        "normalized_wall_throughput", "ncu_launches", "ncu_csv",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--ncu-csv", type=Path, required=True)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--peak-gbps", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append-csv", type=Path)
    args = parser.parse_args()
    point = summarize(json.loads(args.timing.read_text()), args.ncu_csv,
                      args.peak_tflops, args.peak_gbps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(point, indent=2) + "\n")
    if args.append_csv:
        append_csv(args.append_csv, point)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
