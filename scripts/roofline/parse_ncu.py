#!/usr/bin/env python3
"""Combine benchmark metadata with DRAM-byte metrics exported by Nsight Compute."""

import argparse
import csv
import json
import re
from pathlib import Path


def scaled_number(value, unit):
    number = float(str(value).replace(",", "").strip())
    prefix = unit.strip().lower()
    factors = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
               "byte/s": 1, "kbyte/s": 1e3, "mbyte/s": 1e6, "gbyte/s": 1e9}
    return number * factors.get(prefix, 1)


def parse_metrics(path):
    metrics = {}
    rows = list(csv.reader(path.read_text(errors="replace").splitlines()))
    header = None
    for row in rows:
        if "Metric Name" in row and "Metric Value" in row:
            header = {name: i for i, name in enumerate(row)}
            continue
        if not header:
            continue
        need = max(header.get("Metric Name", 0), header.get("Metric Value", 0), header.get("Metric Unit", 0))
        if len(row) <= need:
            continue
        name = row[header["Metric Name"]].strip()
        if not name or name == "Metric Name":
            continue
        value = row[header["Metric Value"]]
        unit = row[header.get("Metric Unit", 0)] if "Metric Unit" in header else ""
        try:
            metrics[name] = scaled_number(value, unit)
        except ValueError:
            continue
    return metrics


def measured_dram_bytes(metrics):
    read_patterns = (r"dram__bytes_read\.sum", r"dram.*read.*bytes")
    write_patterns = (r"dram__bytes_write\.sum", r"dram.*write.*bytes")
    def find(patterns):
        for pattern in patterns:
            for name, value in metrics.items():
                if re.search(pattern, name, re.I):
                    return value
        return 0.0
    return find(read_patterns) + find(write_patterns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = []
    for csv_path in sorted(args.input_dir.glob("*.ncu.csv")):
        case = csv_path.name.removesuffix(".ncu.csv")
        bench_path = args.input_dir / "benchmarks" / f"{case}.json"
        if not bench_path.exists():
            raise SystemExit(f"Missing benchmark metadata: {bench_path}")
        bench = json.loads(bench_path.read_text())["results"][0]
        metrics = parse_metrics(csv_path)
        dram_bytes = measured_dram_bytes(metrics)
        bench["ncu_csv"] = str(csv_path)
        bench["measured_dram_bytes"] = dram_bytes or None
        bench["arithmetic_intensity_measured"] = bench["flops"] / dram_bytes if dram_bytes else None
        points.append(bench)
    if not points:
        raise SystemExit(f"No *.ncu.csv files found under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "points": points}, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
