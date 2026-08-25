#!/usr/bin/env python3
"""Combine benchmark metadata with DRAM-byte metrics exported by Nsight Compute.

The NCU details-page CSV contains one row per (kernel launch, metric). The
benchmark kernel is launched warmup+iterations times, alongside incidental
initialization kernels (tensor fills). We group rows into launches, keep the
dominant kernel (the most frequently launched), and use its median per-launch
DRAM read+write bytes — matching the per-iteration `flops` metadata.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def scaled_number(value, unit):
    number = float(str(value).replace(",", "").strip())
    prefix = unit.strip().lower()
    factors = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
               "byte/s": 1, "kbyte/s": 1e3, "mbyte/s": 1e6, "gbyte/s": 1e9}
    return number * factors.get(prefix, 1)


def parse_launches(path):
    """Return a list of {kernel, metrics{name: value}} — one per kernel launch."""
    rows = list(csv.reader(path.read_text(errors="replace").splitlines()))
    header = None
    launches = {}
    for row in rows:
        if "Metric Name" in row and "Metric Value" in row:
            header = {name: i for i, name in enumerate(row)}
            continue
        if not header:
            continue
        need = max(header.get(c, 0) for c in ("ID", "Kernel Name", "Metric Name", "Metric Value", "Metric Unit"))
        if len(row) <= need:
            continue
        launch_id = row[header.get("ID", 0)].strip()
        kernel = row[header.get("Kernel Name", 0)].strip()
        name = row[header["Metric Name"]].strip()
        if not name or name == "Metric Name":
            continue
        unit = row[header["Metric Unit"]] if "Metric Unit" in header else ""
        try:
            value = scaled_number(row[header["Metric Value"]], unit)
        except ValueError:
            continue
        entry = launches.setdefault(launch_id, {"kernel": kernel, "metrics": {}})
        entry["metrics"][name] = value
    return list(launches.values())


def launch_dram_bytes(metrics):
    read = next((v for n, v in metrics.items() if n.startswith("dram__bytes_read.sum")), None)
    write = next((v for n, v in metrics.items() if n.startswith("dram__bytes_write.sum")), None)
    if read is None and write is None:
        return None
    return (read or 0.0) + (write or 0.0)


def dominant_kernel_bytes(launches):
    """Median per-launch DRAM bytes of the most frequently launched kernel."""
    usable = [(l["kernel"], launch_dram_bytes(l["metrics"])) for l in launches]
    usable = [(k, b) for k, b in usable if b is not None]
    if not usable:
        return None, None
    counts = Counter(k for k, _ in usable)
    kernel = max(counts, key=lambda k: (counts[k], statistics.median(b for kk, b in usable if kk == k)))
    per_launch = [b for k, b in usable if k == kernel]
    return kernel, statistics.median(per_launch)


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
        launches = parse_launches(csv_path)
        kernel, dram_bytes = dominant_kernel_bytes(launches)
        bench["ncu_csv"] = str(csv_path)
        bench["ncu_kernel"] = kernel
        bench["ncu_launches"] = len(launches)
        bench["measured_dram_bytes"] = dram_bytes
        bench["arithmetic_intensity_measured"] = (
            bench["flops"] / dram_bytes if dram_bytes else None)
        points.append(bench)
    if not points:
        raise SystemExit(f"No *.ncu.csv files found under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "points": points}, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
