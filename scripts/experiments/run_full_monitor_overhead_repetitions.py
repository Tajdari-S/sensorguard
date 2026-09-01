#!/usr/bin/env python3
"""Measure useful-work overhead with NVML, DCGM, and Pico streaming together."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOGGER = ROOT / "scripts" / "loggers"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-s", type=float, default=10)
    parser.add_argument("--interval-s", type=float, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--pico-python", default="/home/felkru/picoenv/bin/python")
    parser.add_argument("--sample-interval-us", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "final_paper_completion"
        / "full_monitor_overhead",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device(args.device)
    a = torch.randn(8192, 8192, device=device)
    b = torch.randn(8192, 8192, device=device)

    def throughput(seconds: float) -> float:
        end = time.monotonic() + seconds
        count = 0
        while time.monotonic() < end:
            a @ b
            torch.cuda.synchronize(device)
            count += 1
        return count / seconds

    def monitored(rep_dir: Path) -> tuple[float, dict]:
        capture_s = args.interval_s + 5
        nvml = subprocess.Popen([
            sys.executable,
            str(LOGGER / "nvml_logger.py"),
            "--output",
            str(rep_dir / "nvml.csv"),
            "--gpus",
            args.gpu,
        ])
        dcgm = subprocess.Popen([
            sys.executable,
            str(LOGGER / "dcgm_logger.py"),
            "--output",
            str(rep_dir / "dcgm.tsv"),
            "--gpus",
            args.gpu,
        ])
        pico = subprocess.Popen([
            args.pico_python,
            str(LOGGER / "pico_logger.py"),
            "--parallel",
            "--duration-s",
            str(capture_s),
            "--sample-interval-us",
            str(args.sample_interval_us),
            "--output-prefix",
            str(rep_dir / "pico"),
        ])
        time.sleep(2)
        rate = throughput(args.interval_s)
        for process in (nvml, dcgm):
            process.terminate()
            process.wait(timeout=15)
        pico.wait(timeout=120)
        if pico.returncode:
            raise RuntimeError(f"Pico logger exited {pico.returncode}")
        metas = [
            json.loads(path.read_text())
            for path in sorted(rep_dir.glob("pico_u*_meta.json"))
        ]
        if len(metas) != 6:
            raise RuntimeError(f"expected six PicoScopes, observed {len(metas)}")
        if any(int(meta["overflow_flags"]) for meta in metas):
            raise RuntimeError("PicoScope overflow occurred")
        metadata = {
            "pico_units": len(metas),
            "pico_channels": 2 * len(metas),
            "pico_samples_per_channel_mean": statistics.mean(
                int(meta["samples"]) for meta in metas
            ),
            "pico_overflow_units": sum(
                bool(meta["overflow_flags"]) for meta in metas
            ),
            "nvml_bytes": (rep_dir / "nvml.csv").stat().st_size,
            "dcgm_bytes": (rep_dir / "dcgm.tsv").stat().st_size,
        }
        return rate, metadata

    rows = []
    for repetition in range(1, args.repetitions + 1):
        rep_dir = args.output_dir / f"rep{repetition:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        throughput(args.warmup_s)
        order = "baseline-first" if repetition % 2 else "monitor-first"
        if order == "baseline-first":
            baseline = throughput(args.interval_s)
            monitor, metadata = monitored(rep_dir)
        else:
            monitor, metadata = monitored(rep_dir)
            baseline = throughput(args.interval_s)
        penalty = 100 * (baseline - monitor) / baseline
        rows.append({
            "repetition": repetition,
            "condition_order": order,
            "baseline_gemm_per_s": baseline,
            "full_monitor_gemm_per_s": monitor,
            "useful_work_penalty_pct": penalty,
            "runtime_multiplier": baseline / monitor,
            **metadata,
        })

    csv_path = args.output_dir / "full_monitor_overhead_repetitions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    penalties = [float(row["useful_work_penalty_pct"]) for row in rows]
    multipliers = [float(row["runtime_multiplier"]) for row in rows]
    mean_penalty = statistics.mean(penalties)
    stdev = statistics.stdev(penalties) if len(penalties) > 1 else 0.0
    t_critical = 2.776 if len(penalties) == 5 else None
    half_width = (
        t_critical * stdev / len(penalties) ** 0.5
        if t_critical is not None
        else None
    )
    summary = {
        "monitor": "NVML + DCGM + six PicoScopes streaming concurrently",
        "repetitions": len(rows),
        "condition_order": "alternating paired baseline/monitor",
        "workload": "steady 8192x8192 CUDA GEMM",
        "sample_interval_us": args.sample_interval_us,
        "mean_runtime_multiplier": statistics.mean(multipliers),
        "mean_useful_work_penalty_pct": mean_penalty,
        "ci95_useful_work_penalty_pct": (
            [mean_penalty - half_width, mean_penalty + half_width]
            if half_width is not None
            else None
        ),
        "all_six_scopes_present": all(row["pico_units"] == 6 for row in rows),
        "any_scope_overflow": any(row["pico_overflow_units"] for row in rows),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (
        args.output_dir / "full_monitor_overhead_summary.json"
    ).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
