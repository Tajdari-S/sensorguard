#!/usr/bin/env python3
"""Run and aggregate paired PicoScope monitor-overhead repetitions."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MEASURE = ROOT / "scripts" / "loggers" / "measure_overhead.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=90)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--pico-python", default="/home/felkru/picoenv/bin/python")
    parser.add_argument("--sample-interval-us", type=int, default=100)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="reuse existing repXX/result.json files")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "physical_overhead_20260830",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rep in range(1, args.repetitions + 1):
        rep_dir = args.output_dir / f"rep{rep:02d}"
        order = "baseline-first" if rep % 2 else "logger-first"
        output = rep_dir / "result.json"
        if not args.aggregate_only:
            subprocess.run(
                [
                    sys.executable, str(MEASURE), "--sensors", "pico",
                    "--device", args.device, "--gpus", args.gpus,
                    "--duration-s", str(args.duration_s),
                    "--pico-python", args.pico_python,
                    "--pico-sample-interval-us", str(args.sample_interval_us),
                    "--penalty-order", order, "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
        elif not output.exists():
            raise FileNotFoundError(f"missing completed repetition: {output}")
        payload = json.loads(output.read_text())
        result = payload["sensors"][0]
        if not result.get("available"):
            raise RuntimeError(f"PicoScope acquisition unavailable in repetition {rep}: {result}")
        if int(result.get("units", 0)) != 6:
            raise RuntimeError(f"expected six PicoScopes in repetition {rep}: {result}")
        rows.append({"repetition": rep, "condition_order": order, **result})

    csv_path = args.output_dir / "pico_overhead_repetitions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    penalties = [float(row["useful_work_penalty_pct"]) for row in rows]
    mean_penalty = statistics.mean(penalties)
    stdev_penalty = statistics.stdev(penalties) if len(penalties) > 1 else 0.0
    # Two-sided 95% Student-t critical value for the preregistered five reps.
    t_critical = 2.776 if len(penalties) == 5 else None
    ci_half_width = (
        t_critical * stdev_penalty / len(penalties) ** 0.5
        if t_critical is not None else None
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    summary = {
        "git_commit": git_commit,
        "repetitions": len(rows),
        "sample_interval_us": args.sample_interval_us,
        "mean_useful_work_penalty_pct": mean_penalty,
        "stdev_useful_work_penalty_pct": stdev_penalty,
        "ci95_useful_work_penalty_pct":
            [mean_penalty - ci_half_width, mean_penalty + ci_half_width]
            if ci_half_width is not None else None,
        "min_useful_work_penalty_pct": min(penalties),
        "max_useful_work_penalty_pct": max(penalties),
        "all_units_present": all(int(row["units"]) == 6 for row in rows),
        "any_overflow": any(int(row["overflow_units"]) for row in rows),
        "source_csv": str(csv_path.resolve().relative_to(ROOT)),
    }
    summary_path = args.output_dir / "pico_overhead_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
