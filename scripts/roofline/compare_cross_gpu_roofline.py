#!/usr/bin/env python3
"""Aggregate matching RTX/H200 bridge cases into normalized roofline pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {
    "case_id", "suite", "platform", "repetition", "mode", "gap_ms",
    "arithmetic_intensity", "wall_tflops", "peak_tflops", "peak_gbps",
    "normalized_arithmetic_intensity", "normalized_wall_throughput",
}


def build_table(frames: list[pd.DataFrame], minimum_repetitions: int = 3) -> pd.DataFrame:
    data = pd.concat(frames, ignore_index=True)
    missing = sorted(REQUIRED - set(data.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    data = data[data["suite"] == "cross_gpu_bridge"].copy()
    if data.empty:
        raise ValueError("no cross_gpu_bridge rows found")
    if data[["peak_tflops", "peak_gbps"]].isna().any().any():
        raise ValueError("normalized comparison requires measured peak TFLOP/s and GB/s")
    if minimum_repetitions < 1:
        raise ValueError("minimum_repetitions must be positive")
    counts = data.groupby("case_id")["platform"].nunique()
    incomplete = sorted(counts[counts < 2].index)
    if incomplete:
        raise ValueError(f"bridge cases missing a second platform: {incomplete}")
    repetitions = data.groupby(["case_id", "platform"])["repetition"].nunique()
    insufficient = repetitions[repetitions < minimum_repetitions]
    if not insufficient.empty:
        details = {
            f"{case_id}/{platform}": int(count)
            for (case_id, platform), count in insufficient.items()
        }
        raise ValueError(
            f"fewer than {minimum_repetitions} independent repetitions: {details}"
        )
    aggregate = data.groupby(
        ["case_id", "platform", "mode", "gap_ms"], as_index=False
    ).agg(
        repetitions=("repetition", "nunique"),
        arithmetic_intensity_median=("arithmetic_intensity", "median"),
        wall_tflops_median=("wall_tflops", "median"),
        normalized_ai_median=("normalized_arithmetic_intensity", "median"),
        normalized_throughput_median=("normalized_wall_throughput", "median"),
    )
    aggregate.insert(0, "pair_id", aggregate["case_id"])
    return aggregate.sort_values(["case_id", "platform"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/paper/cross_gpu_roofline_link.csv"))
    parser.add_argument("--minimum-repetitions", type=int, default=3)
    args = parser.parse_args()
    table = build_table(
        [pd.read_csv(path) for path in args.input], args.minimum_repetitions
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
