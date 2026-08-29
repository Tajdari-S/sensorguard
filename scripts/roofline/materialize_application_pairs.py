#!/usr/bin/env python3
"""Materialize preregistered application roofline pairs without post-hoc matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_POINT_COLUMNS = {
    "case_id", "platform", "suite", "repetition", "mode",
    "arithmetic_intensity", "wall_tflops",
}
REQUIRED_PAIR_COLUMNS = {
    "pair_id", "training_case", "inference_case", "model_family", "freeze_note",
}


def materialize(points: pd.DataFrame, pairs: pd.DataFrame,
                minimum_repetitions: int = 3) -> pd.DataFrame:
    missing_points = sorted(REQUIRED_POINT_COLUMNS - set(points.columns))
    missing_pairs = sorted(REQUIRED_PAIR_COLUMNS - set(pairs.columns))
    if missing_points:
        raise ValueError(f"points CSV is missing columns: {missing_points}")
    if missing_pairs:
        raise ValueError(f"pair definition is missing columns: {missing_pairs}")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique")
    if minimum_repetitions < 1:
        raise ValueError("minimum_repetitions must be positive")

    selected = set(pairs["training_case"]) | set(pairs["inference_case"])
    available = set(points["case_id"])
    absent = sorted(selected - available)
    if absent:
        raise ValueError(f"preregistered cases are missing from points: {absent}")

    counts = points.groupby(["platform", "case_id"])["repetition"].nunique()
    insufficient = counts[counts < minimum_repetitions]
    relevant_insufficient = {
        f"{platform}/{case_id}": int(count)
        for (platform, case_id), count in insufficient.items() if case_id in selected
    }
    if relevant_insufficient:
        raise ValueError(
            f"fewer than {minimum_repetitions} independent repetitions: "
            f"{relevant_insufficient}"
        )

    numeric = [
        column for column in [
            "arithmetic_intensity", "active_tflops", "wall_tflops",
            "normalized_arithmetic_intensity", "normalized_active_throughput",
            "normalized_wall_throughput",
        ] if column in points.columns
    ]
    medians = points.groupby(["platform", "case_id"], as_index=False)[numeric].median()
    modes = points.groupby(["platform", "case_id"], as_index=False)["mode"].first()
    aggregates = medians.merge(modes, on=["platform", "case_id"], validate="one_to_one")

    rows: list[dict] = []
    for pair in pairs.to_dict("records"):
        for role, case_column in [("training", "training_case"),
                                  ("inference", "inference_case")]:
            case_id = pair[case_column]
            matches = aggregates[aggregates["case_id"] == case_id]
            for point in matches.to_dict("records"):
                rows.append({
                    "pair_id": pair["pair_id"],
                    "role": role,
                    "case_id": case_id,
                    "model_family": pair["model_family"],
                    "freeze_note": pair["freeze_note"],
                    **point,
                })
    result = pd.DataFrame(rows)
    expected = len(pairs) * 2 * points["platform"].nunique()
    if len(result) != expected:
        raise ValueError("a preregistered pair is not represented on every platform")
    return result.sort_values(["platform", "pair_id", "role"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--pairs", type=Path,
                        default=Path("configs/application_roofline_pairs.csv"))
    parser.add_argument("--minimum-repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("results/paper/matched_application_roofline.csv"))
    args = parser.parse_args()
    result = materialize(
        pd.read_csv(args.points), pd.read_csv(args.pairs), args.minimum_repetitions
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
