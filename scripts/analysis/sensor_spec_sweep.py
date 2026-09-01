#!/usr/bin/env python3
"""Find the lowest offline electrical acquisition spec preserving frozen decisions."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_synchronized_physical import (  # noqa: E402
    build,
    evaluate,
    physical_seconds,
    windows,
)


def electrical_from_plan(plan: dict, pico_root: Path, sample_hz: int,
                         bits: int) -> pd.DataFrame:
    frames = []
    for run in plan["runs"]:
        start = float(run["start_epoch_s"])
        end = start + float(run["duration_s"])
        seconds = physical_seconds(
            pico_root / run["run_id"], start, end, sample_hz, bits
        )
        frames.append(windows(
            seconds,
            [column for column in seconds.columns if column != "second"],
            "elec_",
            run["run_id"],
            run["mode"],
            int(run["target"]),
        ))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--node-root",
        type=Path,
        help="optional legacy timing source; omit to use frozen plan times",
    )
    parser.add_argument("--pico-root", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    baseline = pd.read_csv(args.baseline_predictions)
    baseline = baseline[baseline.modality == "Electrical"].set_index("run_id")["alert"].sort_index()
    rows = []
    for sample_hz in (10, 100, 1000, 10000):
        for bits in (8, 12, 16):
            if args.node_root:
                electrical, _ = build(
                    plan, args.node_root, args.pico_root, sample_hz, bits
                )
            else:
                electrical = electrical_from_plan(
                    plan, args.pico_root, sample_hz, bits
                )
            predictions, _ = evaluate(electrical, f"Electrical {sample_hz} Hz/{bits} bit")
            decisions = predictions.set_index("run_id")["alert"].sort_index()
            positives = predictions[predictions.target == 1]
            negatives = predictions[predictions.target == 0]
            exact = bool(decisions.equals(baseline))
            rows.append({
                "sample_hz": sample_hz, "bits": bits, "runs": len(predictions),
                "tp": int(positives.alert.sum()), "fn": int((1 - positives.alert).sum()),
                "fp": int(negatives.alert.sum()), "tn": int((1 - negatives.alert).sum()),
                "exactly_preserves_full_rate_run_decisions": exact,
            })
    result = pd.DataFrame(rows)
    eligible = result[result.exactly_preserves_full_rate_run_decisions]
    selected = None if eligible.empty else eligible.sort_values(["sample_hz", "bits"]).iloc[0].to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "sensor_spec_sweep.csv", index=False)
    (args.output_dir / "minimum_retained_sensor_spec.json").write_text(json.dumps({
        "selection_rule": "lowest sample rate, then lowest bit depth, exactly preserving every full-rate leave-one-family-out run decision",
        "selected": selected,
        "qualification": "offline decimation and quantization; hardware-in-loop confirmation is separate",
    }, indent=2) + "\n")
    print(result.to_string(index=False))
    print("selected", selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
