#!/usr/bin/env python3
"""Score synchronized NVML/current negatives with the frozen detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from analyze_synchronized_physical import META, THRESHOLD
from post_collection_architecture import (
    MODEL_FILES,
    calibrated_late_fusion,
    frozen_probabilities,
    run_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--nvml-features", type=Path, required=True)
    parser.add_argument("--electrical-features", type=Path, required=True)
    parser.add_argument("--development-window-predictions", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    expected = {
        record["modality"]: record["sha256"]
        for record in json.loads(args.freeze_manifest.read_text())["models"]
    }
    nvml = pd.read_csv(args.nvml_features)
    electrical = pd.read_csv(args.electrical_features)
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {
        "NVML": nvml,
        "Electrical": electrical,
        "NVML + electrical": fusion,
    }
    modal_windows, predictions = [], []
    for modality, data in datasets.items():
        artifact_path = args.frozen_dir / MODEL_FILES[modality]
        observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if observed != expected[modality]:
            raise RuntimeError(f"artifact hash mismatch for {modality}")
        artifact = joblib.load(artifact_path)
        if (
            float(artifact.get("threshold", THRESHOLD)) != THRESHOLD
            or artifact.get("run_rule") != "3-of-5"
        ):
            raise RuntimeError(f"frozen decision contract mismatch for {modality}")
        scored = frozen_probabilities(data, artifact_path)
        scored["modality"] = modality
        modal_windows.append(scored)
        predictions.append(run_predictions(scored, modality))

    hard_modal = pd.concat(modal_windows, ignore_index=True)
    late, contract = calibrated_late_fusion(
        args.development_window_predictions, hard_modal
    )
    late["modality"] = "Calibrated late fusion"
    predictions.append(run_predictions(late, "Calibrated late fusion"))
    runs = pd.concat(predictions, ignore_index=True)
    duration = {
        record["run_id"]: float(record["duration_s"])
        for record in plan["runs"]
    }
    runs["duration_s"] = runs.run_id.map(duration)
    if runs.duration_s.isna().any():
        raise RuntimeError("prediction contains a run absent from the plan")

    rows = []
    for modality, group in runs.groupby("modality", sort=False):
        hours = float(group.duration_s.sum() / 3600)
        false_alerts = int(group.alert.sum())
        rows.append({
            "modality": modality,
            "runs": len(group),
            "false_alert_runs": false_alerts,
            "true_negative_runs": int((1 - group.alert).sum()),
            "negative_gpu_hours": hours,
            "false_alerts_per_gpu_hour": false_alerts / hours,
            "threshold": THRESHOLD,
            "run_rule": "3-of-5",
        })
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(
        args.output_dir / "physical_negative_run_predictions.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "physical_negative_summary.csv", index=False
    )
    (
        args.output_dir / "physical_negative_late_fusion_contract.json"
    ).write_text(json.dumps(contract, indent=2) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
