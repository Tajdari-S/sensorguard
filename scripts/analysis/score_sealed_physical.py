#!/usr/bin/env python3
"""Score a fused-update confirmation with hash-verified frozen artifacts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_synchronized_physical import META, THRESHOLD, alert, build  # noqa: E402


MODELS = {
    "NVML": "frozen_nvml.joblib",
    "Electrical": "frozen_electrical.joblib",
    "NVML + electrical": "frozen_nvml_plus_electrical.joblib",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-root", type=Path)
    parser.add_argument("--pico-root", type=Path)
    parser.add_argument("--nvml-features", type=Path)
    parser.add_argument("--electrical-features", type=Path)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    compact = args.nvml_features is not None or args.electrical_features is not None
    raw = args.node_root is not None or args.pico_root is not None
    if compact == raw:
        parser.error(
            "provide either both compact feature tables or both raw roots"
        )
    if compact:
        if args.nvml_features is None or args.electrical_features is None:
            parser.error("both compact feature tables are required")
        nvml = pd.read_csv(args.nvml_features)
        electrical = pd.read_csv(args.electrical_features)
    else:
        if args.node_root is None or args.pico_root is None:
            parser.error("both raw roots are required")
        electrical, nvml = build(plan, args.node_root, args.pico_root)
    expected = {
        record["modality"]: record["sha256"]
        for record in json.loads(args.freeze_manifest.read_text())["models"]
    }
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {"NVML": nvml, "Electrical": electrical, "NVML + electrical": fusion}
    rows = []
    window_rows = []
    for modality, data in datasets.items():
        artifact_path = args.frozen_dir / MODELS[modality]
        observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if observed != expected[modality]:
            raise RuntimeError(f"artifact hash mismatch for {modality}")
        frozen = joblib.load(artifact_path)
        if (
            float(frozen.get("threshold", THRESHOLD)) != THRESHOLD
            or frozen.get("run_rule") != "3-of-5"
        ):
            raise RuntimeError(f"frozen decision contract mismatch for {modality}")
        model, features = frozen["model"], frozen["features"]
        probabilities = model.predict_proba(data[features].fillna(0))[:, list(model.classes_).index(1)]
        scored = data.copy()
        scored["probability"] = probabilities
        window_rows.extend(scored[["run_id", "family", "target", "window_index",
                                   "window_end_s", "probability"]].assign(
                                       modality=modality).to_dict("records"))
        for run_id, run in scored.groupby("run_id"):
            run = run.sort_values("window_index")
            detected, tta = alert(run["probability"].to_numpy())
            rows.append({
                "modality": modality, "run_id": run_id, "family": run["family"].iloc[0],
                "target": int(run["target"].iloc[0]), "alert": int(detected),
                "time_to_alert_s": tta, "max_probability": float(run["probability"].max()),
                "mean_probability": float(run["probability"].mean()),
            })
    predictions = pd.DataFrame(rows)
    summaries = []
    for modality, group in predictions.groupby("modality", sort=False):
        for family, runs in group.groupby("family"):
            summaries.append({"modality": modality, "family": family, "runs": len(runs),
                              "target": int(runs.target.iloc[0]), "alerts": int(runs.alert.sum()),
                              "detection_rate": float(runs.alert.mean()),
                              "median_time_to_alert_s": runs.time_to_alert_s.dropna().median()
                              if runs.alert.any() else None})
    summary = pd.DataFrame(summaries)
    positives = predictions[predictions.target == 1]
    negatives = predictions[predictions.target == 0]
    overall = []
    for modality in datasets:
        pos = positives[positives.modality == modality]
        neg = negatives[negatives.modality == modality]
        detected = pos.loc[pos.alert == 1, "time_to_alert_s"].dropna()
        negative_hours = sum(
            float(record["duration_s"])
            for record in plan["runs"]
            if int(record["target"]) == 0
        ) / 3600
        family_rates = pos.groupby("family").alert.mean()
        overall.append({
            "modality": modality,
            "tp": int(pos.alert.sum()),
            "fn": int((1 - pos.alert).sum()),
            "fp": int(neg.alert.sum()),
            "tn": int((1 - neg.alert).sum()),
            "overall_detection_rate": float(pos.alert.mean()),
            "worst_family_detection_rate": float(family_rates.min()),
            "false_alerts_per_gpu_hour": (
                float(neg.alert.sum() / negative_hours)
                if negative_hours else None
            ),
            "median_time_to_alert_s": (
                float(detected.median()) if len(detected) else None
            ),
            "threshold": THRESHOLD,
            "run_rule": "3-of-5",
        })
    overall = pd.DataFrame(overall)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "sealed_run_predictions.csv", index=False)
    pd.DataFrame(window_rows).to_csv(args.output_dir / "sealed_window_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "sealed_family_metrics.csv", index=False)
    overall.to_csv(args.output_dir / "sealed_overall_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
