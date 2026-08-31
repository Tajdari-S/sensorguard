#!/usr/bin/env python3
"""Score the sealed fused-update corpus with pre-frozen model artifacts."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_synchronized_physical import META, alert, build  # noqa: E402


MODELS = {
    "NVML": "frozen_nvml.joblib",
    "Electrical": "frozen_electrical.joblib",
    "NVML + electrical": "frozen_nvml_plus_electrical.joblib",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--pico-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    electrical, nvml = build(plan, args.node_root, args.pico_root)
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {"NVML": nvml, "Electrical": electrical, "NVML + electrical": fusion}
    rows = []
    for modality, data in datasets.items():
        frozen = joblib.load(args.frozen_dir / MODELS[modality])
        model, features = frozen["model"], frozen["features"]
        probabilities = model.predict_proba(data[features].fillna(0))[:, list(model.classes_).index(1)]
        scored = data.copy()
        scored["probability"] = probabilities
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "sealed_run_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "sealed_family_metrics.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
