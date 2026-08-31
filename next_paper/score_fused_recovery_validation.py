#!/usr/bin/env python3
"""Score fresh fused-update validation data with the frozen recovery model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_recovery_diagnostic import build, build_trend, summarize_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--pico-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    digest = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    if digest != plan["frozen_artifact_sha256"]:
        raise RuntimeError(f"artifact hash mismatch: {digest}")
    frozen = joblib.load(args.artifact)
    if "drift" in frozen["modality"]:
        electrical, _ = build_trend(plan, args.node_root, args.pico_root)
    else:
        electrical, _ = build(plan, args.node_root, args.pico_root)
    features = frozen["features"]
    probabilities = frozen["model"].predict_proba(
        electrical[features].fillna(0))[:, list(frozen["model"].classes_).index(1)]
    scored = electrical[["run_id", "family", "target", "window_index", "window_end_s"]].copy()
    scored["probability"] = probabilities
    rows = pd.DataFrame(summarize_runs(
        "fresh_recovery_validation", frozen["modality"], frozen["model_name"],
        scored, 45, frozen["threshold"], frozen["rule_windows"],
        frozen["rule_hits"], split="fresh_validation"))
    variant_by_run = {run["run_id"]: run["variant"] for run in plan["runs"]}
    rows["variant"] = rows.run_id.map(variant_by_run)
    summary_rows = []
    for family, group in rows.groupby("family"):
        summary_rows.append({
            "family": family, "target": int(group.target.iloc[0]),
            "runs": len(group), "alerts": int(group.alert.sum()),
            "detection_rate": float(group.alert.mean()),
            "median_time_to_alert_s": group.loc[group.alert == 1, "time_to_alert_s"].median()
                if group.alert.any() else None,
        })
    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "fresh_recovery_run_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "fresh_recovery_family_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
