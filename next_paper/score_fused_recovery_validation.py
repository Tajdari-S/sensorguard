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
    controls = rows[rows.target == 0]
    fused = rows[rows.family == "fused_update"]
    adamw = rows[rows.family == "adamw"]
    negative_hours = sum(
        float(run["duration_s"]) / 3600
        for run in plan["runs"] if int(run["target"]) == 0
    )
    confusion = pd.DataFrame([{
        "artifact_sha256": digest,
        "fused_runs": len(fused), "fused_detected": int(fused.alert.sum()),
        "adamw_runs": len(adamw), "adamw_detected": int(adamw.alert.sum()),
        "matched_control_runs": len(controls),
        "matched_control_false_alerts": int(controls.alert.sum()),
        "matched_control_negative_gpu_hours": negative_hours,
        "false_alerts_per_gpu_hour": float(controls.alert.sum()) / negative_hours,
        "acceptable_recovery": bool(
            fused.alert.sum() == len(fused) and controls.alert.sum() == 0),
    }])
    node_status = pd.read_csv(args.node_root / "status_node.csv").set_index("run_id")
    pico_status = pd.read_csv(args.pico_root / "status_verifier.csv").set_index("run_id")
    audit_rows = []
    for run in plan["runs"]:
        run_id = run["run_id"]
        meta = json.loads((args.pico_root / run_id / "pico_u0_meta.json").read_text())
        records = [
            json.loads(line.removeprefix("useful_work "))
            for line in (args.node_root / run_id / "workload.stdout").read_text().splitlines()
            if line.startswith("useful_work ")
        ]
        workload = records[0] if len(records) == 1 else {}
        audit_rows.append({
            "run_id": run_id, "family": run["mode"], "variant": run["variant"],
            "target": int(run["target"]),
            "node_return_code": int(node_status.loc[run_id, "return_code"]),
            "verifier_return_code": int(pico_status.loc[run_id, "return_code"]),
            "scope_serial": meta["serial"], "scope_samples": int(meta["samples"]),
            "scope_overflow_flags": int(meta["overflow_flags"]),
            "scope_clipping_fraction_a": float(meta["clipping_fraction_a"]),
            "scheduled_start_error_ms": 1000 * abs(
                float(workload.get("start_epoch_s", 0)) - float(run["start_epoch_s"])),
            "meaningful_optimization_progress": workload.get(
                "meaningful_optimization_progress")
                if run["mode"] in {"fused_update", "adamw"} else None,
        })
    audit = pd.DataFrame(audit_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "fresh_recovery_run_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "fresh_recovery_family_summary.csv", index=False)
    confusion.to_csv(args.output_dir / "fresh_recovery_confusion.csv", index=False)
    audit.to_csv(args.output_dir / "fresh_recovery_collection_audit.csv", index=False)
    (args.output_dir / "fresh_recovery_validation_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
