#!/usr/bin/env python3
"""Train and freeze a post-sealed fused-update recovery detector.

The original sealed outcome remains immutable. This development script uses
the opened fused-update repetitions in leave-one-repetition-out predictions to
select a model and sequential rule. The resulting artifact must be evaluated
only on fresh workload configurations collected after its hash is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_recovery_diagnostic import (  # noqa: E402
    META, build, build_trend, choose_rule, fit_and_predict, models,
    repetition, summarize_runs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-plan", type=Path, required=True)
    parser.add_argument("--development-node-root", type=Path, required=True)
    parser.add_argument("--development-pico-root", type=Path, required=True)
    parser.add_argument("--opened-plan", type=Path, required=True)
    parser.add_argument("--opened-node-root", type=Path, required=True)
    parser.add_argument("--opened-pico-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dev_plan = json.loads(args.development_plan.read_text())
    opened_plan = json.loads(args.opened_plan.read_text())
    dev_e, dev_n = build(dev_plan, args.development_node_root, args.development_pico_root)
    opened_e, opened_n = build(opened_plan, args.opened_node_root, args.opened_pico_root)
    dev_et, dev_nt = build_trend(
        dev_plan, args.development_node_root, args.development_pico_root)
    opened_et, opened_nt = build_trend(
        opened_plan, args.opened_node_root, args.opened_pico_root)
    datasets = {
        "Electrical": (dev_e, opened_e),
        "NVML + electrical": (
            dev_n.merge(dev_e, on=list(META), validate="one_to_one"),
            opened_n.merge(opened_e, on=list(META), validate="one_to_one"),
        ),
        "Electrical + drift": (dev_et, opened_et),
        "NVML + electrical + drift": (
            dev_nt.merge(dev_et, on=list(META), validate="one_to_one"),
            opened_nt.merge(opened_et, on=list(META), validate="one_to_one"),
        ),
    }

    all_run_rows = []
    candidate_rows = []
    fitted_data = {}
    for modality, (development, opened) in datasets.items():
        opened = opened.copy()
        opened["repetition"] = opened.run_id.map(repetition)
        development = development.copy()
        development["repetition"] = development.run_id.map(repetition)
        fitted_data[modality] = pd.concat([development, opened], ignore_index=True)
        for model_name in models():
            fold_windows = []
            for held_out_rep in (1, 2, 3):
                fit = pd.concat(
                    [development, opened[opened.repetition != held_out_rep]],
                    ignore_index=True,
                )
                test = opened[opened.repetition == held_out_rep]
                model = models()[model_name]
                scored = fit_and_predict(model, fit, test)
                scored["held_out_repetition"] = held_out_rep
                fold_windows.append(scored)
            pooled = pd.concat(fold_windows, ignore_index=True)
            threshold, windows, hits, candidates = choose_rule(pooled)
            candidates.insert(0, "modality", modality)
            candidates.insert(1, "model", model_name)
            candidates["selected"] = (
                (candidates.threshold == threshold)
                & (candidates.rule_windows == windows)
                & (candidates.rule_hits == hits)
            )
            candidate_rows.append(candidates)
            all_run_rows.extend(summarize_runs(
                "leave_one_repetition_out", modality, model_name, pooled,
                0, threshold, windows, hits, split="development_oof"))

    runs = pd.DataFrame(all_run_rows)
    summaries = []
    for (modality, model_name), group in runs.groupby(["modality", "model"]):
        fused = group[group.family == "fused_update"]
        controls = group[group.target == 0]
        positives = group[group.target == 1]
        summaries.append({
            "modality": modality,
            "model": model_name,
            "fused_runs": len(fused),
            "fused_detected": int(fused.alert.sum()),
            "positive_runs": len(positives),
            "positives_detected": int(positives.alert.sum()),
            "control_runs": len(controls),
            "false_alerts": int(controls.alert.sum()),
            "threshold": float(group.threshold.iloc[0]),
            "rule_windows": int(group.rule_windows.iloc[0]),
            "rule_hits": int(group.rule_hits.iloc[0]),
            "median_fused_time_to_alert_s": fused.loc[
                fused.alert == 1, "time_to_alert_s"].median()
                if fused.alert.any() else None,
        })
    summary = pd.DataFrame(summaries)
    eligible = summary[
        (summary.fused_detected == summary.fused_runs)
        & (summary.positives_detected == summary.positive_runs)
        & (summary.false_alerts == 0)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no zero-FP recovery candidate detects every opened fused run")
    # Prefer the smallest external modality and a persistent rule. A 3-window
    # streak is less vulnerable to isolated benign spikes than a 1-window hit.
    eligible["modality_rank"] = eligible.modality.map({
        "Electrical": 0, "Electrical + drift": 1,
        "NVML + electrical": 2, "NVML + electrical + drift": 3,
    })
    eligible["model_rank"] = eligible.model.map(
        {"random_forest": 0, "extra_trees": 1, "logistic": 2})
    selected = eligible.sort_values(
        ["modality_rank", "rule_hits", "threshold", "model_rank"],
        ascending=[True, False, False, True],
    ).iloc[0]

    modality = str(selected.modality)
    model_name = str(selected.model)
    final_data = fitted_data[modality]
    features = sorted(set(final_data.columns) - META - {"repetition"})
    final_model = models()[model_name]
    final_model.fit(final_data[features].fillna(0), final_data.target)
    artifact = args.output_dir / "fused_recovery_frozen.joblib"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "features": features,
        "modality": modality,
        "model_name": model_name,
        "threshold": float(selected.threshold),
        "rule_windows": int(selected.rule_windows),
        "rule_hits": int(selected.rule_hits),
        "window_s": 30,
        "stride_s": 15,
        "status": "post_sealed_recovery_candidate_requires_fresh_validation",
    }
    joblib.dump(payload, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "artifact": str(artifact),
        "sha256": digest,
        "selection_protocol": "pooled leave-one-repetition-out on opened repetitions",
        "selected": {
            key: value for key, value in payload.items() if key != "model"
        } | {"fit_runs": int(final_data.run_id.nunique())},
        "required_next_step": (
            "Evaluate once on fresh, previously unused fused-update configurations "
            "and matched controls without changing this artifact or alert rule."
        ),
    }
    runs.to_csv(args.output_dir / "fused_recovery_oof_run_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "fused_recovery_oof_summary.csv", index=False)
    pd.concat(candidate_rows, ignore_index=True).to_csv(
        args.output_dir / "fused_recovery_oof_rule_search.csv", index=False)
    (args.output_dir / "fused_recovery_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
