#!/usr/bin/env python3
"""Score hard families and run calibrated fusion, health, and privacy analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_synchronized_physical import META, THRESHOLD, alert  # noqa: E402


MODEL_FILES = {
    "NVML": "frozen_nvml.joblib",
    "Electrical": "frozen_electrical.joblib",
    "NVML + electrical": "frozen_nvml_plus_electrical.joblib",
}


def frozen_probabilities(data: pd.DataFrame, artifact: Path) -> pd.DataFrame:
    frozen = joblib.load(artifact)
    model, features = frozen["model"], frozen["features"]
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise RuntimeError(f"missing frozen features for {artifact.name}: {missing}")
    probabilities = model.predict_proba(data[features].fillna(0))[
        :, list(model.classes_).index(1)
    ]
    return data[list(META)].assign(probability=probabilities)


def run_predictions(windows: pd.DataFrame, modality: str) -> pd.DataFrame:
    rows = []
    for run_id, run in windows.groupby("run_id"):
        run = run.sort_values("window_index")
        detected, tta = alert(run.probability.to_numpy())
        rows.append({
            "modality": modality,
            "run_id": run_id,
            "family": run.family.iloc[0],
            "target": int(run.target.iloc[0]),
            "alert": int(detected),
            "time_to_alert_s": tta,
            "max_probability": float(run.probability.max()),
            "mean_probability": float(run.probability.mean()),
        })
    return pd.DataFrame(rows)


def metric_tables(predictions: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = {
        run["mode"]: run.get("prior_nvml_leave_family_out_detection")
        for run in plan["runs"]
    }
    families = []
    for (modality, family), group in predictions.groupby(["modality", "family"], sort=False):
        families.append({
            "modality": modality,
            "family": family,
            "target": int(group.target.iloc[0]),
            "runs": len(group),
            "alerts": int(group.alert.sum()),
            "detection_or_false_alert_rate": float(group.alert.mean()),
            "prior_multigpu_nvml_window_detection_rate": prior.get(family),
            "median_time_to_alert_s": (
                float(group.time_to_alert_s.dropna().median()) if group.alert.any() else None
            ),
        })
    overall = []
    for modality, group in predictions.groupby("modality", sort=False):
        positives, negatives = group[group.target == 1], group[group.target == 0]
        family_rates = positives.groupby("family").alert.mean()
        negative_hours = len(negatives) * 300 / 3600
        detected = positives.loc[positives.alert == 1, "time_to_alert_s"].dropna()
        overall.append({
            "modality": modality,
            "runs": len(group),
            "tp": int(positives.alert.sum()),
            "fn": int((1 - positives.alert).sum()),
            "fp": int(negatives.alert.sum()),
            "tn": int((1 - negatives.alert).sum()),
            "overall_detection_rate": float(positives.alert.mean()),
            "worst_family_detection_rate": float(family_rates.min()),
            "false_alerts_per_gpu_hour": float(negatives.alert.sum() / negative_hours),
            "median_time_to_alert_s": float(detected.median()) if len(detected) else None,
            "threshold": THRESHOLD,
            "run_rule": "3-of-5",
        })
    return pd.DataFrame(families), pd.DataFrame(overall)


def calibrated_late_fusion(dev_path: Path, hard_modal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dev = pd.read_csv(dev_path)
    keys = ["run_id", "family", "target", "window_index", "window_end_s"]
    dev_n = dev[dev.modality == "NVML"][keys + ["probability"]].rename(
        columns={"probability": "nvml_probability"}
    )
    dev_e = dev[dev.modality == "Electrical"][keys + ["probability"]].rename(
        columns={"probability": "electrical_probability"}
    )
    train = dev_n.merge(dev_e, on=keys, validate="one_to_one")
    x = train[["nvml_probability", "electrical_probability"]].to_numpy()
    y = train.target.to_numpy()

    oof = np.zeros(len(train), dtype=float)
    for family in sorted(train.family.unique()):
        test_mask = train.family.eq(family).to_numpy()
        model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000)
        model.fit(x[~test_mask], y[~test_mask])
        oof[test_mask] = model.predict_proba(x[test_mask])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    fusion = LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000)
    fusion.fit(x, y)

    hard = hard_modal.pivot_table(
        index=keys, columns="modality", values="probability", aggfunc="first"
    ).reset_index()
    hard_x = hard[["NVML", "Electrical"]].to_numpy()
    raw = fusion.predict_proba(hard_x)[:, 1]
    hard["probability"] = calibrator.predict(raw)
    scored = hard[keys + ["probability"]]
    metadata = {
        "method": "logistic late fusion with isotonic calibration",
        "calibration_data": "development leave-one-family-out modality probabilities only",
        "hard_family_labels_used_for_fit_or_calibration": False,
        "threshold": THRESHOLD,
        "run_rule": "3-of-5",
        "coefficients": fusion.coef_.tolist(),
        "intercept": fusion.intercept_.tolist(),
        "development_oof_brier": float(np.mean((calibrator.predict(oof) - y) ** 2)),
    }
    return scored, metadata


def health_evaluation(dev_path: Path, hard_modal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dev = pd.read_csv(dev_path)
    keys = ["run_id", "family", "target", "window_index", "window_end_s"]
    dn = dev[dev.modality == "NVML"][keys + ["probability"]].rename(
        columns={"probability": "nvml"}
    )
    de = dev[dev.modality == "Electrical"][keys + ["probability"]].rename(
        columns={"probability": "current"}
    )
    dev_pair = dn.merge(de, on=keys, validate="one_to_one")
    dev_disagreement = dev_pair.groupby("run_id").apply(
        lambda run: float(np.mean(np.abs(run.nvml - run.current)))
    )
    disagreement_threshold = float(dev_disagreement.max() + 1e-9)

    hard = hard_modal.pivot_table(
        index=keys, columns="modality", values="probability", aggfunc="first"
    ).reset_index().rename(columns={"NVML": "nvml", "Electrical": "current"})
    run_arrays = {
        run_id: run.sort_values("window_index").copy()
        for run_id, run in hard.groupby("run_id")
    }
    control = next(run for run in run_arrays.values() if int(run.target.iloc[0]) == 0)

    def resized(values: np.ndarray, size: int) -> np.ndarray:
        if len(values) == size:
            return values.copy()
        x_old = np.linspace(0, 1, len(values))
        return np.interp(np.linspace(0, 1, size), x_old, values)

    def health_alert(frame: pd.DataFrame) -> tuple[bool, str]:
        if frame[["nvml", "current"]].isna().any().any():
            return True, "missing"
        if frame.nvml.std(ddof=0) < 1e-8 or frame.current.std(ddof=0) < 1e-8:
            return True, "frozen"
        disagreement = float(np.mean(np.abs(frame.nvml - frame.current)))
        if disagreement > disagreement_threshold:
            return True, "cross_modal_disagreement"
        return False, "healthy"

    scenario_rows = []
    names = [
        "no_fault", "nvml_freeze", "current_freeze", "nvml_drop", "current_drop",
        "nvml_bias_scale", "nvml_time_shift", "nvml_replay_control",
        "current_channel_swap",
    ]
    ordered_ids = sorted(run_arrays)
    for index, run_id in enumerate(ordered_ids):
        original = run_arrays[run_id]
        swap = run_arrays[ordered_ids[(index + 1) % len(ordered_ids)]]
        for scenario in names:
            injected = original.copy()
            if scenario == "nvml_freeze":
                injected["nvml"] = injected.nvml.iloc[0]
            elif scenario == "current_freeze":
                injected["current"] = injected.current.iloc[0]
            elif scenario == "nvml_drop":
                injected["nvml"] = np.nan
            elif scenario == "current_drop":
                injected["current"] = np.nan
            elif scenario == "nvml_bias_scale":
                injected["nvml"] = np.clip(0.25 + 0.5 * injected.nvml, 0, 1)
            elif scenario == "nvml_time_shift":
                injected["nvml"] = np.roll(injected.nvml.to_numpy(), 3)
            elif scenario == "nvml_replay_control":
                injected["nvml"] = resized(control.nvml.to_numpy(), len(injected))
            elif scenario == "current_channel_swap":
                injected["current"] = resized(swap.current.to_numpy(), len(injected))
            detected, reason = health_alert(injected)
            scenario_rows.append({
                "run_id": run_id,
                "family": original.family.iloc[0],
                "target": int(original.target.iloc[0]),
                "scenario": scenario,
                "injected_fault": int(scenario != "no_fault"),
                "health_alert": int(detected),
                "health_reason": reason,
            })
    scenarios = pd.DataFrame(scenario_rows)
    summary = scenarios.groupby(["scenario", "injected_fault"], as_index=False).agg(
        runs=("run_id", "count"), health_alerts=("health_alert", "sum")
    )
    summary["health_alert_rate"] = summary.health_alerts / summary.runs
    contract = {
        "evaluation": "simulated probability-trace injection; no physical cable manipulation",
        "development_run_disagreement_threshold": disagreement_threshold,
        "missing_rule": "flag any missing modality probability",
        "freeze_rule": "flag within-run probability standard deviation below 1e-8",
        "consistency_rule": "flag mean absolute modality disagreement above the maximum development-run value",
    }
    return scenarios, summary, contract


def privacy_evaluation(dev_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(dev_path)
    electrical = dev[dev.modality == "Electrical"].copy()
    excluded = set(META) | {"modality", "probability"}
    features = sorted(set(electrical.columns) - excluded)
    run_level = electrical.groupby(["run_id", "family"], as_index=False)[features].mean()
    run_level["repetition"] = run_level.run_id.str.extract(r"_r(\d+)$")[0].astype(int)
    folds = []
    for repetition in sorted(run_level.repetition.unique()):
        train = run_level[run_level.repetition != repetition]
        test = run_level[run_level.repetition == repetition]
        model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ).fit(train[features].fillna(0), train.family)
        predicted = model.predict(test[features].fillna(0))
        folds.append({
            "held_out_repetition": int(repetition),
            "training_runs": len(train),
            "test_runs": len(test),
            "families": int(run_level.family.nunique()),
            "application_identity_accuracy": float(accuracy_score(test.family, predicted)),
            "application_identity_macro_f1": float(f1_score(test.family, predicted, average="macro", zero_division=0)),
        })
    fold_frame = pd.DataFrame(folds)
    summary = pd.DataFrame([{
        "evaluation": "GPU-current application-family identity leakage",
        "runs": len(run_level),
        "families": int(run_level.family.nunique()),
        "split": "leave-one-repetition-out; run-level features",
        "mean_accuracy": float(fold_frame.application_identity_accuracy.mean()),
        "mean_macro_f1": float(fold_frame.application_identity_macro_f1.mean()),
        "interpretation": "higher accuracy means greater workload-identity leakage",
    }])
    return fold_frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--nvml-features", type=Path, required=True)
    parser.add_argument("--electrical-features", type=Path, required=True)
    parser.add_argument("--development-window-predictions", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    nvml = pd.read_csv(args.nvml_features)
    electrical = pd.read_csv(args.electrical_features)
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {"NVML": nvml, "Electrical": electrical, "NVML + electrical": fusion}
    modal_windows, modal_runs = [], []
    for modality, data in datasets.items():
        scored = frozen_probabilities(data, args.frozen_dir / MODEL_FILES[modality])
        scored["modality"] = modality
        modal_windows.append(scored)
        modal_runs.append(run_predictions(scored, modality))
    hard_modal = pd.concat(modal_windows, ignore_index=True)

    late_windows, late_metadata = calibrated_late_fusion(
        args.development_window_predictions, hard_modal
    )
    late_windows["modality"] = "Calibrated late fusion"
    all_windows = pd.concat([hard_modal, late_windows], ignore_index=True)
    all_runs = pd.concat([
        *modal_runs,
        run_predictions(late_windows, "Calibrated late fusion"),
    ], ignore_index=True)
    family_metrics, overall_metrics = metric_tables(all_runs, plan)
    health_runs, health_summary, health_contract = health_evaluation(
        args.development_window_predictions, hard_modal
    )
    privacy_folds, privacy_summary = privacy_evaluation(
        args.development_window_predictions
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(args.output_dir / "hard_family_run_predictions.csv", index=False)
    family_metrics.to_csv(args.output_dir / "hard_family_metrics.csv", index=False)
    overall_metrics.to_csv(args.output_dir / "hard_family_overall.csv", index=False)
    health_runs.to_csv(args.output_dir / "simulated_health_run_results.csv", index=False)
    health_summary.to_csv(args.output_dir / "simulated_health_scenario_metrics.csv", index=False)
    privacy_folds.to_csv(args.output_dir / "privacy_folds.csv", index=False)
    privacy_summary.to_csv(args.output_dir / "privacy_summary.csv", index=False)
    (args.output_dir / "late_fusion_contract.json").write_text(
        json.dumps(late_metadata, indent=2) + "\n"
    )
    (args.output_dir / "simulated_health_contract.json").write_text(
        json.dumps(health_contract, indent=2) + "\n"
    )
    manifest = {
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "nvml_features_sha256": hashlib.sha256(args.nvml_features.read_bytes()).hexdigest(),
        "electrical_features_sha256": hashlib.sha256(args.electrical_features.read_bytes()).hexdigest(),
        "runs": int(all_runs.run_id.nunique()),
        "complete": True,
    }
    (args.output_dir / "READY.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        rendered = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for record in frame.fillna("").astype(str).to_dict("records"):
            rendered.append("| " + " | ".join(record[column].replace("|", "\\|") for column in columns) + " |")
        return "\n".join(rendered)

    lines = [
        "# Hard-family architecture tests", "",
        "All results use 30-second causal windows, 15-second stride, probability threshold 0.85, and the fixed 3-of-5 run rule.", "",
        "## Detector comparison", "",
        markdown(overall_metrics), "",
        "## Simulated sensor-health and replay tests", "",
        "These are probability-trace injections, not physical cable manipulations.", "",
        markdown(health_summary), "",
        "## Privacy", "",
        markdown(privacy_summary), "",
    ]
    (args.output_dir / "TESTS_AND_RESULTS.md").write_text("\n".join(lines))
    print(overall_metrics.to_string(index=False))
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
