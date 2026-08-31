#!/usr/bin/env python3
"""Causal CUSUM calibration and post-hoc evaluation on frozen RF scores.

Decision-rule parameters are selected using development repetitions 1--2.
Development repetition 3 is excluded from calibration. The prior fused-update
corpus is scored only after parameters are selected, but is labelled post-hoc
because it had already been opened by the fixed-rule analysis; it is not a new
sealed primary test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def cusum_first_alert(probabilities, drift: float, threshold: float) -> int | None:
    """Return the first causal alert index for one-sided score CUSUM."""
    cumulative = 0.0
    for index, probability in enumerate(np.asarray(probabilities, dtype=float)):
        cumulative = max(0.0, cumulative + float(probability) - drift)
        if cumulative + 1e-12 >= threshold:
            return index
    return None


def fixed_first_alert(probabilities, probability_threshold=0.85, k=3, n=5):
    hits = np.asarray(probabilities, dtype=float) >= probability_threshold
    for index in range(n - 1, len(hits)):
        if hits[index - n + 1:index + 1].sum() >= k:
            return index
    return None


def unit_checks() -> dict:
    checks = {
        "sustained_evidence": cusum_first_alert([0.7, 0.8, 0.9], 0.5, 0.8) == 2,
        "reset_after_negative_increment": cusum_first_alert([0.8, 0.1, 0.8, 0.8], 0.5, 0.6) == 3,
        "right_censored": cusum_first_alert([0.2, 0.3, 0.4], 0.5, 0.2) is None,
        "fixed_rule_reference": fixed_first_alert([0.9, 0.1, 0.9, 0.1, 0.9]) == 4,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def run_predictions(windows: pd.DataFrame, method: str,
                    drift: float | None = None, threshold: float | None = None) -> pd.DataFrame:
    rows = []
    for (modality, run_id), run in windows.groupby(["modality", "run_id"], sort=False):
        run = run.sort_values("window_index")
        probabilities = run["probability"].to_numpy(dtype=float)
        if method == "CUSUM":
            index = cusum_first_alert(probabilities, float(drift), float(threshold))
        else:
            index = fixed_first_alert(probabilities)
        rows.append({
            "modality": modality, "run_id": run_id, "family": run["family"].iloc[0],
            "target": int(run["target"].iloc[0]), "method": method,
            "alert": int(index is not None),
            "time_to_alert_s": None if index is None else float(run.iloc[index]["window_end_s"]),
            "right_censored": index is None,
            "censor_time_s": float(run["window_end_s"].max()),
            "drift": drift, "threshold": threshold,
        })
    return pd.DataFrame(rows)


def metrics(predictions: pd.DataFrame, dataset: str) -> list[dict]:
    rows = []
    for (modality, method), group in predictions.groupby(["modality", "method"], sort=False):
        positives, negatives = group[group.target == 1], group[group.target == 0]
        detected = positives.loc[positives.alert == 1, "time_to_alert_s"]
        family_rates = positives.groupby("family")["alert"].mean()
        rows.append({
            "dataset": dataset, "modality": modality, "method": method,
            "positive_runs": len(positives), "tp": int(positives.alert.sum()),
            "negative_runs": len(negatives), "fp": int(negatives.alert.sum()),
            "worst_family_detection_rate": float(family_rates.min()) if len(family_rates) else None,
            "median_time_to_alert_s": float(detected.median()) if len(detected) else None,
            "p95_time_to_alert_s": float(detected.quantile(0.95)) if len(detected) else None,
            "right_censored_positive_runs": int((positives.alert == 0).sum()),
        })
    return rows


def calibrate(modality_windows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    calibration = modality_windows[~modality_windows.run_id.str.endswith("r3")]
    fixed = run_predictions(calibration, "fixed_3of5")
    fp_budget = int(fixed.loc[fixed.target == 0, "alert"].sum())
    candidates = []
    for drift in np.round(np.arange(0.05, 0.81, 0.05), 2):
        for threshold in np.round(np.arange(0.1, 5.01, 0.1), 2):
            predicted = run_predictions(calibration, "CUSUM", drift, threshold)
            positives, negatives = predicted[predicted.target == 1], predicted[predicted.target == 0]
            fp = int(negatives.alert.sum())
            if fp > fp_budget:
                continue
            tta = positives.loc[positives.alert == 1, "time_to_alert_s"]
            candidates.append({
                "modality": modality_windows.modality.iloc[0], "drift": drift,
                "threshold": threshold, "fp_budget_runs": fp_budget,
                "calibration_tp": int(positives.alert.sum()), "calibration_fp": fp,
                "calibration_median_tta_s": float(tta.median()) if len(tta) else math.inf,
            })
    table = pd.DataFrame(candidates)
    if table.empty:
        raise RuntimeError("no CUSUM candidate satisfies the fixed-rule false-alert budget")
    chosen = table.sort_values(
        ["calibration_tp", "calibration_median_tta_s", "calibration_fp", "threshold", "drift"],
        ascending=[False, True, True, False, False],
    ).iloc[0].to_dict()
    table["selected"] = (
        (table.drift == chosen["drift"]) & (table.threshold == chosen["threshold"])
    )
    return chosen, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-windows", type=Path, required=True)
    parser.add_argument("--sealed-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checks = unit_checks()
    development = pd.read_csv(args.development_windows)
    development = development[["modality", "run_id", "family", "target", "window_index",
                               "window_end_s", "probability"]]
    sealed = pd.read_csv(args.sealed_windows)
    calibration_tables, configurations = [], {}
    all_predictions, all_metrics = [], []
    for modality in sorted(development.modality.unique()):
        dev_modality = development[development.modality == modality]
        chosen, candidates = calibrate(dev_modality)
        configurations[modality] = {
            key: (None if pd.isna(value) or (isinstance(value, float) and not math.isfinite(value))
                  else value)
            for key, value in chosen.items()
        }
        calibration_tables.append(candidates)
        validation = dev_modality[dev_modality.run_id.str.endswith("r3")]
        sealed_modality = sealed[sealed.modality == modality]
        for dataset_name, windows in (("development_rep3_validation", validation),
                                      ("prior_fused_posthoc", sealed_modality)):
            fixed = run_predictions(windows, "fixed_3of5")
            cusum = run_predictions(windows, "CUSUM", chosen["drift"], chosen["threshold"])
            combined = pd.concat([fixed, cusum], ignore_index=True)
            combined["dataset"] = dataset_name
            all_predictions.append(combined)
            all_metrics.extend(metrics(combined, dataset_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(calibration_tables, ignore_index=True).to_csv(
        args.output_dir / "cusum_calibration.csv", index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(args.output_dir / "cusum_run_predictions.csv", index=False)
    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv(args.output_dir / "sequential_comparison.csv", index=False)
    (args.output_dir / "cusum_implementation.json").write_text(json.dumps({
        "rule": "S_t=max(0,S_(t-1)+p_t-drift); alert at first S_t>=threshold",
        "causal": True, "right_censoring": True, "unit_checks": checks,
        "calibration_partition": "development repetitions 1-2 only",
        "validation_partition": "development repetition 3",
        "fixed_rule_false_alert_budget": True,
        "selected_parameters": configurations,
        "sealed_status": "post-hoc secondary; not a new untouched primary test",
    }, indent=2) + "\n")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
