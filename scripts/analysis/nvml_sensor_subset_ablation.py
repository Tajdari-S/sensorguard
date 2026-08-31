#!/usr/bin/env python3
"""Evaluate NVML plus every available physical-sensor subset on 36 paired runs."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_physical_nvml_confusion as matched  # noqa: E402
import physical_sensor_ablation as physical  # noqa: E402


SENSORS = {
    "GPU current": physical.MODALITIES["GPU current clamp"],
    "Motherboard current": physical.MODALITIES["Motherboard clamp"],
    "Ultrasound": physical.MODALITIES["UltraMic"],
}
METADATA = {"run_id", "window_order", "workload_label", "target"}


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["run_id"] = out["run_id"].astype(str)
    out["window_order"] = out.groupby("run_id").cumcount()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--gate-output", type=Path,
                        default=Path("next_paper/results/optional_sensor_gate.csv"))
    args = parser.parse_args()

    sensor_data = physical.load_base_campaign(args.source_repo / "sensor_logs")
    physical_windows = {}
    for name, specification in SENSORS.items():
        physical_windows[name] = ordered(physical.make_windows(
            sensor_data, specification["channels"], specification["temporal"], 30, 15))
    reference = physical_windows["GPU current"]
    eligible_run_ids = set(reference.run_id.unique())
    if len(eligible_run_ids) != 36:
        raise RuntimeError(f"expected 36 paired runs, found {len(eligible_run_ids)}")

    source_nvml = matched.load_paired_nvml(args.source_repo, eligible_run_ids)
    nvml_module = matched.load_source_nvml_module(args.source_repo)
    source_nvml = nvml_module._normalize_columns(source_nvml)
    source_nvml["threeway_label"] = source_nvml["workload_label"].where(
        ~source_nvml["workload_label"].isin(physical.TRAINING_LABELS), "ml_training")
    nvml = nvml_module.sliding_windows(source_nvml, window_sec=30, stride_sec=15)
    nvml_features = nvml_module.get_feature_cols(nvml)
    nvml = ordered(nvml)
    nvml["target"] = (nvml["threeway_label"] == "ml_training").astype(int)
    labels = reference[["run_id", "window_order", "workload_label", "target"]]
    combined_base = nvml[["run_id", "window_order", *nvml_features]].merge(
        labels, on=["run_id", "window_order"], validate="one_to_one")

    folds = matched.make_family_heldout_folds(reference)
    configurations = [("NVML", tuple())]
    sensor_names = list(SENSORS)
    for count in range(1, len(sensor_names) + 1):
        for subset in itertools.combinations(sensor_names, count):
            configurations.append(("NVML + " + " + ".join(subset), subset))

    all_runs, all_windows, summary_rows, family_rows, run_by_name = [], [], [], [], {}
    for name, subset in configurations:
        combined = combined_base.copy()
        features = list(nvml_features)
        for sensor_name in subset:
            frame = physical_windows[sensor_name]
            sensor_features = [column for column in frame.columns if column not in METADATA]
            combined = combined.merge(
                frame[["run_id", "window_order", *sensor_features]],
                on=["run_id", "window_order"], validate="one_to_one")
            features.extend(sensor_features)
        predictions = matched.out_of_fold_predictions(combined, features, folds, name)
        runs = matched.aggregate_runs(predictions)
        run_by_name[name] = runs
        all_runs.append(runs)
        all_windows.append(predictions)
        counts = runs.outcome.value_counts()
        tp, fn = int(counts.get("TP", 0)), int(counts.get("FN", 0))
        fp, tn = int(counts.get("FP", 0)), int(counts.get("TN", 0))
        positive = runs[runs.target == 1]
        rates = positive.groupby("workload_label")["prediction"].mean()
        summary_rows.append({
            "sensor_subset": name, "added_physical_sensors": len(subset),
            "feature_count": len(features), "runs": len(runs),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "detection_rate": tp / (tp + fn),
            "worst_family_detection_rate": float(rates.min()),
            "false_alert_rate_per_run": fp / (fp + tn),
        })
        for family, rate in rates.items():
            family_rows.append({"sensor_subset": name, "family": family,
                                "runs": int((positive.workload_label == family).sum()),
                                "detection_rate": float(rate)})

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary.sensor_subset == "NVML"].iloc[0]
    summary["detection_gain_over_nvml_pp"] = 100 * (summary.detection_rate - baseline.detection_rate)
    summary["worst_family_gain_over_nvml_pp"] = 100 * (
        summary.worst_family_detection_rate - baseline.worst_family_detection_rate)
    baseline_predictions = run_by_name["NVML"]
    baseline_predictions = baseline_predictions[baseline_predictions.target == 1].set_index("run_id")
    ci_low, ci_high = [], []
    for name in summary.sensor_subset:
        candidate = run_by_name[name]
        candidate = candidate[candidate.target == 1].set_index("run_id")
        differences = (
            candidate.loc[baseline_predictions.index, "prediction"].to_numpy(dtype=float)
            - baseline_predictions["prediction"].to_numpy(dtype=float)
        )
        if name == "NVML":
            low, high = 0.0, 0.0
        else:
            generator = np.random.default_rng(20260831)
            indices = generator.integers(0, len(differences), size=(10000, len(differences)))
            gains = differences[indices].mean(axis=1) * 100
            low, high = np.quantile(gains, [0.025, 0.975])
        ci_low.append(float(low))
        ci_high.append(float(high))
    summary["detection_gain_ci95_low_pp"] = ci_low
    summary["detection_gain_ci95_high_pp"] = ci_high
    summary["retained_over_nvml"] = (
        (summary.detection_gain_ci95_low_pp > 0)
        & (summary.fp <= baseline.fp)
        & (summary.worst_family_detection_rate >= baseline.worst_family_detection_rate)
    )
    gpu_row = summary[summary.sensor_subset == "NVML + GPU current"].iloc[0]
    summary["incremental_gain_over_nvml_plus_gpu_pp"] = summary.apply(
        lambda row: 100 * (row.detection_rate - gpu_row.detection_rate)
        if "GPU current" in row.sensor_subset else None,
        axis=1,
    )
    summary["minimal_selected_subset"] = summary.sensor_subset == "NVML + GPU current"
    summary["rank"] = summary.sort_values(
        ["worst_family_detection_rate", "detection_rate", "fp", "feature_count"],
        ascending=[False, False, True, True]).reset_index().index + 1
    summary["rank"] = summary.sensor_subset.map(
        summary.sort_values(["worst_family_detection_rate", "detection_rate", "fp", "feature_count"],
                            ascending=[False, False, True, True])
        .assign(order=lambda frame: range(1, len(frame) + 1)).set_index("sensor_subset")["order"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values("rank").to_csv(args.output_dir / "nvml-sensor-subset-ablation.csv", index=False)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values("rank").to_csv(args.gate_output, index=False)
    pd.DataFrame(family_rows).to_csv(args.output_dir / "nvml-sensor-subset-family-results.csv", index=False)
    pd.concat(all_runs, ignore_index=True).to_csv(
        args.evaluation_dir / "nvml-sensor-subset-run-predictions.csv", index=False)
    pd.concat(all_windows, ignore_index=True).to_csv(
        args.evaluation_dir / "nvml-sensor-subset-window-predictions.csv", index=False)
    print(summary.sort_values("rank").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
