#!/usr/bin/env python3
"""Compare NVML and the GPU-current SensorGuard pilot on the same 36 runs.

The paired traces come from Robi Rahman's ``physical-sensor-detection`` branch.
Both modalities use 30-second windows, a 15-second stride, the same five
run-grouped folds, the same random-forest hyperparameters, and the current
paper's amended candidate run rule: at least three probabilities at or above 0.85 in any
five consecutive windows.

This is an in-corpus, run-grouped proof of concept.  It is not the sealed
held-out-family evaluation requested by the final SensorGuard protocol.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import physical_sensor_ablation as physical  # noqa: E402


RUN_THRESHOLD = 0.85
EXTENDED_TRAINING_PREFIXES = physical.FOLLOWUP_PREFIXES


def load_extended_campaign(sensor_dir: Path) -> pd.DataFrame:
    """Load all paired development runs, including adaptive physical attacks."""
    frames = []
    for filename in sorted(glob.glob(str(sensor_dir / "*_sensors.parquet"))):
        frame = pd.read_parquet(filename)
        label = str(frame["workload_label"].iloc[0])
        is_training = label in physical.TRAINING_LABELS or label.startswith(EXTENDED_TRAINING_PREFIXES)
        frame["target"] = int(is_training)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No sensor logs found in {sensor_dir}")
    return pd.concat(frames, ignore_index=True)


def load_source_nvml_module(source_repo: Path):
    module_path = source_repo / "classifier" / "threeway_improved.py"
    spec = importlib.util.spec_from_file_location("robi_threeway_improved", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import NVML feature extractor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_paired_nvml(source_repo: Path, eligible_run_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for filename in sorted(glob.glob(str(source_repo / "data" / "*RTX_3090*.parquet"))):
        try:
            frame = pd.read_parquet(filename)
        except Exception:
            # Unfetched Git LFS pointer files are intentionally ignored.
            continue
        if frame.empty or "run_id" not in frame.columns:
            continue
        run_id = str(frame["run_id"].iloc[0])
        if run_id not in eligible_run_ids or run_id in seen:
            continue
        seen.add(run_id)
        frames.append(frame)
    missing = sorted(eligible_run_ids - seen)
    if missing:
        raise RuntimeError(f"Missing paired NVML parquet for {len(missing)} runs: {missing}")
    data = pd.concat(frames, ignore_index=True)
    data["run_id"] = data["run_id"].astype(str)
    data["target"] = data["workload_label"].isin(physical.TRAINING_LABELS).astype(int)
    return data


def make_identical_folds(current_windows: pd.DataFrame) -> list[tuple[set[str], set[str]]]:
    y = current_windows["target"].to_numpy(dtype=int)
    groups = current_windows["run_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    placeholder = np.zeros((len(current_windows), 1), dtype=np.float32)
    folds = []
    for train, test in splitter.split(placeholder, y, groups):
        train_runs = set(groups[train])
        test_runs = set(groups[test])
        if train_runs & test_runs:
            raise AssertionError("A run appears in both train and test")
        folds.append((train_runs, test_runs))
    test_counts = pd.Series([rid for _, test in folds for rid in test]).value_counts()
    if not (test_counts == 1).all():
        raise AssertionError("Each eligible run must appear in exactly one test fold")
    return folds


def make_family_heldout_folds(current_windows: pd.DataFrame) -> list[tuple[set[str], set[str]]]:
    """Leave each workload label out once, using the same split for both modalities."""
    run_table = current_windows[["run_id", "workload_label", "target"]].drop_duplicates()
    all_runs = set(run_table["run_id"].astype(str))
    folds = []
    for workload_label in sorted(run_table["workload_label"].unique()):
        test_runs = set(
            run_table.loc[run_table["workload_label"] == workload_label, "run_id"].astype(str)
        )
        train_runs = all_runs - test_runs
        train_targets = set(run_table.loc[run_table["run_id"].isin(train_runs), "target"])
        if train_targets != {0, 1}:
            raise RuntimeError(f"Held-out family {workload_label} leaves only one training class")
        folds.append((train_runs, test_runs))
    test_counts = pd.Series([rid for _, test in folds for rid in test]).value_counts()
    if not (test_counts == 1).all():
        raise AssertionError("Each eligible run must appear in exactly one family-held-out test fold")
    return folds


def out_of_fold_predictions(
    windows: pd.DataFrame,
    feature_columns: list[str],
    folds: list[tuple[set[str], set[str]]],
    method: str,
) -> pd.DataFrame:
    outputs = []
    run_ids = windows["run_id"].astype(str)
    for fold_index, (train_runs, test_runs) in enumerate(folds, start=1):
        train_mask = run_ids.isin(train_runs)
        test_mask = run_ids.isin(test_runs)
        x_train = windows.loc[train_mask, feature_columns].fillna(0).to_numpy(dtype=np.float32)
        y_train = windows.loc[train_mask, "target"].to_numpy(dtype=int)
        x_test = windows.loc[test_mask, feature_columns].fillna(0).to_numpy(dtype=np.float32)
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, int(np.where(model.classes_ == 1)[0][0])]
        output = windows.loc[test_mask, ["run_id", "workload_label", "target"]].copy()
        output["fold"] = fold_index
        output["method"] = method
        output["training_probability"] = probability
        output["window_prediction"] = (probability >= 0.5).astype(int)
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def aggregate_runs(window_predictions: pd.DataFrame, threshold: float = RUN_THRESHOLD) -> pd.DataFrame:
    def fixed_three_of_five(probabilities: pd.Series) -> int:
        hits = (probabilities.to_numpy(dtype=float) >= threshold).astype(int)
        if len(hits) < 5:
            return int(hits.sum() >= 3)
        return int((np.convolve(hits, np.ones(5, dtype=int), mode="valid") >= 3).any())

    runs = (
        window_predictions.groupby(["method", "run_id", "workload_label", "target", "fold"], as_index=False)
        .agg(
            training_probability=("training_probability", "mean"),
            n_windows=("window_prediction", "size"),
            correct_windows=("window_prediction", lambda values: int((values.to_numpy() == window_predictions.loc[values.index, "target"].to_numpy()).sum())),
            prediction=("training_probability", fixed_three_of_five),
        )
    )
    runs["outcome"] = np.select(
        [
            (runs["target"] == 1) & (runs["prediction"] == 1),
            (runs["target"] == 1) & (runs["prediction"] == 0),
            (runs["target"] == 0) & (runs["prediction"] == 1),
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    return runs


def summarize(run_predictions: pd.DataFrame, window_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, runs in run_predictions.groupby("method", sort=False):
        windows = window_predictions[window_predictions["method"] == method]
        counts = runs["outcome"].value_counts()
        tp = int(counts.get("TP", 0))
        fn = int(counts.get("FN", 0))
        fp = int(counts.get("FP", 0))
        tn = int(counts.get("TN", 0))
        rows.append({
            "method": method,
            "total_runs": len(runs),
            "training_runs": tp + fn,
            "nontraining_runs": tn + fp,
            "true_positive_detected_training": tp,
            "false_negative_missed_training": fn,
            "false_positive_false_alert": fp,
            "true_negative_correct_rejection": tn,
            "correct_runs": tp + tn,
            "run_accuracy": (tp + tn) / len(runs),
            "training_detection_rate": tp / (tp + fn),
            "false_alert_rate_per_run": fp / (fp + tn),
            "window_count": len(windows),
            "correct_windows": int((windows["window_prediction"] == windows["target"]).sum()),
            "window_accuracy": float((windows["window_prediction"] == windows["target"]).mean()),
            "window_sec": 30,
            "stride_sec": 15,
            "run_rule": f"fixed 3-of-5 consecutive windows at probability >= {RUN_THRESHOLD:.2f}",
            "split": "same 5-fold StratifiedGroupKFold by run_id",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    sensor_data = physical.load_base_campaign(args.source_repo / "sensor_logs")
    current_windows = physical.make_windows(
        sensor_data,
        physical.MODALITIES["GPU current clamp"]["channels"],
        physical.MODALITIES["GPU current clamp"]["temporal"],
        30,
        15,
    )
    current_windows["run_id"] = current_windows["run_id"].astype(str)
    eligible_run_ids = set(current_windows["run_id"].unique())
    if len(eligible_run_ids) != 36:
        raise RuntimeError(f"Expected 36 eligible runs, found {len(eligible_run_ids)}")

    source_nvml = load_paired_nvml(args.source_repo, eligible_run_ids)
    nvml_module = load_source_nvml_module(args.source_repo)
    source_nvml = nvml_module._normalize_columns(source_nvml)
    source_nvml["threeway_label"] = np.where(source_nvml["target"] == 1, "ml_training", "other")
    nvml_windows = nvml_module.sliding_windows(source_nvml, window_sec=30, stride_sec=15)
    nvml_windows["target"] = (nvml_windows["threeway_label"] == "ml_training").astype(int)
    nvml_windows["run_id"] = nvml_windows["run_id"].astype(str)
    if set(nvml_windows["run_id"].unique()) != eligible_run_ids:
        raise RuntimeError("NVML and current-sensor eligible run sets differ")

    folds = make_identical_folds(current_windows)
    physical_features = [
        column for column in current_windows.columns
        if column not in {"run_id", "workload_label", "target"}
    ]
    nvml_features = nvml_module.get_feature_cols(nvml_windows)

    window_predictions = pd.concat([
        out_of_fold_predictions(nvml_windows, nvml_features, folds, "NVML"),
        out_of_fold_predictions(current_windows, physical_features, folds, "SensorGuard (GPU current)"),
    ], ignore_index=True)
    run_predictions = aggregate_runs(window_predictions)
    summary = summarize(run_predictions, window_predictions)

    tables = args.results_dir / "tables"
    evaluation = args.results_dir / "evaluation"
    tables.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "matched-36-run-confusion.csv", index=False)
    run_predictions.to_csv(evaluation / "matched-36-run-predictions.csv", index=False)
    window_predictions.to_csv(evaluation / "matched-36-window-predictions.csv", index=False)
    print(summary.to_string(index=False))

    family_folds = make_family_heldout_folds(current_windows)
    family_window_predictions = pd.concat([
        out_of_fold_predictions(nvml_windows, nvml_features, family_folds, "NVML"),
        out_of_fold_predictions(current_windows, physical_features, family_folds, "SensorGuard (GPU current)"),
    ], ignore_index=True)
    family_run_predictions = aggregate_runs(family_window_predictions)
    family_summary = summarize(family_run_predictions, family_window_predictions)
    family_summary["split"] = "leave-one-workload-family-out by workload_label"
    family_summary.to_csv(tables / "matched-36-family-heldout-confusion.csv", index=False)
    family_run_predictions.to_csv(
        evaluation / "matched-36-family-heldout-run-predictions.csv", index=False
    )
    family_window_predictions.to_csv(
        evaluation / "matched-36-family-heldout-window-predictions.csv", index=False
    )
    print("\nLeave-one-workload-family-out:")
    print(family_summary.to_string(index=False))

    # Broader development-only diagnostic. These additional attacks were
    # created adaptively against earlier physical models, so they increase
    # stress-test breadth but do not constitute a sealed independent test.
    extended_data = load_extended_campaign(args.source_repo / "sensor_logs")
    extended_windows = physical.make_windows(
        extended_data,
        physical.MODALITIES["GPU current clamp"]["channels"],
        physical.MODALITIES["GPU current clamp"]["temporal"],
        30,
        15,
    )
    extended_windows["run_id"] = extended_windows["run_id"].astype(str)
    extended_run_ids = set(extended_windows["run_id"].unique())
    extended_nvml = load_paired_nvml(args.source_repo, extended_run_ids)
    extended_nvml = nvml_module._normalize_columns(extended_nvml)
    extended_targets = (
        extended_nvml["workload_label"].isin(physical.TRAINING_LABELS)
        | extended_nvml["workload_label"].str.startswith(EXTENDED_TRAINING_PREFIXES)
    )
    extended_nvml["target"] = extended_targets.astype(int)
    extended_nvml["threeway_label"] = np.where(
        extended_nvml["target"] == 1, "ml_training", "other"
    )
    extended_nvml_windows = nvml_module.sliding_windows(
        extended_nvml, window_sec=30, stride_sec=15
    )
    extended_nvml_windows["target"] = (
        extended_nvml_windows["threeway_label"] == "ml_training"
    ).astype(int)
    extended_nvml_windows["run_id"] = extended_nvml_windows["run_id"].astype(str)
    if set(extended_nvml_windows["run_id"].unique()) != extended_run_ids:
        raise RuntimeError("Extended NVML and current-sensor run sets differ")

    extended_folds = make_family_heldout_folds(extended_windows)
    extended_physical_features = [
        column for column in extended_windows.columns
        if column not in {"run_id", "workload_label", "target"}
    ]
    extended_nvml_features = nvml_module.get_feature_cols(extended_nvml_windows)
    extended_window_predictions = pd.concat([
        out_of_fold_predictions(
            extended_nvml_windows, extended_nvml_features, extended_folds, "NVML"
        ),
        out_of_fold_predictions(
            extended_windows, extended_physical_features, extended_folds,
            "SensorGuard (GPU current)",
        ),
    ], ignore_index=True)
    extended_run_predictions = aggregate_runs(extended_window_predictions)
    extended_summary = summarize(extended_run_predictions, extended_window_predictions)
    extended_summary["split"] = "leave-one-workload-label-out; adaptive development stress test"
    extended_summary["campaign_note"] = (
        "includes adaptive physical red-team attacks; not a sealed final test"
    )
    extended_summary.to_csv(
        tables / "matched-extended-family-heldout-confusion.csv", index=False
    )
    extended_run_predictions.to_csv(
        evaluation / "matched-extended-family-heldout-run-predictions.csv", index=False
    )
    extended_window_predictions.to_csv(
        evaluation / "matched-extended-family-heldout-window-predictions.csv", index=False
    )
    print("\nExtended adaptive development stress test:")
    print(extended_summary.to_string(index=False))


if __name__ == "__main__":
    main()
