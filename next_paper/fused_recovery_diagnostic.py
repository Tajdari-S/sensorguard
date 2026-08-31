#!/usr/bin/env python3
"""Post-sealed diagnostics for recovering fused-update detection.

This script cannot revise the original sealed result. It asks two engineering
questions: whether AdamW transfers to fused update without fused examples, and
whether fused update is learnable when repetitions 1--2 become development
data and repetition 3 remains outside fitting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from analyze_synchronized_physical import (  # noqa: E402
    META, STRIDE_S, WINDOW_S, build, nvml_seconds, physical_seconds, summaries,
)


THRESHOLD = 0.85


def repetition(run_id: str) -> int:
    match = re.search(r"_r([123])$", str(run_id))
    if not match:
        raise ValueError(f"run ID lacks repetition suffix: {run_id}")
    return int(match.group(1))


def trend_summaries(frame: pd.DataFrame, columns: list[str], prefix: str) -> dict[str, float]:
    """Ordinary distribution summaries plus causal within-window drift."""
    out = summaries(frame, columns, prefix)
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        name = f"{prefix}{column}"
        if len(values) < 2:
            out[f"{name}_time_slope"] = 0.0
            out[f"{name}_end_minus_start"] = 0.0
            out[f"{name}_diff_std"] = 0.0
            continue
        time_axis = np.arange(len(values), dtype=float)
        out[f"{name}_time_slope"] = float(np.polyfit(time_axis, values, 1)[0])
        edge = max(1, min(5, len(values) // 3))
        out[f"{name}_end_minus_start"] = float(
            values[-edge:].mean() - values[:edge].mean())
        out[f"{name}_diff_std"] = float(np.diff(values).std())
    return out


def trend_windows(seconds: pd.DataFrame, columns: list[str], prefix: str,
                  run_id: str, family: str, target: int) -> pd.DataFrame:
    rows = []
    for index, offset in enumerate(range(0, 106 - WINDOW_S, STRIDE_S)):
        chunk = seconds[(seconds.second >= offset) & (seconds.second < offset + WINDOW_S)]
        if len(chunk) < 20:
            continue
        row = trend_summaries(chunk, columns, prefix)
        row.update({"run_id": run_id, "family": family, "target": target,
                    "window_index": index, "window_end_s": offset + WINDOW_S})
        rows.append(row)
    return pd.DataFrame(rows)


def build_trend(plan: dict, node_root: Path, pico_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    electrical_frames, nvml_frames = [], []
    for run in plan["runs"]:
        run_id = run["run_id"]
        workload_path = node_root / run_id / "workload.json"
        if workload_path.is_file():
            workload = json.loads(workload_path.read_text())
        else:
            lines = (node_root / run_id / "workload.stdout").read_text().splitlines()
            records = [json.loads(line.removeprefix("useful_work "))
                       for line in lines if line.startswith("useful_work ")]
            if len(records) != 1:
                raise RuntimeError(f"expected one useful_work record for {run_id}")
            workload = records[0]
        start, end = float(workload["start_epoch_s"]), float(workload["end_epoch_s"])
        physical = physical_seconds(pico_root / run_id, start, end)
        nvml = nvml_seconds(node_root / run_id / "nvml.csv", start, end)
        electrical_columns = [column for column in physical.columns if column != "second"]
        nvml_columns = ["util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w", "temp_c",
                        "clock_sm_mhz", "clock_mem_mhz", "pcie_tx_kbps", "pcie_rx_kbps"]
        electrical_frames.append(trend_windows(
            physical, electrical_columns, "elec_", run_id, run["mode"], int(run["target"])))
        nvml_frames.append(trend_windows(
            nvml, nvml_columns, "nvml_", run_id, run["mode"], int(run["target"])))
    return pd.concat(electrical_frames, ignore_index=True), pd.concat(nvml_frames, ignore_index=True)


def models():
    common = dict(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                  class_weight="balanced", random_state=20260831, n_jobs=-1)
    return {
        "random_forest": RandomForestClassifier(**common),
        "extra_trees": ExtraTreesClassifier(**common),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000,
                               random_state=20260831),
        ),
    }


def first_alert_rule(probabilities: np.ndarray, threshold: float = THRESHOLD,
                     rule_windows: int = 5, rule_hits: int = 3) -> int | None:
    hits = np.asarray(probabilities) >= threshold
    if len(hits) < rule_windows:
        return None
    for index in range(rule_windows - 1, len(hits)):
        if hits[index - rule_windows + 1:index + 1].sum() >= rule_hits:
            return index
    return None


def fit_and_predict(model, fit: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    features = sorted(set(fit.columns) - META - {"repetition"})
    if features != sorted(set(test.columns) - META - {"repetition"}):
        raise AssertionError("fit/test feature schemas differ")
    if set(fit.target.unique()) != {0, 1}:
        raise RuntimeError(f"{protocol}/{modality} fit data lacks a class")
    model.fit(fit[features].fillna(0), fit.target)
    probabilities = model.predict_proba(test[features].fillna(0))[:, list(model.classes_).index(1)]
    scored = test[["run_id", "family", "target", "window_index", "window_end_s"]].copy()
    scored["probability"] = probabilities
    return scored


def summarize_runs(protocol: str, modality: str, model_name: str,
                   scored: pd.DataFrame, fit_runs: int,
                   threshold: float = THRESHOLD, rule_windows: int = 5,
                   rule_hits: int = 3, split: str = "test") -> list[dict]:
    rows = []
    for run_id, run in scored.groupby("run_id"):
        run = run.sort_values("window_index")
        index = first_alert_rule(run.probability.to_numpy(), threshold,
                                 rule_windows, rule_hits)
        rows.append({
            "protocol": protocol, "modality": modality, "model": model_name,
            "split": split,
            "run_id": run_id, "family": run.family.iloc[0], "target": int(run.target.iloc[0]),
            "alert": int(index is not None),
            "time_to_alert_s": None if index is None else float(run.iloc[index].window_end_s),
            "max_probability": float(run.probability.max()),
            "mean_probability": float(run.probability.mean()),
            "fit_runs": int(fit_runs),
            "threshold": float(threshold), "rule_windows": int(rule_windows),
            "rule_hits": int(rule_hits),
        })
    return rows


def score(protocol: str, modality: str, model_name: str, model,
          fit: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    scored = fit_and_predict(model, fit, test)
    return summarize_runs(protocol, modality, model_name, scored,
                          fit.run_id.nunique())


def choose_rule(calibration: pd.DataFrame) -> tuple[float, int, int, pd.DataFrame]:
    """Choose a zero-control-FP sequential rule on calibration runs only.

    Candidate rules remain deliberately small and deployable: threshold a
    window score and require H hits in the last W windows. Selection first
    maximizes detected calibration attacks, then prefers stricter thresholds,
    more required hits, and shorter windows.
    """
    candidates = []
    for threshold in np.round(np.arange(0.10, 0.851, 0.025), 3):
        for windows in range(1, 6):
            for hits in range(1, windows + 1):
                runs = pd.DataFrame(summarize_runs(
                    "calibration", "calibration", "calibration", calibration,
                    0, threshold, windows, hits, split="calibration"))
                positives = runs[runs.target == 1]
                controls = runs[runs.target == 0]
                candidates.append({
                    "threshold": float(threshold), "rule_windows": windows,
                    "rule_hits": hits,
                    "positive_detected": int(positives.alert.sum()),
                    "positive_runs": len(positives),
                    "false_alerts": int(controls.alert.sum()),
                    "control_runs": len(controls),
                })
    table = pd.DataFrame(candidates)
    eligible = table[table.false_alerts == 0].copy()
    if eligible.empty:
        selected = table.sort_values(
            ["false_alerts", "positive_detected", "threshold", "rule_hits", "rule_windows"],
            ascending=[True, False, False, False, True]).iloc[0]
    else:
        selected = eligible.sort_values(
            ["positive_detected", "threshold", "rule_hits", "rule_windows"],
            ascending=[False, False, False, True]).iloc[0]
    return (float(selected.threshold), int(selected.rule_windows),
            int(selected.rule_hits), table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-plan", type=Path, required=True)
    parser.add_argument("--development-node-root", type=Path, required=True)
    parser.add_argument("--development-pico-root", type=Path, required=True)
    parser.add_argument("--sealed-plan", type=Path, required=True)
    parser.add_argument("--sealed-node-root", type=Path, required=True)
    parser.add_argument("--sealed-pico-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dev_plan = json.loads(args.development_plan.read_text())
    sealed_plan = json.loads(args.sealed_plan.read_text())
    dev_e, dev_n = build(dev_plan, args.development_node_root, args.development_pico_root)
    sealed_e, sealed_n = build(sealed_plan, args.sealed_node_root, args.sealed_pico_root)
    dev_et, dev_nt = build_trend(
        dev_plan, args.development_node_root, args.development_pico_root)
    sealed_et, sealed_nt = build_trend(
        sealed_plan, args.sealed_node_root, args.sealed_pico_root)
    datasets = {
        "NVML": (dev_n, sealed_n),
        "Electrical": (dev_e, sealed_e),
        "NVML + electrical": (
            dev_n.merge(dev_e, on=list(META), validate="one_to_one"),
            sealed_n.merge(sealed_e, on=list(META), validate="one_to_one"),
        ),
        "Electrical + drift": (dev_et, sealed_et),
        "NVML + electrical + drift": (
            dev_nt.merge(dev_et, on=list(META), validate="one_to_one"),
            sealed_nt.merge(sealed_et, on=list(META), validate="one_to_one"),
        ),
    }

    output_rows = []
    calibration_rows = []
    for modality, (development, sealed) in datasets.items():
        development = development.copy()
        sealed = sealed.copy()
        sealed["repetition"] = sealed.run_id.map(repetition)
        development["repetition"] = development.run_id.map(repetition)

        # Protocol A: fused-update is entirely absent from fitting. AdamW and
        # matched controls from repetitions 1--2 augment ordinary development.
        transfer_extra = sealed[(sealed.repetition <= 2) & (sealed.family != "fused_update")]
        transfer_fit = pd.concat([development, transfer_extra], ignore_index=True)
        transfer_test = sealed[
            (sealed.family == "fused_update")
            | ((sealed.repetition == 3) & (sealed.family != "fused_update"))
        ]

        # Protocol B: repetitions 1--2 of every family, including fused update,
        # become development data; repetition 3 is a run-level holdout.
        adaptation_fit = pd.concat(
            [development, sealed[sealed.repetition <= 2]], ignore_index=True)
        adaptation_test = sealed[sealed.repetition == 3]

        for protocol, fit, test in (
            ("adamw_transfer_no_fused_fit", transfer_fit, transfer_test),
            ("fused_r1r2_to_r3_adaptation", adaptation_fit, adaptation_test),
        ):
            for model_name, model in models().items():
                output_rows.extend(score(protocol, modality, model_name, model, fit, test))

        # Protocol C is temporally nested: repetition 1 may alter the model,
        # repetition 2 selects only the alert threshold/vote rule, and
        # repetition 3 is not used until the final evaluation.
        nested_fit = pd.concat(
            [development, sealed[sealed.repetition == 1]], ignore_index=True)
        nested_calibration = sealed[sealed.repetition == 2]
        nested_test = sealed[sealed.repetition == 3]
        for model_name, model in models().items():
            calibration_scored = fit_and_predict(model, nested_fit, nested_calibration)
            threshold, windows, hits, candidate_table = choose_rule(calibration_scored)
            candidate_table.insert(0, "protocol", "nested_r1_fit_r2_rule_r3_test")
            candidate_table.insert(1, "modality", modality)
            candidate_table.insert(2, "model", model_name)
            candidate_table["selected"] = (
                (candidate_table.threshold == threshold)
                & (candidate_table.rule_windows == windows)
                & (candidate_table.rule_hits == hits)
            )
            calibration_rows.append(candidate_table)
            output_rows.extend(summarize_runs(
                "nested_r1_fit_r2_rule_r3_test", modality, model_name,
                calibration_scored, nested_fit.run_id.nunique(), threshold,
                windows, hits, split="calibration"))
            test_scored = fit_and_predict(model, nested_fit, nested_test)
            output_rows.extend(summarize_runs(
                "nested_r1_fit_r2_rule_r3_test", modality, model_name,
                test_scored, nested_fit.run_id.nunique(), threshold, windows,
                hits, split="test"))

        # Diagnostic-only rotation: repeat the same separation for every
        # assignment of one fused repetition to fitting, one to rule
        # calibration, and one to testing. This is not a new sealed result,
        # but exposes whether a proposed remedy depends on a lucky split.
        for fit_rep in (1, 2, 3):
            for calibration_rep in (1, 2, 3):
                if calibration_rep == fit_rep:
                    continue
                test_rep = ({1, 2, 3} - {fit_rep, calibration_rep}).pop()
                protocol = (
                    f"rotation_fit_r{fit_rep}_cal_r{calibration_rep}_test_r{test_rep}")
                rotation_fit = pd.concat(
                    [development, sealed[sealed.repetition == fit_rep]],
                    ignore_index=True)
                rotation_calibration = sealed[sealed.repetition == calibration_rep]
                rotation_test = sealed[sealed.repetition == test_rep]
                for model_name, model in models().items():
                    calibration_scored = fit_and_predict(
                        model, rotation_fit, rotation_calibration)
                    threshold, windows, hits, candidate_table = choose_rule(
                        calibration_scored)
                    candidate_table.insert(0, "protocol", protocol)
                    candidate_table.insert(1, "modality", modality)
                    candidate_table.insert(2, "model", model_name)
                    candidate_table["selected"] = (
                        (candidate_table.threshold == threshold)
                        & (candidate_table.rule_windows == windows)
                        & (candidate_table.rule_hits == hits)
                    )
                    calibration_rows.append(candidate_table)
                    output_rows.extend(summarize_runs(
                        protocol, modality, model_name, calibration_scored,
                        rotation_fit.run_id.nunique(), threshold, windows, hits,
                        split="calibration"))
                    test_scored = fit_and_predict(model, rotation_fit, rotation_test)
                    output_rows.extend(summarize_runs(
                        protocol, modality, model_name, test_scored,
                        rotation_fit.run_id.nunique(), threshold, windows, hits,
                        split="test"))

    predictions = pd.DataFrame(output_rows)
    summaries = []
    test_predictions = predictions[predictions.split == "test"]
    for keys, group in test_predictions.groupby(["protocol", "modality", "model"]):
        protocol, modality, model_name = keys
        fused = group[group.family == "fused_update"]
        controls = group[group.target == 0]
        positive_controls = group[(group.target == 1) & (group.family != "fused_update")]
        summaries.append({
            "protocol": protocol, "modality": modality, "model": model_name,
            "fused_runs": len(fused), "fused_detected": int(fused.alert.sum()),
            "fused_detection_rate": float(fused.alert.mean()),
            "matched_control_runs": len(controls), "false_alerts": int(controls.alert.sum()),
            "positive_control_runs": len(positive_controls),
            "positive_controls_detected": int(positive_controls.alert.sum()),
            "median_fused_time_to_alert_s": fused.loc[fused.alert == 1, "time_to_alert_s"].median()
            if fused.alert.any() else None,
            "threshold": float(group.threshold.iloc[0]),
            "rule_windows": int(group.rule_windows.iloc[0]),
            "rule_hits": int(group.rule_hits.iloc[0]),
        })
    summary = pd.DataFrame(summaries)
    summary["candidate_solution"] = (
        (summary.fused_detected == summary.fused_runs)
        & (summary.false_alerts == 0)
        & (summary.positive_controls_detected == summary.positive_control_runs)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "fused_recovery_run_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "fused_recovery_summary.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        args.output_dir / "fused_recovery_rule_search.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
