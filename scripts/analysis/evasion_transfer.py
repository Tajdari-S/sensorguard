#!/usr/bin/env python3
"""Measure whether a lightweight SensorGuard RF transfers across evasions.

Two diagnostic protocols are emitted from the same frozen label table:

* pairwise: fit ordinary data plus evasion A, evaluate unseen evasion B;
* leave_one_out: fit ordinary data plus every evasion except B, evaluate B.

Every evaluation also uses a preassigned, run-disjoint non-training control
set to estimate false alerts. The preregistered sealed family is always
excluded: this development script cannot open the final test.

Input columns:
    run_id,trace_path,gpu_index,label,family,evasion_family,split,
    gpu_uuid,collection_day

``split`` is ``fit`` or ``control_test``. Evasion runs are dynamically held
out by family; non-training ``control_test`` rows are never used for fitting.
Use an empty evasion_family for ordinary training and all non-training runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from evaluation import first_alert_index, run_alert, wilson_interval  # noqa: E402
from features import extract_run, feature_names, stage2_names  # noqa: E402
from train_baseline import K_OF_N, RF_KW, THRESHOLD  # noqa: E402

REQUIRED_COLUMNS = {
    "run_id", "trace_path", "gpu_index", "label", "family",
    "evasion_family", "split", "gpu_uuid", "collection_day",
}
ALLOWED_LABELS = {"training", "inference", "non_ml"}
ALLOWED_SPLITS = {"fit", "control_test"}
ORDINARY_SOURCE = "ordinary_only"


def normalize_labels(runs: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the run-level evasion-transfer contract."""
    missing = sorted(REQUIRED_COLUMNS - set(runs.columns))
    if missing:
        raise ValueError(f"labels CSV is missing required columns: {missing}")
    out = runs.copy()
    out["evasion_family"] = out["evasion_family"].fillna("").astype(str).str.strip()
    out["split"] = out["split"].fillna("").astype(str).str.strip()
    if out["run_id"].isna().any() or out["run_id"].duplicated().any():
        raise ValueError("run_id must be present and unique")
    unexpected_labels = sorted(set(out["label"].dropna()) - ALLOWED_LABELS)
    if unexpected_labels:
        raise ValueError(f"unexpected labels: {unexpected_labels}")
    unexpected_splits = sorted(set(out["split"]) - ALLOWED_SPLITS)
    if unexpected_splits:
        raise ValueError(f"unexpected split values: {unexpected_splits}")
    if ((out["label"] == "training") & (out["split"] == "control_test")).any():
        raise ValueError("training runs cannot use split=control_test")
    if ((out["label"] != "training") & (out["evasion_family"] != "")).any():
        raise ValueError("only training runs may have an evasion_family")
    if not ((out["label"] != "training") & (out["split"] == "fit")).any():
        raise ValueError("at least one fit non-training control is required")
    if not ((out["label"] != "training") & (out["split"] == "control_test")).any():
        raise ValueError("at least one control_test non-training run is required")
    if not ((out["label"] == "training") & (out["evasion_family"] == "")).any():
        raise ValueError("at least one ordinary training run is required")
    if not ((out["label"] == "training") & (out["evasion_family"] != "")).any():
        raise ValueError("at least one evasion-family training run is required")
    return out


def evasion_families(runs: pd.DataFrame, sealed_family: str) -> list[str]:
    families = sorted(set(runs.loc[runs["label"] == "training", "evasion_family"]) - {""})
    return [family for family in families if family != sealed_family]


def plan_comparisons(runs: pd.DataFrame, sealed_family: str) -> list[dict]:
    """Return auditable run-ID splits for pairwise and leave-one-out tests."""
    families = evasion_families(runs, sealed_family)
    fit_controls = (runs["label"] != "training") & (runs["split"] == "fit")
    ordinary_training = (runs["label"] == "training") & (runs["evasion_family"] == "")
    test_controls = (runs["label"] != "training") & (runs["split"] == "control_test")
    sealed = runs["evasion_family"] == sealed_family
    plans = []
    for target in families:
        target_rows = (runs["label"] == "training") & (runs["evasion_family"] == target)
        test_mask = target_rows | test_controls

        # Ordinary-only is the no-evasion baseline for every target.
        fit_mask = (fit_controls | ordinary_training) & ~sealed
        plans.append(_plan_row(runs, "pairwise", ORDINARY_SOURCE, target, fit_mask, test_mask))

        for source in families:
            if source == target:
                continue
            source_rows = (runs["label"] == "training") & (runs["evasion_family"] == source)
            fit_mask = (fit_controls | ordinary_training | source_rows) & ~sealed
            plans.append(_plan_row(runs, "pairwise", source, target, fit_mask, test_mask))

        other_evasions = (
            (runs["label"] == "training")
            & (runs["evasion_family"] != "")
            & (runs["evasion_family"] != target)
            & ~sealed
        )
        fit_mask = fit_controls | ordinary_training | other_evasions
        plans.append(_plan_row(runs, "leave_one_out", "all_other_evasions", target,
                               fit_mask, test_mask))
    return plans


def plan_declared_holdout(runs: pd.DataFrame, seen_families: list[str],
                          target_family: str, sealed_family: str) -> list[dict]:
    """Plan two-seen/one-model-unseen comparisons from a frozen YAML split."""
    if target_family == sealed_family:
        raise ValueError("the preregistered final family cannot be opened by this diagnostic")
    if target_family in seen_families:
        raise ValueError("model-unseen family must not also appear in seen_families")
    available = set(runs.loc[runs["label"] == "training", "evasion_family"])
    missing = sorted((set(seen_families) | {target_family}) - available)
    if missing:
        raise ValueError(f"declared evasion families are absent from labels: {missing}")
    fit_controls = (runs["label"] != "training") & (runs["split"] == "fit")
    ordinary_training = (runs["label"] == "training") & (runs["evasion_family"] == "")
    test_controls = (runs["label"] != "training") & (runs["split"] == "control_test")
    target_rows = (runs["label"] == "training") & (runs["evasion_family"] == target_family)
    test_mask = target_rows | test_controls
    plans = []
    for source in seen_families:
        source_rows = (runs["label"] == "training") & (runs["evasion_family"] == source)
        plans.append(_plan_row(
            runs, "declared_pairwise", source, target_family,
            fit_controls | ordinary_training | source_rows, test_mask,
        ))
    all_seen = (runs["label"] == "training") & runs["evasion_family"].isin(seen_families)
    plans.append(_plan_row(
        runs, "declared_two_seen", "+".join(seen_families), target_family,
        fit_controls | ordinary_training | all_seen, test_mask,
    ))
    return plans


def _plan_row(runs, protocol, source, target, fit_mask, test_mask) -> dict:
    fit_ids = runs.loc[fit_mask, "run_id"].astype(str).tolist()
    test_ids = runs.loc[test_mask, "run_id"].astype(str).tolist()
    overlap = sorted(set(fit_ids) & set(test_ids))
    if overlap:
        raise AssertionError(f"run leakage in {protocol} {source}->{target}: {overlap[:5]}")
    if not fit_ids or not test_ids:
        raise ValueError(f"empty fit or test split in {protocol} {source}->{target}")
    return {
        "protocol": protocol,
        "source_family": source,
        "target_family": target,
        "fit_run_ids": fit_ids,
        "test_run_ids": test_ids,
    }


def extract_all_windows(runs: pd.DataFrame, window_s: int, stride_s: int) -> pd.DataFrame:
    frames = []
    for _, run in runs.iterrows():
        features = extract_run(run["trace_path"], int(run["gpu_index"]), window_s, stride_s)
        start = run.get("evaluation_start_raw_s")
        end = run.get("evaluation_end_raw_s")
        if start is not None and end is not None and math.isfinite(float(start)):
            # Keep only causal windows wholly inside the workload.  Raw files
            # also contain synchronization load markers, which must never be
            # allowed to trigger an attack detection.
            features = features[
                (features["window_end_raw_s"] >= float(start) + window_s - 1)
                & (features["window_end_raw_s"] <= float(end))
            ].copy()
        if features.empty:
            print(f"WARNING: no windows for {run['run_id']}", file=sys.stderr)
            continue
        for column in ["run_id", "label", "family", "evasion_family", "split",
                       "gpu_uuid", "collection_day"]:
            features[column] = run[column]
        frames.append(features)
    if not frames:
        raise ValueError("no usable feature windows were extracted")
    return pd.concat(frames, ignore_index=True)


def evaluate_plan(plan: dict, runs: pd.DataFrame, windows: pd.DataFrame,
                  names: list[str], s2_names: list[str]) -> tuple[dict, list[dict]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    fit = windows[windows["run_id"].isin(plan["fit_run_ids"])].copy()
    test = windows[windows["run_id"].isin(plan["test_run_ids"])].copy()
    if fit["run_id"].isin(test["run_id"]).any():
        raise AssertionError("window-level run leakage detected")

    x_fit = fit[names].to_numpy()
    x_test = test[names].to_numpy()
    fit_ml = fit["label"].isin(["training", "inference"]).to_numpy()
    fit_training = (fit["label"] == "training").to_numpy()
    test_training = (test["label"] == "training").to_numpy()
    ml_fit_indices = np.flatnonzero(fit_ml)
    if len(np.unique(fit_ml)) < 2:
        raise ValueError("fit split lacks ML or non-ML examples for stage 1")
    if len(np.unique(fit_training[ml_fit_indices])) < 2:
        raise ValueError("fit split lacks training or inference examples for stage 2")

    stage2_indices = [names.index(name) for name in s2_names]
    rf1 = RandomForestClassifier(**RF_KW).fit(x_fit, fit_ml)
    rf2 = RandomForestClassifier(**RF_KW).fit(
        x_fit[ml_fit_indices][:, stage2_indices], fit_training[ml_fit_indices]
    )
    probabilities = (
        rf1.predict_proba(x_test)[:, 1]
        * rf2.predict_proba(x_test[:, stage2_indices])[:, 1]
    )

    prediction_rows = []
    for run_id in sorted(test["run_id"].unique()):
        mask = test["run_id"].to_numpy() == run_id
        order = np.argsort(test.loc[mask, "window_end_raw_s"].to_numpy())
        run_probabilities = probabilities[mask][order]
        run_times = test.loc[mask, "window_end_raw_s"].to_numpy()[order]
        trigger_index = first_alert_index(run_probabilities, THRESHOLD, *K_OF_N)
        metadata = runs.loc[runs["run_id"] == run_id].iloc[0]
        trigger_time = None if trigger_index is None else float(run_times[trigger_index])
        evaluation_start = metadata.get("evaluation_start_raw_s")
        time_to_alert = None
        if trigger_time is not None and evaluation_start is not None:
            time_to_alert = trigger_time - float(evaluation_start)
        prediction_rows.append({
            "protocol": plan["protocol"],
            "source_family": plan["source_family"],
            "target_family": plan["target_family"],
            "run_id": run_id,
            "truth_training": bool(metadata["label"] == "training"),
            "alert": run_alert(run_probabilities, THRESHOLD, *K_OF_N),
            "evasion_family": metadata["evasion_family"],
            "duration_hours": float(metadata["evaluation_duration_hours"]),
            "max_p_training": float(np.max(run_probabilities)),
            "mean_p_training": float(np.mean(run_probabilities)),
            "first_trigger_window_end_raw_s": trigger_time,
            "time_to_alert_s": time_to_alert,
        })

    frame = pd.DataFrame(prediction_rows)
    positives = frame["truth_training"]
    negatives = frame[~frame["truth_training"]]
    tp = int(positives["alert"].sum())
    fn = int((~positives["alert"]).sum())
    fp = int(negatives["alert"].sum())
    tn = int((~negatives["alert"]).sum())
    negative_hours = float(negatives["duration_hours"].sum())
    auc = None
    if len(np.unique(test_training)) == 2:
        auc = float(roc_auc_score(test_training, probabilities))
    summary = {
        "protocol": plan["protocol"],
        "source_family": plan["source_family"],
        "target_family": plan["target_family"],
        "fit_runs": len(plan["fit_run_ids"]),
        "target_runs": int(len(positives)),
        "control_runs": int(len(negatives)),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "target_tpr": tp / (tp + fn) if tp + fn else None,
        "target_tpr_ci_low": wilson_interval(tp, tp + fn)[0],
        "target_tpr_ci_high": wilson_interval(tp, tp + fn)[1],
        "window_auc": auc,
        "negative_gpu_hours": negative_hours,
        "false_alerts_per_gpu_hour": fp / negative_hours if negative_hours else None,
    }
    return summary, prediction_rows


def evaluation_interval(trace_path: str | Path, gpu_index: int) -> tuple[float, float, str]:
    """Return the workload-only interval, falling back to the usable trace."""
    trace = Path(trace_path)
    manifest_path = trace.with_name("manifest.yaml")
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text())
        workload = manifest.get("workload") or {}
        start = workload.get("start_raw_s")
        end = workload.get("end_raw_s")
        if start is not None and end is not None:
            return float(start), float(end), "exact_manifest"
        pre = (manifest.get("marker_bursts") or {}).get("pre") or []
        duration = workload.get("duration_s")
        if pre and duration is not None:
            start = max(float(burst[1]) for burst in pre) + 2.0
            return start, start + float(duration), "inferred_manifest"

    trace_frame = pd.read_csv(trace)
    usable = trace_frame[
        (trace_frame["status"] == "ok")
        & (trace_frame["gpu_index"] == int(gpu_index))
    ]
    if usable.empty:
        raise ValueError(f"no usable rows for GPU {gpu_index} in {trace}")
    return (
        float(usable["t_target_raw_s"].min()),
        float(usable["t_target_raw_s"].max()),
        "full_usable_trace_no_manifest",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--window-s", type=int, default=30)
    parser.add_argument("--stride-s", type=int, default=15)
    parser.add_argument("--preregistration", type=Path,
                        default=Path("configs/preregistration.yaml"))
    parser.add_argument("--family-plan", type=Path,
                        help="optional YAML declaring seen_families and model_unseen_family")
    parser.add_argument("--output", type=Path,
                        default=Path("results/evaluation/evasion-transfer.json"))
    args = parser.parse_args()

    preregistration = yaml.safe_load(args.preregistration.read_text())
    sealed_family = str(preregistration["held_out_evasion_family"])
    runs = normalize_labels(pd.read_csv(args.labels, keep_default_na=False))
    intervals = [
        evaluation_interval(row["trace_path"], int(row["gpu_index"]))
        for _, row in runs.iterrows()
    ]
    runs["evaluation_start_raw_s"] = [interval[0] for interval in intervals]
    runs["evaluation_end_raw_s"] = [interval[1] for interval in intervals]
    runs["evaluation_interval_source"] = [interval[2] for interval in intervals]
    runs["evaluation_duration_hours"] = [
        (interval[1] - interval[0]) / 3600.0 for interval in intervals
    ]
    if args.family_plan:
        family_plan = yaml.safe_load(args.family_plan.read_text())
        seen_families = [str(value) for value in family_plan["seen_families"]]
        target_family = str(family_plan["model_unseen_family"])
        plans = plan_declared_holdout(runs, seen_families, target_family, sealed_family)
        plan_id = str(family_plan.get("split_id", args.family_plan.stem))
    else:
        families = evasion_families(runs, sealed_family)
        if len(families) < 2:
            raise ValueError("pairwise transfer requires at least two non-sealed evasion families")
        plans = plan_comparisons(runs, sealed_family)
        plan_id = "all_development_families"
    windows = extract_all_windows(runs, args.window_s, args.stride_s)
    names = feature_names()
    s2_names = stage2_names(names)

    summaries = []
    predictions = []
    for plan in plans:
        summary, rows = evaluate_plan(plan, runs, windows, names, s2_names)
        summaries.append(summary)
        predictions.extend(rows)
        print(summary)

    summary_frame = pd.DataFrame(summaries)
    evaluated_families = sorted(summary_frame["target_family"].unique())
    loo = summary_frame[summary_frame["protocol"].isin(["leave_one_out", "declared_two_seen"])]
    worst_family_tpr = float(loo["target_tpr"].min()) if not loo.empty else math.nan

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_name(args.output.stem + "_matrix.csv")
    predictions_path = args.output.with_name(args.output.stem + "_run_predictions.csv")
    summary_frame.to_csv(summary_path, index=False)
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    args.output.write_text(json.dumps({
        "status": "diagnostic_development_evaluation",
        "plan_id": plan_id,
        "sealed_family_excluded": sealed_family,
        "evaluated_families": evaluated_families,
        "config": {
            "window_s": args.window_s,
            "stride_s": args.stride_s,
            "rf": RF_KW,
            "threshold": THRESHOLD,
            "rule": f"{K_OF_N[0]}of{K_OF_N[1]}",
        },
        "leave_one_out_worst_family_tpr": worst_family_tpr,
        "comparisons": summaries,
        "artifacts": {
            "matrix": str(summary_path),
            "run_predictions": str(predictions_path),
        },
    }, indent=2, default=str))
    print(f"Wrote {args.output}, {summary_path}, and {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
