#!/usr/bin/env python3
"""E2: two-stage random-forest NVML baseline with grouped, leakage-safe splits.

Input: a labels CSV with one row per run:
    run_id,trace_path,gpu_index,label,family,gpu_uuid,collection_day
where label is one of {training, inference, non_ml} and trace_path points to
the run's nvml.csv. Windows are extracted per run (30 s / 15 s causal),
split ALWAYS by run (never by window), grouped by gpu_uuid/day/family.

Stage 1: ML (training+inference) vs non-ML, all 166 features.
Stage 2: training vs inference, level features dropped.
Run-level alert: 3-of-5 consecutive windows with P(training) >= 0.75.

Outputs per-fold window accuracy/AUC and run-level TPR/FPR.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_run, feature_names, stage2_names  # noqa: E402
from evaluation import (  # noqa: E402
    assert_group_disjoint,
    poisson_zero_event_upper,
    run_alert,
    validate_labels,
    wilson_interval,
)

RF_KW = dict(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
             class_weight="balanced", n_jobs=-1, random_state=0)
THRESHOLD = 0.75
K_OF_N = (3, 5)


def trace_duration_hours(trace_path: str, gpu_index: int) -> float:
    trace = pd.read_csv(trace_path)
    trace = trace[(trace["status"] == "ok") & (trace["gpu_index"] == gpu_index)]
    if len(trace) < 2:
        return 0.0
    return float(trace["t_target_raw_s"].iloc[-1] - trace["t_target_raw_s"].iloc[0]) / 3600.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--window-s", type=int, default=30)
    parser.add_argument("--stride-s", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--group-by",
        default="run_id",
        help=("One labels-CSV column whose values may not cross folds. "
              "Use family, gpu_uuid, and collection_day in separate "
              "generalization runs; run_id is diagnostic only."),
    )
    parser.add_argument("--output", type=Path, default=Path("results/e2_baseline.json"))
    args = parser.parse_args()

    runs = pd.read_csv(args.labels)
    validate_labels(runs, args.group_by)
    if args.group_by == "run_id":
        print("WARNING: run_id grouping prevents window leakage but is not a transfer test", file=sys.stderr)

    # Import the modeling stack only after the data contract is validated. This
    # keeps validation/test helpers usable on acquisition hosts.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    names = feature_names()
    s2_names = stage2_names(names)

    frames = []
    for _, r in runs.iterrows():
        feats = extract_run(r["trace_path"], int(r["gpu_index"]),
                            args.window_s, args.stride_s)
        if feats.empty:
            print(f"WARNING: no windows for {r['run_id']}", file=sys.stderr)
            continue
        feats["run_id"] = r["run_id"]
        feats["label"] = r["label"]
        feats["family"] = r["family"]
        feats["gpu_uuid"] = r["gpu_uuid"]
        feats["collection_day"] = r["collection_day"]
        feats["split_group"] = r[args.group_by]
        frames.append(feats)
    data = pd.concat(frames, ignore_index=True)
    X = data[names].to_numpy()
    y_ml = data["label"].isin(["training", "inference"]).to_numpy()
    y_train = (data["label"] == "training").to_numpy()
    groups = data["split_group"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    fold_rows = []
    run_rows = []
    window_rows = []
    run_ids = data["run_id"].to_numpy()
    for fold, (tr, te) in enumerate(gkf.split(X, y_train, groups)):
        assert_group_disjoint(tr, te, groups)
        ml_tr = tr[y_ml[tr]]
        held_out_groups = sorted(set(groups[te]))
        if len(np.unique(y_ml[tr])) < 2 or len(ml_tr) == 0 or len(np.unique(y_train[ml_tr])) < 2:
            fold_rows.append({
                "fold": fold,
                "status": "not_estimable",
                "reason": "training fold lacks a required stage-1 or stage-2 class",
                "held_out_groups": held_out_groups,
                "runs": int(len(np.unique(run_ids[te]))),
                "window_auc_training": None,
                "tp": None, "fn": None, "fp": None, "tn": None,
                "run_tpr": None, "run_fpr": None,
            })
            print(fold_rows[-1])
            continue
        rf1 = RandomForestClassifier(**RF_KW).fit(X[tr], y_ml[tr])
        s2_idx = [names.index(n) for n in s2_names]
        rf2 = RandomForestClassifier(**RF_KW).fit(X[ml_tr][:, s2_idx], y_train[ml_tr])

        p_ml = rf1.predict_proba(X[te])[:, 1]
        p_tr_given_ml = rf2.predict_proba(X[te][:, s2_idx])[:, 1]
        p_training = p_ml * p_tr_given_ml

        te_runs = run_ids[te]
        run_metrics = []
        for run_id in np.unique(te_runs):
            mask = te_runs == run_id
            truth = bool(y_train[te][mask][0])
            ordered = np.argsort(data.loc[te[mask], "window_end_raw_s"].to_numpy())
            run_probabilities = p_training[mask][ordered]
            alert = run_alert(run_probabilities, THRESHOLD, *K_OF_N)
            run_meta = runs.loc[runs["run_id"] == run_id].iloc[0]
            duration_hours = trace_duration_hours(run_meta["trace_path"], int(run_meta["gpu_index"]))
            run_metrics.append((run_id, truth, alert))
            run_rows.append({
                "fold": fold,
                "run_id": run_id,
                "truth_training": truth,
                "alert": alert,
                "family": run_meta["family"],
                "gpu_uuid": run_meta["gpu_uuid"],
                "collection_day": run_meta["collection_day"],
                "duration_hours": duration_hours,
                "max_p_training": float(np.max(run_probabilities)),
                "mean_p_training": float(np.mean(run_probabilities)),
            })
        for row_index, probability in zip(te, p_training):
            window_rows.append({
                "fold": fold,
                "run_id": data.iloc[row_index]["run_id"],
                "window_end_raw_s": data.iloc[row_index]["window_end_raw_s"],
                "truth_training": bool(y_train[row_index]),
                "p_training": float(probability),
            })
        tp = sum(1 for _, t, a in run_metrics if t and a)
        fn = sum(1 for _, t, a in run_metrics if t and not a)
        fp = sum(1 for _, t, a in run_metrics if not t and a)
        tn = sum(1 for _, t, a in run_metrics if not t and not a)
        fold_rows.append({
            "fold": fold,
            "status": "evaluated",
            "held_out_groups": held_out_groups,
            "window_auc_training": float(roc_auc_score(y_train[te], p_training))
            if len(np.unique(y_train[te])) > 1 else None,
            "runs": len(run_metrics), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "run_tpr": tp / (tp + fn) if tp + fn else None,
            "run_fpr": fp / (fp + tn) if fp + tn else None,
        })
        print(fold_rows[-1])

    run_predictions = pd.DataFrame(run_rows, columns=[
        "fold", "run_id", "truth_training", "alert", "family", "gpu_uuid",
        "collection_day", "duration_hours", "max_p_training", "mean_p_training",
    ])
    tp = int(((run_predictions["truth_training"]) & (run_predictions["alert"])).sum())
    fn = int(((run_predictions["truth_training"]) & (~run_predictions["alert"])).sum())
    fp = int(((~run_predictions["truth_training"]) & (run_predictions["alert"])).sum())
    tn = int(((~run_predictions["truth_training"]) & (~run_predictions["alert"])).sum())
    tpr_ci = wilson_interval(tp, tp + fn)
    negative_exposure = float(run_predictions.loc[~run_predictions["truth_training"], "duration_hours"].sum())
    false_alert_rate = fp / negative_exposure if negative_exposure else None
    zero_event_upper = poisson_zero_event_upper(negative_exposure) if fp == 0 else None
    family_rows = []
    for family, frame in run_predictions[run_predictions["truth_training"]].groupby("family"):
        family_tp = int(frame["alert"].sum())
        family_n = len(frame)
        family_rows.append({
            "family": family,
            "runs": family_n,
            "tpr": family_tp / family_n,
            "tpr_ci_95": wilson_interval(family_tp, family_n),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_path = args.output.with_name(args.output.stem + "_run_predictions.csv")
    window_path = args.output.with_name(args.output.stem + "_window_predictions.csv")
    run_predictions.to_csv(run_path, index=False)
    pd.DataFrame(window_rows).to_csv(window_path, index=False)
    args.output.write_text(json.dumps({
        "config": {"window_s": args.window_s, "stride_s": args.stride_s,
                   "rf": RF_KW, "threshold": THRESHOLD, "rule": f"{K_OF_N[0]}of{K_OF_N[1]}",
                   "n_features_stage1": len(names), "n_features_stage2": len(s2_names),
                   "group_by": args.group_by,
                   "diagnostic_only": args.group_by == "run_id"},
        "folds": fold_rows,
        "aggregate": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "run_tpr": tp / (tp + fn) if tp + fn else None,
            "run_tpr_ci_95": tpr_ci,
            "negative_gpu_hours": negative_exposure,
            "false_alerts_per_gpu_hour": false_alert_rate,
            "zero_event_rate_upper_95": zero_event_upper,
        },
        "per_training_family": family_rows,
        "artifacts": {
            "run_predictions": str(run_path),
            "window_predictions": str(window_path),
        },
    }, indent=2, default=str))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
