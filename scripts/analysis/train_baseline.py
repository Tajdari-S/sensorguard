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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_run, feature_names, stage2_names  # noqa: E402

RF_KW = dict(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
             class_weight="balanced", n_jobs=-1, random_state=0)
THRESHOLD = 0.75
K_OF_N = (3, 5)


def run_alert(p_training: np.ndarray) -> bool:
    k, n = K_OF_N
    if len(p_training) < n:
        return bool((p_training >= THRESHOLD).sum() >= k)
    hits = (p_training >= THRESHOLD).astype(int)
    windowed = np.convolve(hits, np.ones(n, dtype=int), mode="valid")
    return bool((windowed >= k).any())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--window-s", type=int, default=30)
    parser.add_argument("--stride-s", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/e2_baseline.json"))
    args = parser.parse_args()

    runs = pd.read_csv(args.labels)
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
        frames.append(feats)
    data = pd.concat(frames, ignore_index=True)
    X = data[names].to_numpy()
    y_ml = data["label"].isin(["training", "inference"]).to_numpy()
    y_train = (data["label"] == "training").to_numpy()
    groups = data["run_id"].to_numpy()

    gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    fold_rows = []
    for fold, (tr, te) in enumerate(gkf.split(X, y_train, groups)):
        rf1 = RandomForestClassifier(**RF_KW).fit(X[tr], y_ml[tr])
        s2_idx = [names.index(n) for n in s2_names]
        ml_tr = tr[y_ml[tr]]
        rf2 = RandomForestClassifier(**RF_KW).fit(X[ml_tr][:, s2_idx], y_train[ml_tr])

        p_ml = rf1.predict_proba(X[te])[:, 1]
        p_tr_given_ml = rf2.predict_proba(X[te][:, s2_idx])[:, 1]
        p_training = p_ml * p_tr_given_ml

        te_runs = groups[te]
        run_metrics = []
        for run_id in np.unique(te_runs):
            mask = te_runs == run_id
            truth = bool(y_train[te][mask][0])
            alert = run_alert(p_training[mask])
            run_metrics.append((run_id, truth, alert))
        tp = sum(1 for _, t, a in run_metrics if t and a)
        fn = sum(1 for _, t, a in run_metrics if t and not a)
        fp = sum(1 for _, t, a in run_metrics if not t and a)
        tn = sum(1 for _, t, a in run_metrics if not t and not a)
        fold_rows.append({
            "fold": fold,
            "window_auc_training": float(roc_auc_score(y_train[te], p_training))
            if len(np.unique(y_train[te])) > 1 else None,
            "runs": len(run_metrics), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "run_tpr": tp / (tp + fn) if tp + fn else None,
            "run_fpr": fp / (fp + tn) if fp + tn else None,
        })
        print(fold_rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {"window_s": args.window_s, "stride_s": args.stride_s,
                   "rf": RF_KW, "threshold": THRESHOLD, "rule": f"{K_OF_N[0]}of{K_OF_N[1]}",
                   "n_features_stage1": len(names), "n_features_stage2": len(s2_names)},
        "folds": fold_rows,
    }, indent=2, default=str))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
