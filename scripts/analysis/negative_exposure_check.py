#!/usr/bin/env python3
"""R06 negative-exposure accounting + false-alert DIAGNOSTIC.

Sums completed negative (non-ML) GPU-hours from run manifests, then trains
the two-stage RF on the development corpus (all non-`neg-` completed runs)
and applies it to every negative run, counting run-level training alerts
under the amended fixed rule (3-of-5 windows at P>=0.85).

This is a development-time DIAGNOSTIC, not the frozen-model evaluation: the
model here is trained on the current dev corpus, not the sealed one. Zero
false alerts over the exposure supports the R06 bound; the authoritative
operating curve comes from the frozen model at Gate 2.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_run, feature_names, stage2_names  # noqa: E402

RF_KW = dict(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
             class_weight="balanced", n_jobs=-1, random_state=0)
THRESHOLD = 0.85
K, N = 3, 5


def run_alert(p):
    hits = (np.asarray(p) >= THRESHOLD).astype(int)
    if len(hits) < N:
        return int(hits.sum()) >= K
    return bool((np.convolve(hits, np.ones(N, int), "valid") >= K).any())


def load_runs():
    dev, neg = [], []
    for m in sorted(glob.glob("data/runs/*/manifest.yaml")):
        d = yaml.safe_load(open(m))
        if d.get("status") != "completed":
            continue
        fam = d["workload"]["family"]
        label = fam if fam in ("training", "inference") else "non_ml"
        rec = {"run_id": d["run_id"], "trace": str(Path(m).parent / "nvml.csv"),
               "gpu_index": d["hardware"]["gpu_index_under_test"], "label": label,
               "dur": (d.get("workload") or {}).get("duration_s") or 0}
        if "neg-" in d["run_id"]:
            neg.append(rec)
        else:
            dev.append(rec)
    return dev, neg


def features_for(runs, names):
    X, y_ml, y_tr, ids = [], [], [], []
    for r in runs:
        f = extract_run(r["trace"], int(r["gpu_index"]))
        if f.empty:
            continue
        X.append(f[names].to_numpy())
        y_ml.append(np.full(len(f), r["label"] in ("training", "inference")))
        y_tr.append(np.full(len(f), r["label"] == "training"))
        ids.append(np.full(len(f), r["run_id"]))
    if not X:
        return (np.empty((0, len(names))),) * 4
    return np.vstack(X), np.concatenate(y_ml), np.concatenate(y_tr), np.concatenate(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results/negative_exposure_check.json"))
    args = ap.parse_args()

    dev, neg = load_runs()
    neg_hours = sum(r["dur"] for r in neg) / 3600.0
    names = feature_names()
    s2 = stage2_names(names)
    s2i = [names.index(n) for n in s2]

    # Train two-stage RF on the development corpus.
    Xd, yml, ytr, _ = features_for(dev, names)
    rf1 = RandomForestClassifier(**RF_KW).fit(Xd, yml)
    ml = yml
    rf2 = RandomForestClassifier(**RF_KW).fit(Xd[ml][:, s2i], ytr[ml])

    # Apply to each negative run; count run-level training alerts.
    false_alerts = []
    checked = 0
    for r in neg:
        f = extract_run(r["trace"], int(r["gpu_index"]))
        if f.empty:
            continue
        checked += 1
        X = f[names].to_numpy()
        p = rf1.predict_proba(X)[:, 1] * rf2.predict_proba(X[:, s2i])[:, 1]
        if run_alert(p):
            false_alerts.append({"run_id": r["run_id"], "max_p": round(float(p.max()), 3)})

    out = {
        "negative_completed_runs": len(neg),
        "negative_runs_checked": checked,
        "negative_GPU_hours": round(neg_hours, 2),
        "dev_corpus_runs": len(dev),
        "rule": f"{K}-of-{N} at P>={THRESHOLD}",
        "false_alerts": len(false_alerts),
        "false_alert_detail": false_alerts,
        "note": "DIAGNOSTIC: model trained on current dev corpus, not the frozen "
                "sealed model. Authoritative operating curve is a Gate-2 artifact.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "false_alert_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
