#!/usr/bin/env python3
"""Audit frozen-model integrity and distribution shift on hard-family features.

This audit never fits a model, changes the threshold, or uses hard-family labels
for calibration. It verifies the frozen artifacts, reproduces their scores, and
quantifies whether the new feature values lie outside the development range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analyze_synchronized_physical import META, THRESHOLD, alert


MODEL_FILES = {
    "NVML": "frozen_nvml.joblib",
    "Electrical": "frozen_electrical.joblib",
    "NVML + electrical": "frozen_nvml_plus_electrical.joblib",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_scores(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    rows = []
    for run_id, run in frame.groupby("run_id", sort=True):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-windows", type=Path, required=True)
    parser.add_argument("--nvml-features", type=Path, required=True)
    parser.add_argument("--electrical-features", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--existing-run-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.freeze_manifest.read_text())
    expected_hashes = {
        record["modality"]: record["sha256"] for record in manifest["models"]
    }
    dev = pd.read_csv(args.development_windows)
    nvml = pd.read_csv(args.nvml_features)
    electrical = pd.read_csv(args.electrical_features)
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {
        "NVML": nvml,
        "Electrical": electrical,
        "NVML + electrical": fusion,
    }

    audits, shift_rows, rescored = [], [], []
    for modality, hard in datasets.items():
        artifact_path = args.frozen_dir / MODEL_FILES[modality]
        artifact = joblib.load(artifact_path)
        model, features = artifact["model"], list(artifact["features"])
        missing = sorted(set(features) - set(hard.columns))
        if missing:
            raise RuntimeError(f"{modality} missing frozen features: {missing}")
        development = dev.loc[dev.modality.eq(modality), features]
        if development.empty:
            raise RuntimeError(f"no development rows for {modality}")
        probabilities = model.predict_proba(hard[features].fillna(0))[
            :, list(model.classes_).index(1)
        ]
        scored = hard[list(META)].copy()
        scored["probability"] = probabilities
        run_frame = run_scores(scored, modality)
        rescored.append(run_frame)

        feature_ood = []
        for feature in features:
            train_values = (
                development[feature].replace([np.inf, -np.inf], np.nan).dropna()
            )
            hard_values = hard[feature].replace([np.inf, -np.inf], np.nan)
            if train_values.empty:
                outside = pd.Series(False, index=hard.index)
                low = high = np.nan
            else:
                low, high = float(train_values.min()), float(train_values.max())
                outside = (hard_values < low) | (hard_values > high)
            fraction = float(outside.fillna(False).mean())
            feature_ood.append(fraction)
            shift_rows.append({
                "modality": modality,
                "feature": feature,
                "development_min": low,
                "development_max": high,
                "hard_min": float(hard_values.min()),
                "hard_max": float(hard_values.max()),
                "hard_outside_development_fraction": fraction,
                "hard_missing_fraction": float(hard_values.isna().mean()),
            })

        positives = run_frame[run_frame.target.eq(1)]
        negatives = run_frame[run_frame.target.eq(0)]
        observed_hash = sha256(artifact_path)
        audits.append({
            "modality": modality,
            "artifact": str(artifact_path),
            "expected_sha256": expected_hashes[modality],
            "observed_sha256": observed_hash,
            "artifact_hash_matches": observed_hash == expected_hashes[modality],
            "frozen_threshold": float(artifact.get("threshold", THRESHOLD)),
            "frozen_run_rule": artifact.get("run_rule"),
            "hard_runs": int(run_frame.run_id.nunique()),
            "hard_windows": int(len(scored)),
            "positive_runs": int(len(positives)),
            "negative_runs": int(len(negatives)),
            "positive_alerts": int(positives.alert.sum()),
            "negative_alerts": int(negatives.alert.sum()),
            "positive_mean_run_probability": float(
                positives.mean_probability.mean()
            ),
            "negative_mean_run_probability": float(
                negatives.mean_probability.mean()
            ),
            "positive_minus_negative_mean_probability": float(
                positives.mean_probability.mean()
                - negatives.mean_probability.mean()
            ),
            "mean_feature_outside_development_fraction": float(
                np.mean(feature_ood)
            ),
            "features_over_50pct_outside_development": int(
                np.sum(np.asarray(feature_ood) > 0.5)
            ),
            "hard_feature_missing_cells": int(
                hard[features].isna().sum().sum()
            ),
        })

    rescored_frame = pd.concat(rescored, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    if args.existing_run_predictions:
        prior = pd.read_csv(args.existing_run_predictions)
        prior = prior[prior.modality.isin(MODEL_FILES)]
        compare = rescored_frame.merge(
            prior,
            on=["modality", "run_id"],
            suffixes=("_audit", "_existing"),
            validate="one_to_one",
        )
        by_modality = {
            modality: float(
                np.max(
                    np.abs(
                        group.mean_probability_audit
                        - group.mean_probability_existing
                    )
                )
            )
            for modality, group in compare.groupby("modality")
        }
        audit_frame["max_abs_reproduction_difference"] = (
            audit_frame.modality.map(by_modality)
        )
    else:
        audit_frame["max_abs_reproduction_difference"] = np.nan

    integrity_pass = bool(
        audit_frame.artifact_hash_matches.all()
        and audit_frame.hard_feature_missing_cells.eq(0).all()
        and audit_frame.max_abs_reproduction_difference.fillna(0).le(1e-12).all()
    )
    conclusion = {
        "integrity_pass": integrity_pass,
        "labels_used_for_fit_calibration_or_threshold_selection": False,
        "threshold": THRESHOLD,
        "run_rule": "3-of-5",
        "interpretation": (
            "Frozen artifacts and feature contracts reproduce exactly; the poor "
            "hard-family separation is an observed detector result, not an "
            "unfinished analysis."
            if integrity_pass
            else "At least one artifact, feature, or reproduction check failed."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(
        args.output_dir / "hard_family_contract_audit.csv", index=False
    )
    pd.DataFrame(shift_rows).to_csv(
        args.output_dir / "hard_family_feature_shift.csv", index=False
    )
    rescored_frame.to_csv(
        args.output_dir / "hard_family_rescored_predictions.csv", index=False
    )
    (args.output_dir / "hard_family_audit_conclusion.json").write_text(
        json.dumps(conclusion, indent=2) + "\n"
    )
    print(audit_frame.to_string(index=False))
    print(json.dumps(conclusion, indent=2))
    return 0 if integrity_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
