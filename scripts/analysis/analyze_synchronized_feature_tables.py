#!/usr/bin/env python3
"""Train, evaluate, and freeze synchronized models from compact feature tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_synchronized_physical import (
    META,
    evaluate,
    freeze,
    summarize,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-status", type=Path, required=True)
    parser.add_argument("--verifier-status", type=Path, required=True)
    parser.add_argument("--nvml-features", type=Path, required=True)
    parser.add_argument("--electrical-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    expected_runs = [record["run_id"] for record in plan["runs"]]
    node_status = pd.read_csv(args.node_status)
    verifier_status = pd.read_csv(args.verifier_status)
    for role, status in (
        ("node", node_status),
        ("verifier", verifier_status),
    ):
        if set(status.run_id) != set(expected_runs):
            raise RuntimeError(f"{role} status does not match plan")
        if len(status) != len(expected_runs) or (status.return_code != 0).any():
            raise RuntimeError(f"{role} status contains incomplete/failed cells")

    nvml = pd.read_csv(args.nvml_features)
    electrical = pd.read_csv(args.electrical_features)
    for modality, frame in (("NVML", nvml), ("Electrical", electrical)):
        if set(frame.run_id) != set(expected_runs):
            raise RuntimeError(f"{modality} feature table does not match plan")
        if frame[list(META)].isna().any().any():
            raise RuntimeError(f"{modality} metadata contains missing values")
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {
        "NVML": nvml,
        "Electrical": electrical,
        "NVML + electrical": fusion,
    }
    evaluated = [evaluate(data, name) for name, data in datasets.items()]
    runs = pd.concat([item[0] for item in evaluated], ignore_index=True)
    windows = pd.concat([item[1] for item in evaluated], ignore_index=True)
    summary = summarize(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "run_predictions.csv", index=False)
    windows.to_csv(args.output_dir / "window_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "final_metrics.csv", index=False)
    audit = node_status.merge(
        verifier_status,
        on="run_id",
        suffixes=("_node", "_verifier"),
        validate="one_to_one",
    )
    audit["both_sides_ok"] = (
        audit.return_code_node.eq(0) & audit.return_code_verifier.eq(0)
    )
    audit.to_csv(args.output_dir / "collection_audit.csv", index=False)
    manifests = [
        freeze(data, name, args.output_dir, expected_runs)
        for name, data in datasets.items()
    ]
    frozen = {
        "protocol": {
            "window_s": 30,
            "stride_s": 15,
            "threshold": 0.85,
            "run_rule": "3-of-5",
            "split": "leave-one-workload-family-out",
        },
        "source_contract": {
            "plan_sha256": digest(args.plan),
            "node_status_sha256": digest(args.node_status),
            "verifier_status_sha256": digest(args.verifier_status),
            "nvml_features_sha256": digest(args.nvml_features),
            "electrical_features_sha256": digest(
                args.electrical_features
            ),
            "expected_cuda_uuid": plan.get("expected_cuda_uuid"),
            "cuda_visible_devices": plan.get("cuda_visible_devices"),
        },
        "models": manifests,
    }
    (args.output_dir / "freeze_manifest.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(frozen["source_contract"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
