#!/usr/bin/env python3
"""Apply the frozen NVML detector to auditable non-training exposure.

Only completed, healthy manifests for the requested host are scored. The
script uses the workload interval bracketed by synchronization markers, keeps
the frozen 30 s/15 s feature contract, and never refits or recalibrates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from analyze_synchronized_physical import THRESHOLD, alert, summaries


NVML_COLUMNS = [
    "util_gpu_pct",
    "util_mem_pct",
    "mem_used_mib",
    "power_w",
    "temp_c",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "pcie_tx_kbps",
    "pcie_rx_kbps",
]
WINDOW_S = 30
STRIDE_S = 15


def paths(inputs: list[Path]) -> list[Path]:
    found = set()
    for item in inputs:
        if item.is_file() and item.name == "manifest.yaml":
            found.add(item)
        elif item.is_dir():
            found.update(item.rglob("manifest.yaml"))
    return sorted(found)


def marker_interval(manifest: dict, trace: pd.DataFrame) -> tuple[float, float]:
    markers = manifest.get("marker_bursts") or {}
    pre, post = markers.get("pre") or [], markers.get("post") or []
    if pre and post:
        return float(pre[-1][1]), float(post[0][0])
    ok = trace[trace.status.eq("ok")]
    start = float(ok.t_target_raw_s.min())
    duration = float((manifest.get("workload") or {}).get("duration_s") or 0)
    return start, start + duration


def feature_windows(path: Path, manifest: dict, gpu: int) -> pd.DataFrame:
    trace = pd.read_csv(path)
    start, end = marker_interval(manifest, trace)
    data = trace[(trace.status.eq("ok")) & (trace.gpu_index.eq(gpu))].copy()
    data = data[
        (data.t_target_raw_s >= start) & (data.t_target_raw_s < end)
    ].copy()
    data["second"] = np.floor(data.t_target_raw_s - start).astype(int)
    rows = []
    usable = int(data.second.max()) + 1 if len(data) else 0
    for index, offset in enumerate(
        range(0, usable - WINDOW_S + 1, STRIDE_S)
    ):
        chunk = data[
            (data.second >= offset) & (data.second < offset + WINDOW_S)
        ]
        if len(chunk) < 20:
            continue
        row = summaries(chunk, NVML_COLUMNS, "nvml_")
        row.update({
            "window_index": index,
            "window_end_s": offset + WINDOW_S,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--host", default="node2")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    observed_hash = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    if (
        args.expected_artifact_sha256
        and observed_hash != args.expected_artifact_sha256
    ):
        raise RuntimeError(
            "frozen artifact hash does not match the freeze manifest"
        )
    frozen = joblib.load(args.artifact)
    if float(frozen.get("threshold", THRESHOLD)) != THRESHOLD:
        raise RuntimeError(
            "artifact threshold differs from the frozen 0.85 contract"
        )
    model, features = frozen["model"], list(frozen["features"])

    run_rows, failures = [], []
    intervals: dict[
        int, list[tuple[pd.Timestamp, pd.Timestamp, str]]
    ] = {}
    for manifest_path in paths(args.inputs):
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        run_id = str(manifest.get("run_id", manifest_path.parent.name))
        if f"_{args.host}-gpu" not in run_id or "neg-" not in run_id:
            continue
        gpu = int(
            (manifest.get("hardware") or {}).get(
                "gpu_index_under_test", -1
            )
        )
        channels = manifest.get("sensor_channels") or []
        nvml_channel = next(
            (
                item
                for item in channels
                if str(item.get("channel_id", "")).startswith("nvml.")
            ),
            None,
        )
        reasons = []
        if manifest.get("status") != "completed":
            reasons.append(f"status={manifest.get('status')}")
        if not nvml_channel or nvml_channel.get("health") != "pass":
            reasons.append("nvml_channel_not_healthy")
        trace_path = manifest_path.parent / "nvml.csv"
        if not trace_path.is_file():
            reasons.append("missing_nvml_csv")
        duration_s = float(
            (manifest.get("workload") or {}).get("duration_s") or 0
        )
        if duration_s <= 0:
            reasons.append("nonpositive_duration")
        if reasons:
            failures.append({
                "run_id": run_id,
                "reason": "|".join(reasons),
            })
            continue

        start = pd.Timestamp(manifest.get("start_utc"))
        end = pd.Timestamp(manifest.get("end_utc"))
        intervals.setdefault(gpu, []).append((start, end, run_id))
        frame = feature_windows(trace_path, manifest, gpu)
        missing = sorted(set(features) - set(frame.columns))
        if frame.empty or missing:
            failures.append({
                "run_id": run_id,
                "reason": (
                    "no_feature_windows"
                    if frame.empty
                    else "missing_features=" + ";".join(missing)
                ),
            })
            continue
        probabilities = model.predict_proba(frame[features].fillna(0))[
            :, list(model.classes_).index(1)
        ]
        detected, tta = alert(probabilities)
        run_rows.append({
            "run_id": run_id,
            "host": args.host,
            "gpu": gpu,
            "workload": (manifest.get("workload") or {}).get("name"),
            "duration_s": duration_s,
            "windows": len(frame),
            "false_alert": int(detected),
            "time_to_false_alert_s": tta,
            "max_probability": float(np.max(probabilities)),
            "mean_probability": float(np.mean(probabilities)),
        })

    overlaps = []
    for gpu, group in intervals.items():
        group.sort()
        for first, second in zip(group, group[1:]):
            if second[0] < first[1]:
                overlaps.append({
                    "gpu": gpu,
                    "first": first[2],
                    "second": second[2],
                })
    if overlaps:
        raise RuntimeError(
            f"same-GPU exposure overlaps detected: {overlaps[:5]}"
        )

    runs = pd.DataFrame(run_rows)
    if runs.empty:
        raise RuntimeError("no eligible negative runs were scored")
    hours = float(runs.duration_s.sum() / 3600.0)
    false_alerts = int(runs.false_alert.sum())
    summary = {
        "detector": "frozen NVML random forest",
        "artifact_sha256": observed_hash,
        "threshold": THRESHOLD,
        "run_rule": "3-of-5",
        "host": args.host,
        "eligible_runs_scored": int(len(runs)),
        "excluded_runs": int(len(failures)),
        "negative_gpu_hours": hours,
        "false_alert_runs": false_alerts,
        "false_alerts_per_gpu_hour": false_alerts / hours,
        "one_sided_95pct_rate_upper_bound_if_zero": (
            -math.log(0.05) / hours if false_alerts == 0 else None
        ),
        "scope": (
            "NVML only; these runs contain no synchronized GPU-current channel"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(
        args.output_dir / "frozen_nvml_negative_run_predictions.csv",
        index=False,
    )
    pd.DataFrame(
        failures, columns=["run_id", "reason"]
    ).to_csv(
        args.output_dir / "frozen_nvml_negative_exclusions.csv",
        index=False,
    )
    (
        args.output_dir / "frozen_nvml_negative_summary.json"
    ).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
