#!/usr/bin/env python3
"""Summarize audited node development runs, useful work, and NVML energy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workload_interval(manifest: dict) -> tuple[float, float, str]:
    workload = manifest["workload"]
    if workload.get("start_raw_s") is not None and workload.get("end_raw_s") is not None:
        return float(workload["start_raw_s"]), float(workload["end_raw_s"]), "exact_manifest"
    pre = manifest.get("marker_bursts", {}).get("pre", [])
    if not pre:
        raise ValueError("manifest has no exact interval or pre-marker burst")
    # load_marker sleeps its fixed two-second gap after the final burst before
    # returning to the supervisor.  This is retained only for early runs made
    # before exact raw endpoints were added to the schema.
    start = max(float(burst[1]) for burst in pre) + 2.0
    end = start + float(workload["duration_s"])
    return start, end, "inferred_from_pre_marker_plus_2s_gap"


def nvml_energy(path: Path, start: float, end: float) -> tuple[int, float, float]:
    frame = pd.read_csv(path)
    good = frame[
        (frame["status"] == "ok")
        & (frame["t_target_raw_s"] >= start)
        & (frame["t_target_raw_s"] <= end)
    ].copy()
    if good.empty:
        return 0, float("nan"), float("nan")
    per_gpu = []
    for _, group in good.groupby("gpu_index"):
        group = group.sort_values("t_target_raw_s")
        if len(group) == 1:
            energy_ws = float(group["power_w"].iloc[0])
        else:
            energy_ws = float(
                ((group["power_w"].iloc[:-1].to_numpy() + group["power_w"].iloc[1:].to_numpy())
                 * 0.5
                 * group["t_target_raw_s"].diff().iloc[1:].to_numpy()).sum()
            )
        per_gpu.append(energy_ws)
    energy_wh = sum(per_gpu) / 3600.0
    duration = max(end - start, 1e-9)
    mean_total_power = energy_wh * 3600.0 / duration
    return len(good), energy_wh, mean_total_power


def summarize(manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text())
    run_dir = manifest_path.parent
    useful_path = run_dir / "useful_work.json"
    useful = json.loads(useful_path.read_text()) if useful_path.is_file() else {}
    checksum_errors = []
    for name, expected in (manifest.get("artifact_checksums") or {}).items():
        artifact = run_dir / name
        if not artifact.is_file():
            checksum_errors.append(f"missing:{name}")
        elif sha256(artifact) != expected:
            checksum_errors.append(f"checksum:{name}")
    start, end, timing_source = workload_interval(manifest)
    samples, energy_wh, mean_power_w = nvml_energy(run_dir / "nvml.csv", start, end)
    channels = manifest.get("sensor_channels", [])
    nvml = next((channel for channel in channels if channel["channel_id"].startswith("nvml.")), {})
    dcgm = next((channel for channel in channels if channel["channel_id"].startswith("dcgm.")), {})
    progress = useful.get("relative_loss_reduction")
    energy_per_progress = (
        energy_wh / progress if progress is not None and progress > 0 else None
    )
    valid = (
        manifest.get("status") == "completed"
        and bool(channels)
        and all(channel.get("health") == "pass" for channel in channels)
        and not checksum_errors
        and (useful.get("meaningful_optimization_progress") is True
             or useful.get("mode") == "inference_control")
    )
    return {
        "run_id": manifest["run_id"],
        "mode": useful.get("mode", manifest["workload"].get("name")),
        "valid": valid,
        "manifest_status": manifest.get("status"),
        "git_commit": manifest.get("git_commit"),
        "gpu_indices": manifest.get("hardware", {}).get("gpu_set"),
        "gpu_uuid_primary": manifest.get("hardware", {}).get("gpu_uuid_under_test"),
        "duration_s": manifest["workload"].get("duration_s"),
        "timing_source": timing_source,
        "steps": useful.get("steps"),
        "inference_steps": useful.get("inference_steps"),
        "migrations": useful.get("migrations"),
        "initial_loss": useful.get("initial_loss"),
        "final_loss": useful.get("final_loss"),
        "relative_loss_reduction": progress,
        "max_weight_change": useful.get("max_weight_change"),
        "meaningful_optimization_progress": useful.get("meaningful_optimization_progress"),
        "nvml_workload_samples": samples,
        "nvml_energy_wh": energy_wh,
        "mean_total_gpu_power_w": mean_power_w,
        "energy_wh_per_relative_loss_reduction": energy_per_progress,
        "nvml_health": nvml.get("health"),
        "nvml_missing_fraction": nvml.get("missing_fraction"),
        "nvml_alignment_error_ms": nvml.get("alignment_error_ms"),
        "nvml_edge_latency_ms": nvml.get("edge_latency_ms"),
        "marker_bursts_detected": nvml.get("bursts_detected"),
        "marker_bursts_expected": nvml.get("bursts_expected"),
        "dcgm_health": dcgm.get("health"),
        "checksum_errors": ";".join(checksum_errors),
        "manifest_path": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = []
    for item in args.inputs:
        manifests.extend(sorted(item.rglob("manifest.yaml")) if item.is_dir() else [item])
    rows = [summarize(path) for path in manifests]
    if not rows:
        parser.error("no manifests found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "runs": len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if all(row["valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
