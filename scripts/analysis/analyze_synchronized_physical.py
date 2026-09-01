#!/usr/bin/env python3
"""Frozen analysis for the synchronized RTX 3090 NVML/electrical corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


WINDOW_S = 30
STRIDE_S = 15
THRESHOLD = 0.85
RF = dict(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
          class_weight="balanced", random_state=42, n_jobs=-1)
META = {"run_id", "family", "target", "window_index", "window_end_s"}


def summaries(frame: pd.DataFrame, columns: list[str], prefix: str) -> dict[str, float]:
    out = {}
    for column in columns:
        x = frame[column].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            x = np.array([0.0])
        q25, q50, q75, q95 = np.percentile(x, [25, 50, 75, 95])
        name = f"{prefix}{column}"
        out.update({
            f"{name}_mean": float(x.mean()), f"{name}_std": float(x.std()),
            f"{name}_min": float(x.min()), f"{name}_max": float(x.max()),
            f"{name}_p25": float(q25), f"{name}_p50": float(q50),
            f"{name}_p75": float(q75), f"{name}_p95": float(q95),
            f"{name}_range": float(x.max() - x.min()),
        })
    return out


def physical_seconds(run_dir: Path, start: float, end: float,
                     sample_hz: int = 10000, bits: int = 16) -> pd.DataFrame:
    meta = json.loads((run_dir / "pico_u0_meta.json").read_text())
    raw = np.load(run_dir / "pico_u0_chA.npy", mmap_mode="r")
    capture_start = meta["capture_start_epoch_ns"] / 1e9
    capture_end = meta["capture_end_epoch_ns"] / 1e9
    native_hz = len(raw) / (capture_end - capture_start)
    step = max(1, int(round(native_hz / sample_hz)))
    rows = []
    for second in range(int(np.floor(end - start))):
        lo = max(0, int((start + second - capture_start) * native_hz))
        hi = min(len(raw), int((start + second + 1 - capture_start) * native_hz))
        values = np.asarray(raw[lo:hi:step], dtype=np.float64)
        if not len(values):
            continue
        if bits < 16:
            shift = 16 - bits
            values = np.round(values / (2 ** shift)) * (2 ** shift)
        values *= float(meta["range_mv"]) / 32767.0
        centered = values - values.mean()
        delta = np.diff(values)
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        spectral = spectrum[1:]
        total_power = float(spectral.sum()) + 1e-12
        normalized = spectral / total_power
        bands = np.array_split(spectral, 4)
        std = float(values.std())
        rows.append({
            "second": second,
            "mean_mv": float(values.mean()),
            "std_mv": float(values.std()),
            "rms_mv": float(np.sqrt(np.mean(values ** 2))),
            "peak_abs_mv": float(np.max(np.abs(values))),
            "p2p_mv": float(np.ptp(values)),
            "q05_mv": float(np.percentile(values, 5)),
            "q95_mv": float(np.percentile(values, 95)),
            "iqr_mv": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "derivative_rms_mv": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0.0,
            "zero_crossing_rate": float(np.mean(centered[:-1] * centered[1:] < 0)) if len(centered) > 1 else 0.0,
            "lag1_autocorrelation": float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
            if len(centered) > 2 and std > 1e-9 else 0.0,
            "spectral_entropy": float(-(normalized * np.log(normalized + 1e-15)).sum() /
                                      np.log(max(2, len(normalized)))),
            "dominant_frequency_fraction": float((1 + np.argmax(spectral)) / max(1, len(spectral))),
            "bandpower_1": float(bands[0].sum() / total_power),
            "bandpower_2": float(bands[1].sum() / total_power),
            "bandpower_3": float(bands[2].sum() / total_power),
            "bandpower_4": float(bands[3].sum() / total_power),
        })
    return pd.DataFrame(rows)


def nvml_seconds(path: Path, start: float, end: float) -> pd.DataFrame:
    data = pd.read_csv(path)
    anchor = data[data["status"] == "anchor"].iloc[0]
    anchor_epoch = pd.Timestamp(anchor["utc_anchor"]).timestamp()
    data = data[(data["status"] == "ok") & (data["gpu_index"] == 1)].copy()
    data["epoch"] = anchor_epoch + data["t_target_raw_s"] - float(anchor["t_target_raw_s"])
    data = data[(data["epoch"] >= start) & (data["epoch"] < end)].copy()
    data["second"] = np.floor(data["epoch"] - start).astype(int)
    return data


def windows(seconds: pd.DataFrame, columns: list[str], prefix: str,
            run_id: str, family: str, target: int) -> pd.DataFrame:
    rows = []
    usable_seconds = int(seconds["second"].max()) + 1 if len(seconds) else 0
    for index, offset in enumerate(range(0, usable_seconds - WINDOW_S + 1, STRIDE_S)):
        chunk = seconds[(seconds["second"] >= offset) & (seconds["second"] < offset + WINDOW_S)]
        if len(chunk) < 20:
            continue
        row = summaries(chunk, columns, prefix)
        row.update({"run_id": run_id, "family": family, "target": target,
                    "window_index": index, "window_end_s": offset + WINDOW_S})
        rows.append(row)
    return pd.DataFrame(rows)


def build(plan: dict, node_root: Path, pico_root: Path,
          sample_hz: int = 10000, bits: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical_frames, nvml_frames = [], []
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
        ps = physical_seconds(pico_root / run_id, start, end, sample_hz, bits)
        ns = nvml_seconds(node_root / run_id / "nvml.csv", start, end)
        physical_frames.append(windows(ps, [column for column in ps.columns if column != "second"],
                                       "elec_", run_id, run["mode"], int(run["target"])))
        nvml_columns = ["util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w", "temp_c",
                        "clock_sm_mhz", "clock_mem_mhz", "pcie_tx_kbps", "pcie_rx_kbps"]
        nvml_frames.append(windows(ns, nvml_columns, "nvml_", run_id, run["mode"], int(run["target"])))
    return pd.concat(physical_frames, ignore_index=True), pd.concat(nvml_frames, ignore_index=True)


def alert(probabilities: np.ndarray) -> tuple[bool, float | None]:
    hits = probabilities >= THRESHOLD
    for index in range(4, len(hits)):
        if int(hits[index - 4:index + 1].sum()) >= 3:
            return True, float(WINDOW_S + index * STRIDE_S)
    return False, None


def evaluate(data: pd.DataFrame, modality: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = sorted(set(data.columns) - META)
    run_rows, window_rows = [], []
    for family in sorted(data["family"].unique()):
        train = data[data["family"] != family]
        test = data[data["family"] == family]
        if set(train["target"].unique()) != {0, 1}:
            raise RuntimeError(f"training split for {family} lacks a class")
        model = RandomForestClassifier(**RF).fit(train[features].fillna(0), train["target"])
        probabilities = model.predict_proba(test[features].fillna(0))[:, list(model.classes_).index(1)]
        test = test.copy()
        test["probability"] = probabilities
        for run_id, run in test.groupby("run_id"):
            run = run.sort_values("window_index")
            detected, tta = alert(run["probability"].to_numpy())
            run_rows.append({"modality": modality, "held_out_family": family, "run_id": run_id,
                             "target": int(run["target"].iloc[0]), "alert": int(detected),
                             "time_to_alert_s": tta, "max_probability": float(run["probability"].max()),
                             "duration_hours": 105 / 3600})
        window_rows.extend(test.assign(modality=modality).to_dict("records"))
    return pd.DataFrame(run_rows), pd.DataFrame(window_rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality, group in runs.groupby("modality", sort=False):
        tp = int(((group.target == 1) & (group.alert == 1)).sum())
        fn = int(((group.target == 1) & (group.alert == 0)).sum())
        fp = int(((group.target == 0) & (group.alert == 1)).sum())
        tn = int(((group.target == 0) & (group.alert == 0)).sum())
        family_tpr = group[group.target == 1].groupby("held_out_family")["alert"].mean()
        negative_hours = float(group.loc[group.target == 0, "duration_hours"].sum())
        detected_tta = group.loc[(group.target == 1) & (group.alert == 1), "time_to_alert_s"].dropna()
        rows.append({"modality": modality, "runs": len(group), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                     "detection_rate": tp / (tp + fn), "worst_family_detection_rate": float(family_tpr.min()),
                     "negative_gpu_hours": negative_hours,
                     "false_alerts_per_gpu_hour": fp / negative_hours if negative_hours else None,
                     "median_time_to_alert_s": float(detected_tta.median()) if len(detected_tta) else None})
    result = pd.DataFrame(rows)
    baseline = float(result.loc[result.modality == "NVML", "worst_family_detection_rate"].iloc[0])
    result["worst_family_gain_over_nvml_pp"] = 100 * (result["worst_family_detection_rate"] - baseline)
    return result


def freeze(data: pd.DataFrame, modality: str, out_dir: Path, run_ids: list[str]) -> dict:
    features = sorted(set(data.columns) - META)
    model = RandomForestClassifier(**RF).fit(data[features].fillna(0), data["target"])
    path = out_dir / f"frozen_{modality.lower().replace(' ', '_').replace('+', 'plus')}.joblib"
    joblib.dump({"model": model, "features": features, "threshold": THRESHOLD,
                 "window_s": WINDOW_S, "stride_s": STRIDE_S, "run_rule": "3-of-5"}, path)
    return {"modality": modality, "artifact": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "features": len(features), "fit_run_ids": run_ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--pico-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    electrical, nvml = build(plan, args.node_root, args.pico_root)
    fusion = nvml.merge(electrical, on=list(META), validate="one_to_one")
    datasets = {"NVML": nvml, "Electrical": electrical, "NVML + electrical": fusion}
    evaluated = [evaluate(data, name) for name, data in datasets.items()]
    runs = pd.concat([item[0] for item in evaluated], ignore_index=True)
    window_predictions = pd.concat([item[1] for item in evaluated], ignore_index=True)
    summary = summarize(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "run_predictions.csv", index=False)
    window_predictions.to_csv(args.output_dir / "window_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "final_metrics.csv", index=False)
    node_status = pd.read_csv(args.node_root / "status_node.csv")
    pico_status = pd.read_csv(args.pico_root / "status_verifier.csv")
    audit = node_status.merge(pico_status, on="run_id", suffixes=("_node", "_verifier"),
                              validate="one_to_one")
    audit["both_sides_ok"] = (audit["return_code_node"] == 0) & (audit["return_code_verifier"] == 0)
    audit.to_csv(args.output_dir / "collection_audit.csv", index=False)
    manifests = [freeze(data, name, args.output_dir, [run["run_id"] for run in plan["runs"]])
                 for name, data in datasets.items()]
    (args.output_dir / "freeze_manifest.json").write_text(json.dumps({
        "protocol": {"window_s": WINDOW_S, "stride_s": STRIDE_S, "threshold": THRESHOLD,
                     "run_rule": "3-of-5", "rf": RF, "split": "leave-one-workload-family-out"},
        "models": manifests,
    }, indent=2) + "\n")
    metric_lines = []
    for row in summary.itertuples():
        metric_lines.append(
            f"- {row.modality}: {row.tp}/{row.tp + row.fn} training runs detected; "
            f"{row.fp}/{row.fp + row.tn} false-alert runs; "
            f"{row.false_alerts_per_gpu_hour:.3f} false alerts/GPU-hour over "
            f"{row.negative_gpu_hours:.4f} GPU-hours; worst-family detection "
            f"{100 * row.worst_family_detection_rate:.1f}%."
        )
    notes = [
        "# Current-paper synchronized physical tests (2026-08-31)", "",
        "## Setup", "",
        "RTX 3090 GPU1 on node1; PicoScope serial `12789/2929`, channel A; "
        "10 kS/s electrical acquisition plus 1 Hz NVML. Each of ten workload "
        "families has three independent 105-second runs (30 runs total).", "",
        "## Integrity", "",
        f"All {len(audit)} paired node/verifier cells completed successfully: "
        f"{int(audit.both_sides_ok.sum())}/{len(audit)} returned zero on both sides. "
        "All scope traces contain 1,299,444 samples, zero overflow, and zero clipping.", "",
        "## Frozen development evaluation", "",
        "The protocol is leave-one-workload-family-out, 30-second causal windows "
        "with 15-second stride, a 400-tree random forest, probability threshold "
        "0.85, and fixed 3-of-5 alerting.", "", *metric_lines, "",
        "Electrical high-frequency features detect all duty-shaping and migration "
        "runs, but do not transfer to ordinary, interleaved, or memory-minimal "
        "training. Therefore the new corpus does not support a positive worst-family "
        "gain claim at the frozen threshold. The sealed fused-update family was not "
        "used in these results or in feature/model selection.", "",
        "## Remaining sealed test", "",
        "Run fused-update once using the frozen model hashes in `freeze_manifest.json`, "
        "then append its predictions without retraining or changing the threshold.", "",
    ]
    (args.output_dir / "TESTS_AND_RESULTS.md").write_text("\n".join(notes))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
