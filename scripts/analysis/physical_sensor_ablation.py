#!/usr/bin/env python3
"""Reproduce the RTX 3090 physical-sensor proof-of-concept and ablations.

The source logs were committed by Robi Rahman on the private
``physical-sensor-detection`` branch of ``robirahman/GPU-monitoring``.  This
script deliberately evaluates the original base campaign only: later physical
red-team and multi-GPU follow-up names are excluded so that the combined row is
directly comparable to the committed 37-run result.

The protocol mirrors the source implementation: causal 15/30/60 s windows,
approximately 50% overlap, 400-tree random forest, and five-fold
StratifiedGroupKFold grouped by run ID.  Reported values are window-level mean
accuracy and macro F1 across folds; they are a proof-of-concept, not a held-out
GPU or held-out-workload-family result.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


TRAINING_LABELS = {
    "pytorch_resnet_cifar10",
    "pytorch_resnet_cifar10_amp",
    "pytorch_mlp_cifar10",
    "gpt2_wikitext2",
    "gpt2_wikitext2_amp",
    "bert_sst2",
    "bert_sst2_amp",
    "adversarial_H_mimicry_cufft",
    "adversarial_J_pid_75",
    "adversarial_K_online_learning",
    "adversarial_L_diluted_5",
    "adversarial_L_diluted_10",
    "adversarial_composite_50",
}

# Follow-up attacks and hardening runs were added after the committed base
# classifier result and require a separate, adversarially labelled protocol.
FOLLOWUP_PREFIXES = (
    "adversarial_physical_stealth",
    "dil_v",
    "ev2b_",
    "evasion2_",
    "s2_",
    "s3_",
)

MODALITIES = {
    "GPU current clamp": {
        "channels": ["GPU0_rms_mv", "GPU0_peak_mv"],
        "temporal": ["GPU0_rms_mv"],
    },
    "Motherboard clamp": {
        "channels": ["MOBO_rms_mv"],
        "temporal": [],
    },
    "UltraMic": {
        "channels": ["mic_rms", "mic_dbfs", "mic_peak", "mic_dominant_hz"],
        "temporal": ["mic_rms"],
    },
    "Electrical clamps": {
        "channels": ["GPU0_rms_mv", "GPU0_peak_mv", "MOBO_rms_mv"],
        "temporal": ["GPU0_rms_mv"],
    },
    "Electrical + acoustic": {
        "channels": [
            "GPU0_rms_mv",
            "GPU0_peak_mv",
            "MOBO_rms_mv",
            "mic_rms",
            "mic_dbfs",
            "mic_peak",
            "mic_dominant_hz",
        ],
        "temporal": ["GPU0_rms_mv", "mic_rms"],
    },
}

COLORS = {
    "GPU current clamp": "#C73E1D",
    "Motherboard clamp": "#B5741A",
    "UltraMic": "#2E6F9E",
    "Electrical clamps": "#7B4EA3",
    "Electrical + acoustic": "#237A57",
}


def acf(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) <= lag or np.std(x) < 1e-9:
        return 0.0
    x = x - x.mean()
    denominator = np.dot(x, x)
    return float(np.dot(x[:-lag], x[lag:]) / denominator) if denominator > 1e-12 else 0.0


def fft_peak(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) < 4 or np.std(x) < 1e-9:
        return 0.0, 0.0
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    if len(power) < 2:
        return 0.0, 0.0
    index = 1 + int(np.argmax(power[1:]))
    return float(len(x) / index), float(power[index] / (power[1:].sum() + 1e-12))


def stats(values: np.ndarray, name: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    suffixes = ["mean", "std", "min", "max", "p25", "p50", "p75", "p95", "iqr", "range", "cv", "skew", "kurt"]
    if len(x) == 0:
        return {f"{name}_{suffix}": 0.0 for suffix in suffixes}
    mean = float(x.mean())
    std = float(x.std())
    p25, p50, p75, p95 = np.percentile(x, [25, 50, 75, 95])
    return {
        f"{name}_mean": mean,
        f"{name}_std": std,
        f"{name}_min": float(x.min()),
        f"{name}_max": float(x.max()),
        f"{name}_p25": float(p25),
        f"{name}_p50": float(p50),
        f"{name}_p75": float(p75),
        f"{name}_p95": float(p95),
        f"{name}_iqr": float(p75 - p25),
        f"{name}_range": float(x.max() - x.min()),
        f"{name}_cv": std / mean if abs(mean) > 1e-9 else 0.0,
        f"{name}_skew": float(skew(x)) if std > 1e-9 else 0.0,
        f"{name}_kurt": float(kurtosis(x)) if std > 1e-9 else 0.0,
    }


def load_base_campaign(sensor_dir: Path) -> pd.DataFrame:
    frames = []
    for filename in sorted(glob.glob(os.path.join(sensor_dir, "*_sensors.parquet"))):
        frame = pd.read_parquet(filename)
        label = str(frame["workload_label"].iloc[0])
        if label.startswith(FOLLOWUP_PREFIXES):
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No base-campaign sensor logs found in {sensor_dir}")
    data = pd.concat(frames, ignore_index=True)
    data["target"] = data["workload_label"].isin(TRAINING_LABELS).astype(int)
    return data


def make_windows(
    data: pd.DataFrame,
    channels: list[str],
    temporal_channels: list[str],
    window_sec: int,
    stride_sec: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, run in data.groupby("run_id"):
        run = run.sort_values("timestamp_epoch")
        times = run["timestamp_epoch"].to_numpy()
        start = float(times[0])
        end = float(times[-1])
        while start + window_sec <= end + 1e-6:
            mask = (times >= start) & (times < start + window_sec)
            if int(mask.sum()) >= 3:
                chunk = run.loc[mask]
                features: dict[str, object] = {}
                for channel in channels:
                    features.update(stats(chunk[channel].to_numpy(), channel))
                for channel in temporal_channels:
                    values = chunk[channel].to_numpy()
                    features[f"{channel}_acf1"] = acf(values, 1)
                    features[f"{channel}_acf5"] = acf(values, 5)
                    period, peak = fft_peak(values)
                    features[f"{channel}_fft_period"] = period
                    features[f"{channel}_fft_peak"] = peak
                features["run_id"] = run_id
                features["workload_label"] = str(run["workload_label"].iloc[0])
                features["target"] = int(run["target"].iloc[0])
                rows.append(features)
            start += stride_sec
    return pd.DataFrame(rows)


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality, specification in MODALITIES.items():
        for window_sec, stride_sec in ((15, 7), (30, 15), (60, 30)):
            windows = make_windows(
                data,
                specification["channels"],
                specification["temporal"],
                window_sec,
                stride_sec,
            )
            feature_columns = [
                column for column in windows.columns
                if column not in {"run_id", "workload_label", "target"}
            ]
            x = windows[feature_columns].fillna(0).to_numpy(dtype=np.float32)
            y = windows["target"].to_numpy(dtype=int)
            groups = windows["run_id"].to_numpy()
            run_classes = windows.groupby("run_id")["target"].first()
            n_splits = min(5, int(run_classes.value_counts().min()))
            cross_validation = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )
            accuracies = []
            f1_scores = []
            for train, test in cross_validation.split(x, y, groups):
                model = RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                accuracies.append(accuracy_score(y[test], predictions))
                f1_scores.append(f1_score(y[test], predictions, average="macro", zero_division=0))
            rows.append({
                "dataset": "RTX 3090 base physical-sensor campaign",
                "modality": modality,
                "window_sec": window_sec,
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "f1_macro": float(np.mean(f1_scores)),
                "n_windows": len(windows),
                "n_runs": int(windows["run_id"].nunique()),
                "n_features": len(feature_columns),
                "evaluation_unit": "window",
                "split": "5-fold StratifiedGroupKFold by run_id",
            })
    return pd.DataFrame(rows)


def write_table(results: pd.DataFrame, output: Path) -> None:
    thirty = results[results["window_sec"] == 30].copy()
    lines = [
        "% Generated by scripts/analysis/physical_sensor_ablation.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Sensor set & Accuracy & Macro F1 & Runs & Features \\\\",
        "\\midrule",
    ]
    for row in thirty.itertuples():
        label = row.modality.replace("&", "\\&")
        lines.append(
            f"{label} & {100 * row.accuracy_mean:.1f}\\% & {row.f1_macro:.3f} & {row.n_runs} & {row.n_features} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output.write_text("\n".join(lines))


def plot_ablation(results: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"width_ratios": [1.35, 1]})
    # Only single sensors are drawn as lines.  The electrical and full-fusion
    # curves are numerically identical to the GPU-clamp curve and previously
    # covered it completely.  All five configurations remain visible as bars.
    for modality in ("GPU current clamp", "Motherboard clamp", "UltraMic"):
        subset = results[results["modality"] == modality].sort_values("window_sec")
        axes[0].errorbar(
            subset["window_sec"],
            100 * subset["accuracy_mean"],
            yerr=100 * subset["accuracy_std"],
            marker="o",
            linewidth=1.5,
            capsize=2.5,
            color=COLORS[modality],
            label=modality,
        )
    axes[0].set_xticks([15, 30, 60])
    axes[0].set_xlim(12, 78)
    axes[0].set_ylim(45, 102)
    axes[0].set_xlabel("Causal window (s)")
    axes[0].set_ylabel("Window accuracy (%)")
    axes[0].set_title("Individual sensors across windows")
    axes[0].grid(axis="y", alpha=0.25)
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    end_labels = {
        "GPU current clamp": (100.0, "GPU clamp"),
        "Motherboard clamp": (66.4, "MOBO clamp"),
        "UltraMic": (55.1, "UltraMic"),
    }
    for modality, (value, label) in end_labels.items():
        axes[0].text(61.8, value, label, color=COLORS[modality], va="center", ha="left", fontsize=7.5)

    thirty = results[results["window_sec"] == 30].copy()
    x = np.arange(len(thirty))
    bars = axes[1].bar(
        x,
        100 * thirty["f1_macro"],
        color=[COLORS[name] for name in thirty["modality"]],
        width=0.72,
    )
    axes[1].set_xticks(x, [
        "GPU\nclamp",
        "MOBO\nclamp",
        "UltraMic",
        "Both\nclamps",
        "All\nphysical",
    ])
    axes[1].set_ylim(45, 102)
    axes[1].set_ylabel("Macro F1 (%)")
    axes[1].set_title("Single sensors and combinations (30 s)")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, 100 * thirty["f1_macro"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("RTX 3090 physical-sensor proof-of-concept (grouped by run; base campaign)", y=1.02, fontsize=10)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"physical-sensor-ablation.{suffix}", bbox_inches="tight", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def plot_signatures(data: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for run_id, run in data.groupby("run_id"):
        values = run["GPU0_rms_mv"].to_numpy(dtype=float)
        _, peak = fft_peak(values)
        rows.append({
            "run_id": run_id,
            "label": "Training" if int(run["target"].iloc[0]) else "Other",
            "workload": str(run["workload_label"].iloc[0]),
            "rms_mean": float(np.mean(values)),
            "rms_cv": float(np.std(values) / np.mean(values)) if abs(np.mean(values)) > 1e-9 else 0.0,
            "fft_peak": peak,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir.parent / "tables" / "physical-run-signatures.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    palette = {"Training": "#C73E1D", "Other": "#2E6F9E"}
    for label in ("Other", "Training"):
        subset = summary[summary["label"] == label]
        axes[0].scatter(subset["rms_mean"], subset["rms_cv"], s=28, alpha=0.8, color=palette[label], label=label, edgecolor="white", linewidth=0.4)
        axes[1].scatter(subset["rms_cv"], subset["fft_peak"], s=28, alpha=0.8, color=palette[label], label=label, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("GPU0 clamp RMS mean (mV)")
    axes[0].set_ylabel("RMS coefficient of variation")
    axes[0].set_title("Magnitude alone overlaps")
    axes[1].set_xlabel("RMS coefficient of variation")
    axes[1].set_ylabel("Normalized FFT peak")
    axes[1].set_title("Burstiness and periodicity separate runs")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Physical current signatures across base-campaign runs", y=1.02, fontsize=10)
    fig.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"physical-current-signatures.{suffix}", bbox_inches="tight", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = load_base_campaign(args.sensor_dir)
    results = evaluate(data)
    tables = args.results_dir / "tables"
    figures = args.results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables / "physical-sensor-ablation.csv", index=False)
    write_table(results, tables / "physical-sensor-ablation.tex")
    plot_ablation(results, figures)
    plot_signatures(data, figures)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
