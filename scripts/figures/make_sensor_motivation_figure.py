#!/usr/bin/env python3
"""Build one motivation figure comparing prior monitoring with SensorGuard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

NVML = "#4D4D4D"
WAVE = "#7B4EA3"
OURS = "#237A57"
MUTED = "#666666"
GRID = "#E2E2E2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect_evidence() -> dict[str, dict[str, object]]:
    matched = {
        row["method"]: row
        for row in read_csv(RESULTS / "tables" / "matched-36-family-heldout-confusion.csv")
    }
    expected_methods = {"NVML", "SensorGuard (GPU current)"}
    if set(matched) != expected_methods:
        raise ValueError("matched comparison must contain NVML and SensorGuard")
    for row in matched.values():
        if row["total_runs"] != "36" or row["training_runs"] != "26" or row["nontraining_runs"] != "10":
            raise ValueError("matched comparison must use the same 36-run cohort")

    wave_rows = read_csv(RESULTS / "wave" / "overhead_3090_all.csv")
    sensor_rows = read_csv(RESULTS / "wave" / "matched_sensor_overhead_3090.csv")
    wave_by_id = {row["model_id"]: row for row in wave_rows}
    sensor_by_id = {row["model_id"]: row for row in sensor_rows}
    if len(wave_by_id) != 6 or wave_by_id.keys() != sensor_by_id.keys():
        raise ValueError("WAVE and SensorGuard overhead rows must contain the same six models")

    manifest = json.loads(
        (RESULTS / "wave" / "matched_sensor_overhead_3090.json").read_text()
    )
    if manifest["summary"]["configuration_count"] != 6 or manifest["repetitions"] != 3:
        raise ValueError("matched SensorGuard overhead manifest is incomplete")
    if manifest["gpu"]["uuid"] != "GPU-392b25f7-d685-7d9d-ee07-864670a4e2e9":
        raise ValueError("matched overhead was not collected on the WAVE RTX 3090")
    for config in manifest["configurations"]:
        if any(
            health["nvml_ok_rows"] < 2 or health["dcgm_data_rows"] < 2
            for health in config["trace_health"]
        ):
            raise ValueError("matched overhead contains an unhealthy logger trace")

    wave_multipliers = [
        1.0 + float(wave_by_id[model]["overhead_percent"]) / 100.0
        for model in sorted(wave_by_id)
    ]
    sensor_multipliers = [
        float(sensor_by_id[model]["runtime_multiplier"])
        for model in sorted(sensor_by_id)
    ]

    return {
        "nvml_matched": {
            "panel": "A",
            "method": "NVML",
            "metric": "run-level confusion count",
            "value": matched["NVML"]["correct_runs"],
            "low": "",
            "high": "",
            "unit": "runs",
            "tp": matched["NVML"]["true_positive_detected_training"],
            "fn": matched["NVML"]["false_negative_missed_training"],
            "fp": matched["NVML"]["false_positive_false_alert"],
            "tn": matched["NVML"]["true_negative_correct_rejection"],
            "scope": "same 36 runs: 26 training and 10 non-training",
            "status": "leave-one-workload-family-out; fixed 3-of-5 rule at 0.75",
        },
        "sensor_matched": {
            "panel": "A",
            "method": "SensorGuard (GPU current)",
            "metric": "run-level confusion count",
            "value": matched["SensorGuard (GPU current)"]["correct_runs"],
            "low": "",
            "high": "",
            "unit": "runs",
            "tp": matched["SensorGuard (GPU current)"]["true_positive_detected_training"],
            "fn": matched["SensorGuard (GPU current)"]["false_negative_missed_training"],
            "fp": matched["SensorGuard (GPU current)"]["false_positive_false_alert"],
            "tn": matched["SensorGuard (GPU current)"]["true_negative_correct_rejection"],
            "scope": "same 36 runs: 26 training and 10 non-training",
            "status": "leave-one-workload-family-out; fixed 3-of-5 rule at 0.75",
        },
        "wave": {
            "panel": "B",
            "method": "WAVE",
            "metric": "whole-process runtime multiplier",
            "value": sum(wave_multipliers) / len(wave_multipliers),
            "low": min(wave_multipliers),
            "high": max(wave_multipliers),
            "unit": "x",
            "tp": "",
            "fn": "",
            "fp": "",
            "tn": "",
            "scope": "same 6 configurations: 2 GPT-2, 2 LLaMA, 2 Qwen; 3 repetitions",
            "status": "offline architectural verification",
        },
        "sensorguard": {
            "panel": "B",
            "method": "SensorGuard base logger",
            "metric": "whole-process runtime multiplier",
            "value": sum(sensor_multipliers) / len(sensor_multipliers),
            "low": min(sensor_multipliers),
            "high": max(sensor_multipliers),
            "unit": "x",
            "tp": "",
            "fn": "",
            "fp": "",
            "tn": "",
            "scope": "same 6 configurations: 2 GPT-2, 2 LLaMA, 2 Qwen; 3 repetitions; NVML+DCGM",
            "status": "base logger only; physical logger overhead pending",
        },
    }


def write_evidence(evidence: dict[str, dict[str, object]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "sensor-motivation-evidence.csv"
    columns = [
        "panel", "method", "metric", "value", "low", "high", "unit",
        "tp", "fn", "fp", "tn", "scope", "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence.values())


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.65,
        }
    )


def make_figure(evidence: dict[str, dict[str, object]]) -> None:
    nvml_matched = evidence["nvml_matched"]
    sensor_matched = evidence["sensor_matched"]
    wave = evidence["wave"]
    sensor = evidence["sensorguard"]

    fig, (scope_ax, overhead_ax) = plt.subplots(
        1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.26}
    )

    # A: direct run-level outcomes on the exact same paired cohort.
    outcomes = ["TP", "FN", "FP", "TN"]
    nvml_counts = [int(nvml_matched[key.lower()]) for key in outcomes]
    sensor_counts = [int(sensor_matched[key.lower()]) for key in outcomes]
    x_positions = np.arange(len(outcomes))
    width = 0.34
    nvml_bars = scope_ax.bar(
        x_positions - width / 2, nvml_counts, width=width, color=NVML,
        edgecolor="#444444", linewidth=0.45, label="NVML",
    )
    sensor_bars = scope_ax.bar(
        x_positions + width / 2, sensor_counts, width=width, color="#78E6C9",
        edgecolor="#444444", linewidth=0.45, label="SensorGuard (GPU current)",
    )
    scope_ax.set_ylabel("Runs (count)")
    scope_ax.set_xticks(
        x_positions,
        ["Detected\ntraining (TP)", "Missed\ntraining (FN)",
         "False\nalerts (FP)", "Correct\nrejection (TN)"],
    )
    scope_ax.tick_params(axis="x", labelsize=6.8, pad=3)
    scope_ax.set_ylim(0, 31)
    scope_ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    scope_ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)
    scope_ax.set_axisbelow(True)
    scope_ax.set_title("A", pad=3)
    scope_ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    for bars, values in ((nvml_bars, nvml_counts), (sensor_bars, sensor_counts)):
        for bar, value in zip(bars, values):
            scope_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.55 if value else 0.35),
                str(value), ha="center", va="bottom", fontsize=7.4,
                fontweight="bold",
            )
    scope_ax.text(
        0.5, -0.27,
        "Same 36 runs; leave-one-family-out; fixed 3-of-5 rule at p >= 0.75",
        transform=scope_ax.transAxes, ha="center", va="top", fontsize=6.0, color=MUTED,
    )

    # B: both methods use the exact same six model configurations and whole-
    # process wall-time definition on the same power-capped RTX 3090.
    overhead_values = [float(wave["value"]), float(sensor["value"])]
    overhead_low = [float(wave["low"]), float(sensor["low"])]
    overhead_high = [float(wave["high"]), float(sensor["high"])]
    overhead_bars = overhead_ax.bar(
        [0, 1], overhead_values, width=0.68,
        color=["#F4EADF", "#78E6C9"], edgecolor="#555555", linewidth=0.45,
        yerr=[
            [value - low for value, low in zip(overhead_values, overhead_low)],
            [high - value for value, high in zip(overhead_values, overhead_high)],
        ],
        capsize=3, error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#555555"},
    )
    overhead_ax.set_ylabel("Runtime multiplier (x)")
    overhead_ax.set_xticks(
        [0, 1],
        ["WAVE\nGPT-2 / LLaMA / Qwen", "SensorGuard*\nGPT-2 / LLaMA / Qwen"],
    )
    overhead_ax.tick_params(axis="x", labelsize=7.8, pad=4)
    overhead_ax.set_ylim(0, 36)
    overhead_ax.set_yticks([0, 10, 20, 30])
    overhead_ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)
    overhead_ax.set_axisbelow(True)
    overhead_ax.set_title("B", pad=3)
    overhead_ax.text(
        overhead_bars[0].get_x() + overhead_bars[0].get_width() / 2,
        overhead_high[0] + 0.6,
        f"{overhead_values[0]:.1f}x",
        ha="center", va="bottom", fontsize=10.5,
    )
    overhead_ax.text(
        overhead_bars[1].get_x() + overhead_bars[1].get_width() / 2,
        overhead_values[1] + 1.0,
        f"{overhead_values[1]:.2f}x",
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
    )
    overhead_ax.text(
        0.98, 0.58,
        "*NVML+DCGM base logger only",
        transform=overhead_ax.transAxes, ha="right", va="center",
        fontsize=6.2, color=MUTED,
    )

    fig.subplots_adjust(left=0.095, right=0.99, top=0.92, bottom=0.31)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = FIGURES / f"sensor-motivation-evidence.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04,
                    dpi=280 if suffix == "png" else None)
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> int:
    configure()
    evidence = collect_evidence()
    write_evidence(evidence)
    make_figure(evidence)
    print(f"Wrote {TABLES / 'sensor-motivation-evidence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
