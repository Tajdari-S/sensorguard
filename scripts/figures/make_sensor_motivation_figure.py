#!/usr/bin/env python3
"""Build one motivation figure comparing prior monitoring with SensorGuard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


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
    generalization = {
        row["protocol"]: row
        for row in read_csv(RESULTS / "tables" / "generalization-audit.csv")
    }
    run_grouped = generalization["Run grouped"]
    family = generalization["Held-out family"]

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
        "run_grouped": {
            "panel": "A",
            "method": "NVML run-grouped",
            "metric": "training detection rate",
            "value": 100.0 * float(run_grouped["training_tpr"]),
            "low": 100.0 * float(run_grouped["tpr_ci_95_low"]),
            "high": 100.0 * float(run_grouped["tpr_ci_95_high"]),
            "unit": "percent",
            "scope": "same NVML-only RF; families represented during training",
        },
        "held_out_family": {
            "panel": "A",
            "method": "NVML held-out family",
            "metric": "training detection rate",
            "value": 100.0 * float(family["training_tpr"]),
            "low": 100.0 * float(family["tpr_ci_95_low"]),
            "high": 100.0 * float(family["tpr_ci_95_high"]),
            "unit": "percent",
            "scope": f"same NVML-only RF; {family['tp']}/{int(family['tp']) + int(family['fn'])} training runs detected",
        },
        "wave": {
            "panel": "B",
            "method": "WAVE",
            "metric": "whole-process runtime multiplier",
            "value": sum(wave_multipliers) / len(wave_multipliers),
            "low": min(wave_multipliers),
            "high": max(wave_multipliers),
            "unit": "x",
            "scope": "same 6 configurations: 2 GPT-2, 2 LLaMA, 2 Qwen; 3 repetitions",
        },
        "sensorguard": {
            "panel": "B",
            "method": "SensorGuard base logger",
            "metric": "whole-process runtime multiplier",
            "value": sum(sensor_multipliers) / len(sensor_multipliers),
            "low": min(sensor_multipliers),
            "high": max(sensor_multipliers),
            "unit": "x",
            "scope": "same 6 configurations: 2 GPT-2, 2 LLaMA, 2 Qwen; 3 repetitions; NVML+DCGM",
        },
    }


def write_evidence(evidence: dict[str, dict[str, object]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "sensor-motivation-evidence.csv"
    columns = ["panel", "method", "metric", "value", "low", "high", "unit", "scope"]
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
    run_grouped = evidence["run_grouped"]
    held_out = evidence["held_out_family"]
    wave = evidence["wave"]
    sensor = evidence["sensorguard"]

    fig, (roofline_ax, overhead_ax) = plt.subplots(
        1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.26}
    )

    # A: one detector and decision rule, evaluated with and without family
    # overlap between fitting and evaluation.
    detection_values = [float(run_grouped["value"]), float(held_out["value"])]
    detection_low = [float(run_grouped["low"]), float(held_out["low"])]
    detection_high = [float(run_grouped["high"]), float(held_out["high"])]
    detection_bars = roofline_ax.bar(
        [0, 1], detection_values, width=0.68,
        color=[WAVE, "#AFC4DA"], edgecolor="#555555", linewidth=0.45,
        yerr=[
            [value - low for value, low in zip(detection_values, detection_low)],
            [high - value for value, high in zip(detection_values, detection_high)],
        ],
        capsize=3, error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#555555"},
    )
    roofline_ax.set_ylabel("Training detection rate (%)")
    roofline_ax.set_xticks(
        [0, 1],
        ["Run-grouped\n(families represented)", "Held-out family\n(0/23 runs detected)"],
    )
    roofline_ax.set_ylim(0, 115)
    roofline_ax.set_yticks([0, 25, 50, 75, 100])
    roofline_ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)
    roofline_ax.set_axisbelow(True)
    roofline_ax.set_title("A", pad=3)
    for bar, value, high in zip(detection_bars, detection_values, detection_high):
        roofline_ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 2.0,
            f"{value:.1f}%",
            ha="center", va="bottom", fontsize=10.5,
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
        "*NVML+DCGM base logger only\nphysical logger pending",
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
