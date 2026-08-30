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

    nvml_labels = read_csv(RESULTS / "e2_labels_combined.csv")
    nvml_families = {
        label: {row["family"] for row in nvml_labels if row["label"] == label}
        for label in ("training", "inference", "non_ml")
    }
    expected_training = {
        "train_bert", "train_gpt2_wikitext", "train_resnet_cifar10"
    }
    if nvml_families["training"] != expected_training:
        raise ValueError("unexpected synchronized NVML training scope")

    physical_runs = read_csv(RESULTS / "tables" / "physical-run-signatures.csv")
    physical_training = {
        row["workload"] for row in physical_runs if row["label"] == "Training"
    }
    physical_controls = {
        row["workload"] for row in physical_runs if row["label"] == "Other"
    }
    common_physical = {
        "pytorch_resnet_cifar10", "gpt2_wikitext2", "bert_sst2"
    }
    if not common_physical.issubset(physical_training):
        raise ValueError("physical pilot is missing a common training target")
    physical_result = next(
        row for row in read_csv(RESULTS / "tables" / "physical-sensor-ablation.csv")
        if row["modality"] == "GPU current clamp" and row["window_sec"] == "30"
    )

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
        "nvml_scope": {
            "panel": "A",
            "method": "NVML synchronized corpus",
            "metric": "evaluated workload classes/variants",
            "value": sum(len(families) for families in nvml_families.values()),
            "low": "",
            "high": "",
            "unit": "count",
            "common_core": len(nvml_families["training"]),
            "additional_training": 0,
            "controls": len(nvml_families["inference"] | nvml_families["non_ml"]),
            "scope": (
                f"{len(nvml_labels)} runs; 3 training, "
                f"{len(nvml_families['inference'])} inference, "
                f"{len(nvml_families['non_ml'])} non-ML families"
            ),
            "status": (
                f"run-grouped {run_grouped['tp']}/{int(run_grouped['tp']) + int(run_grouped['fn'])} "
                f"training runs detected; held-out family {family['tp']}/"
                f"{int(family['tp']) + int(family['fn'])}"
            ),
        },
        "sensor_scope": {
            "panel": "A",
            "method": "SensorGuard current-clamp pilot",
            "metric": "evaluated workload classes/variants",
            "value": len(physical_training | physical_controls),
            "low": "",
            "high": "",
            "unit": "count",
            "common_core": len(common_physical),
            "additional_training": len(physical_training - common_physical),
            "controls": len(physical_controls),
            "scope": (
                f"{len(physical_runs)} runs; {len(physical_training)} training and "
                f"{len(physical_controls)} non-training workload variants"
            ),
            "status": (
                f"30-s run-grouped pilot macro-F1 {float(physical_result['f1_macro']):.2f}; "
                f"{physical_result['n_runs']} eligible runs; held-out-family sensor test pending"
            ),
        },
        "wave": {
            "panel": "B",
            "method": "WAVE",
            "metric": "whole-process runtime multiplier",
            "value": sum(wave_multipliers) / len(wave_multipliers),
            "low": min(wave_multipliers),
            "high": max(wave_multipliers),
            "unit": "x",
            "common_core": "",
            "additional_training": "",
            "controls": "",
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
            "common_core": "",
            "additional_training": "",
            "controls": "",
            "scope": "same 6 configurations: 2 GPT-2, 2 LLaMA, 2 Qwen; 3 repetitions; NVML+DCGM",
            "status": "base logger only; physical logger overhead pending",
        },
    }


def write_evidence(evidence: dict[str, dict[str, object]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "sensor-motivation-evidence.csv"
    columns = [
        "panel", "method", "metric", "value", "low", "high", "unit",
        "common_core", "additional_training", "controls", "scope", "status",
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
    nvml_scope = evidence["nvml_scope"]
    sensor_scope = evidence["sensor_scope"]
    wave = evidence["wave"]
    sensor = evidence["sensorguard"]

    fig, (scope_ax, overhead_ax) = plt.subplots(
        1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.26}
    )

    # A: breadth of the two current measured corpora. The shared purple segment
    # is the common ResNet-50/GPT-2/BERT core; it does not claim matched sensor
    # recovery of the NVML misses.
    scope_rows = [nvml_scope, sensor_scope]
    common = [int(row["common_core"]) for row in scope_rows]
    additional = [int(row["additional_training"]) for row in scope_rows]
    controls = [int(row["controls"]) for row in scope_rows]
    totals = [int(row["value"]) for row in scope_rows]
    x_positions = [0, 1]
    common_bars = scope_ax.bar(
        x_positions, common, width=0.68, color=WAVE,
        edgecolor="#555555", linewidth=0.45, label="Common training core",
    )
    additional_bars = scope_ax.bar(
        x_positions, additional, width=0.68, bottom=common, color="#D98C45",
        edgecolor="#555555", linewidth=0.45, label="Additional training variants",
    )
    control_bottom = [base + extra for base, extra in zip(common, additional)]
    control_bars = scope_ax.bar(
        x_positions, controls, width=0.68, bottom=control_bottom, color="#AFC4DA",
        edgecolor="#555555", linewidth=0.45, label="Non-training controls",
    )
    scope_ax.set_ylabel("Evaluated workloads (count)")
    scope_ax.set_xticks(
        [0, 1],
        ["NVML corpus\n118 runs", "SensorGuard pilot\n40 runs*"],
    )
    scope_ax.set_ylim(0, 34)
    scope_ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    scope_ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)
    scope_ax.set_axisbelow(True)
    scope_ax.set_title("A", pad=3)
    scope_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=1,
        frameon=False, fontsize=6.4, handlelength=1.2, labelspacing=0.25,
    )
    annotations = [
        "15 families\n22/23 seen\n0/23 held out",
        "19 variants\ncurrent-clamp F1=1.00\nheld-out pending",
    ]
    for x, total, annotation in zip(x_positions, totals, annotations):
        scope_ax.text(
            x, total + 0.6, annotation,
            ha="center", va="bottom", fontsize=7.0,
        )
    for bars, values in (
        (common_bars, common), (additional_bars, additional), (control_bars, controls)
    ):
        for bar, value in zip(bars, values):
            if value:
                scope_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(value), ha="center", va="center", fontsize=7.4,
                    color="white" if bars is common_bars else NVML,
                    fontweight="bold",
                )
    scope_ax.text(
        0.5, -0.27,
        "Common core: ResNet-50, GPT-2, BERT   |   *30-s pilot uses 36 eligible runs",
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
