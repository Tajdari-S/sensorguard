#!/usr/bin/env python3
"""Build the paper motivation figure from committed measured evidence.

The four panels intentionally retain their evaluation scopes.  In particular,
the physical-sensor pilot is not presented as a paired replacement for the
NVML held-out-family audit; that matched transfer test is still pending.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

TRAINING = "#C73E1D"
INFERENCE = "#2E6F9E"
SENSOR = "#237A57"
NEUTRAL = "#555555"
LIGHT = "#B8B8B8"
WAVE = "#7B4EA3"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def roofline_panel(ax: plt.Axes, evidence: list[dict[str, object]]) -> None:
    path = (
        RESULTS
        / "roofline"
        / "applications"
        / "validated-bb6b232"
        / "rtx3090_application"
        / "application-roofline-points.csv"
    )
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    if len(rows) != 15 or len(grouped) != 5 or any(len(value) != 3 for value in grouped.values()):
        raise ValueError("roofline panel requires five cases with three repetitions each")

    cases = []
    for case_id, repetitions in grouped.items():
        cases.append(
            {
                "case_id": case_id,
                "role": "Training" if "train" in case_id else "Inference",
                "ai": median(float(row["normalized_arithmetic_intensity"]) for row in repetitions),
                "throughput": median(float(row["normalized_wall_throughput"]) for row in repetitions),
            }
        )

    metrics = [("Normalized throughput", "throughput", 0.0), ("Normalized intensity", "ai", 1.0)]
    role_style = {
        "Training": {"color": TRAINING, "marker": "^", "offset": 0.085},
        "Inference": {"color": INFERENCE, "marker": "o", "offset": -0.085},
    }
    for metric_label, key, y_base in metrics:
        for role, style in role_style.items():
            values = [float(case[key]) for case in cases if case["role"] == role]
            y = y_base + float(style["offset"])
            ax.plot([min(values), max(values)], [y, y], color=style["color"], linewidth=2.1, zorder=1)
            ax.scatter(
                values,
                [y] * len(values),
                color=style["color"],
                marker=style["marker"],
                s=34,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            evidence.append(
                {
                    "panel": "roofline_role_overlap",
                    "scope": "RTX 3090 application characterization; median of 3 repetitions",
                    "metric": key,
                    "method": role,
                    "value": median(values),
                    "value_low": min(values),
                    "value_high": max(values),
                    "unit": "fraction_of_measured_ridge_or_peak",
                }
            )
        ax.text(
            0.012,
            y_base + 0.25,
            "training range is inside inference range",
            color=NEUTRAL,
            fontsize=7.0,
            ha="left",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.008, 1.9)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([0, 1], ["Throughput / peak", "Intensity / ridge"])
    ax.set_xlabel("Normalized roofline coordinate")
    ax.set_title("(a) Roofline position is not a role label", loc="left")
    ax.grid(axis="x", which="both", color="#E5E5E5", linewidth=0.6)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=TRAINING,
                   markeredgecolor="white", markersize=6, label="Training"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=INFERENCE,
                   markeredgecolor="white", markersize=6, label="Inference"),
        ],
        frameon=False,
        ncol=2,
        loc="center right",
        handletextpad=0.3,
        columnspacing=0.8,
    )


def nvml_panel(ax: plt.Axes, evidence: list[dict[str, object]]) -> None:
    rows = read_csv(RESULTS / "tables" / "generalization-audit.csv")
    selected_names = ["Run grouped", "Held-out RTX 3090", "Held-out family"]
    selected = {row["protocol"]: row for row in rows if row["protocol"] in selected_names}
    if set(selected) != set(selected_names):
        raise ValueError("NVML panel is missing a required generalization audit")

    x = np.arange(3)
    values = np.array([100.0 * float(selected[name]["training_tpr"]) for name in selected_names])
    lower = np.array([100.0 * float(selected[name]["tpr_ci_95_low"]) for name in selected_names])
    upper = np.array([100.0 * float(selected[name]["tpr_ci_95_high"]) for name in selected_names])
    colors = [NEUTRAL, INFERENCE, TRAINING]
    for index, (value, color) in enumerate(zip(values, colors)):
        ax.errorbar(
            index,
            value,
            yerr=[[value - lower[index]], [upper[index] - value]],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )
        ax.scatter(index, value, s=44, color=color, edgecolor="white", linewidth=0.7, zorder=2)
        row = selected[selected_names[index]]
        total_training = int(row["tp"]) + int(row["fn"])
        total_negative = int(row["fp"]) + int(row["tn"])
        detected_y = 16 if value < 20 else value - 8
        ax.text(index, detected_y, f"{int(row['tp'])}/{total_training} detected",
                ha="center", va="bottom" if value < 20 else "top", fontsize=7.2, color=NEUTRAL)
        ax.text(index, 4, f"FP {int(row['fp'])}/{total_negative}", ha="center", va="bottom",
                fontsize=6.9, color=NEUTRAL)
        evidence.append(
            {
                "panel": "nvml_generalization",
                "scope": "NVML-only two-stage random forest; run-level 3-of-5 rule",
                "metric": "training_tpr",
                "method": selected_names[index],
                "value": value,
                "value_low": lower[index],
                "value_high": upper[index],
                "unit": "percent",
                "false_positives": int(row["fp"]),
                "negative_runs": total_negative,
            }
        )

    ax.set_ylim(-2, 108)
    ax.set_xticks(x, ["Run\ngrouped", "Held-out\nGPU", "Held-out\nfamily"])
    ax.set_ylabel("Training detection rate (%)")
    ax.set_title("(b) NVML fails on unseen families", loc="left")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)


def sensor_panel(ax: plt.Axes, evidence: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in read_csv(RESULTS / "tables" / "physical-sensor-ablation.csv")
        if int(row["window_sec"]) == 30
    ]
    order = [
        "UltraMic",
        "Motherboard clamp",
        "GPU current clamp",
        "Electrical clamps",
        "Electrical + acoustic",
    ]
    selected = {row["modality"]: row for row in rows}
    if set(order) - set(selected):
        raise ValueError("physical-sensor panel is missing a required modality")
    labels = ["UltraMic", "Motherboard", "GPU current", "Both clamps", "All physical"]
    values = [100.0 * float(selected[name]["f1_macro"]) for name in order]
    colors = [LIGHT, LIGHT, SENSOR, SENSOR, SENSOR]
    y = np.arange(len(order))
    ax.barh(y, values, color=colors, height=0.62)
    for index, value in enumerate(values):
        ax.text(value - 1.4 if value > 70 else value + 1.2, index, f"{value:.1f}",
                ha="right" if value > 70 else "left", va="center",
                color="white" if value > 70 else NEUTRAL, fontsize=7.2)
        evidence.append(
            {
                "panel": "physical_sensor_screening",
                "scope": "Separate 36-run base campaign; 30 s windows; grouped by run",
                "metric": "macro_f1",
                "method": order[index],
                "value": value,
                "unit": "percent",
            }
        )
    ax.set_xlim(0, 104)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Window macro-F1 (%)")
    ax.set_title("(c) GPU current is the retained sensor", loc="left")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)


def overhead_panel(ax: plt.Axes, evidence: list[dict[str, object]]) -> None:
    sensor_result = json.loads((RESULTS / "sensor_overhead_verifier.json").read_text())
    sensor_values = {row["sensor"]: float(row["useful_work_penalty_pct"]) for row in sensor_result["sensors"]}
    wave_rows = {row["metric_set"]: row for row in read_csv(RESULTS / "tables" / "wave-overhead-summary.csv")}
    items = [
        ("Our NVML logger", sensor_values["nvml"], 0.0, 0.0, SENSOR),
        ("Our DCGM logger", sensor_values["dcgm"], 0.0, 0.0, SENSOR),
        (
            "WAVE: 1 NCU metric",
            float(wave_rows["One NCU metric"]["mean_overhead_pct"]),
            float(wave_rows["One NCU metric"]["minimum_overhead_pct"]),
            float(wave_rows["One NCU metric"]["maximum_overhead_pct"]),
            WAVE,
        ),
        (
            "WAVE: full metric set",
            float(wave_rows["WAVE metric set"]["mean_overhead_pct"]),
            float(wave_rows["WAVE metric set"]["minimum_overhead_pct"]),
            float(wave_rows["WAVE metric set"]["maximum_overhead_pct"]),
            WAVE,
        ),
    ]
    y = np.arange(len(items))
    for index, (label, value, low, high, color) in enumerate(items):
        if high > low:
            ax.plot([low, high], [index, index], color=color, linewidth=2.0, zorder=1)
        ax.scatter(value, index, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=2)
        label_x = value + (0.35 if value == 0 else value * 0.10)
        ax.text(label_x, index, f"{value:.1f}%", ha="left", va="center", fontsize=7.2, color=NEUTRAL)
        evidence.append(
            {
                "panel": "monitor_overhead",
                "scope": "RTX 3090 measurements; our logger result is a 90 s GEMM check",
                "metric": "useful_work_penalty",
                "method": label,
                "value": value,
                "value_low": low,
                "value_high": high,
                "unit": "percent",
            }
        )
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.set_xlim(-0.2, 4300)
    ax.set_xticks([0, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
    ax.set_yticks(y, [item[0] for item in items])
    ax.invert_yaxis()
    ax.set_xlabel("Measured useful-work penalty (%) - symlog")
    ax.set_title("(d) Low-rate logging avoids NCU overhead", loc="left")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)


def write_evidence_csv(rows: list[dict[str, object]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    columns = [
        "panel",
        "scope",
        "metric",
        "method",
        "value",
        "value_low",
        "value_high",
        "unit",
        "false_positives",
        "negative_runs",
    ]
    with (TABLES / "sensor-motivation-evidence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    configure()
    evidence: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    roofline_panel(axes[0, 0], evidence)
    nvml_panel(axes[0, 1], evidence)
    sensor_panel(axes[1, 0], evidence)
    overhead_panel(axes[1, 1], evidence)
    fig.suptitle("Why SensorGuard adds external sensing to NVML", fontsize=11.2, y=0.995)
    fig.text(
        0.5,
        0.006,
        "Measured RTX 3090 evidence. Physical-sensor screening is a separate pilot; matched held-out-family fusion remains pending.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=NEUTRAL,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=1.8, w_pad=1.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        output = FIGURES / f"sensor-motivation-evidence.{suffix}"
        fig.savefig(output, bbox_inches="tight", pad_inches=0.05, dpi=260 if suffix == "png" else None)
        print(f"Wrote {output}")
    plt.close(fig)
    write_evidence_csv(evidence)
    print(f"Wrote {TABLES / 'sensor-motivation-evidence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
