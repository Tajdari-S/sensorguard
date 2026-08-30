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


def collect_evidence() -> list[dict[str, object]]:
    generalization = {
        row["protocol"]: row
        for row in read_csv(RESULTS / "tables" / "generalization-audit.csv")
    }
    family = generalization["Held-out family"]

    wave = {
        row["metric_set"]: row
        for row in read_csv(RESULTS / "tables" / "wave-overhead-summary.csv")
    }["WAVE metric set"]

    logger = json.loads((RESULTS / "sensor_overhead_verifier.json").read_text())
    penalties = {
        row["sensor"]: float(row["useful_work_penalty_pct"])
        for row in logger["sensors"]
    }
    if penalties != {"nvml": 0.0, "dcgm": 0.0}:
        raise ValueError("the motivation figure expects the committed 0.0% logger check")

    physical = {
        row["modality"]: row
        for row in read_csv(RESULTS / "tables" / "physical-sensor-ablation.csv")
        if int(row["window_sec"]) == 30
    }
    gpu_current_f1 = 100.0 * float(physical["GPU current clamp"]["f1_macro"])

    labels = read_csv(RESULTS / "e2_labels_combined.csv")
    current_scope = {
        label: sorted({row["family"] for row in labels if row["label"] == label})
        for label in ("training", "inference", "non_ml")
    }
    if current_scope["training"] != ["train_bert", "train_gpt2_wikitext", "train_resnet_cifar10"]:
        raise ValueError("unexpected current training-target scope")

    return [
        {
            "method": "Prior NVML",
            "monitoring_role": "continuous hidden-training detection",
            "overhead_mean_pct": 0.0,
            "overhead_low_pct": 0.0,
            "overhead_high_pct": 0.0,
            "application_scope": "162 workloads: 106 training, 40 inference, 16 other",
            "target_applications": "broad published corpus",
            "evidence": (
                f"current held-out-family audit: {family['tp']}/{int(family['tp']) + int(family['fn'])} "
                f"training detected; {family['fp']}/{int(family['fp']) + int(family['tn'])} false positives"
            ),
            "status": "measured baseline",
        },
        {
            "method": "WAVE",
            "monitoring_role": "offline architectural verification",
            "overhead_mean_pct": float(wave["mean_overhead_pct"]),
            "overhead_low_pct": float(wave["minimum_overhead_pct"]),
            "overhead_high_pct": float(wave["maximum_overhead_pct"]),
            "application_scope": "3 decoder families; 6 overhead configurations",
            "target_applications": "GPT-2; LLaMA; Qwen",
            "evidence": "verifies decoder architecture; not a training detector",
            "status": "measured RTX 3090 reproduction",
        },
        {
            "method": "SensorGuard",
            "monitoring_role": "continuous NVML plus independent physical evidence",
            "overhead_mean_pct": 0.0,
            "overhead_low_pct": 0.0,
            "overhead_high_pct": 0.0,
            "application_scope": (
                f"{len(current_scope['training'])} current training targets; "
                f"{len(current_scope['inference'])} inference and {len(current_scope['non_ml'])} control families"
            ),
            "target_applications": "ResNet-50/CIFAR-10; GPT-2/WikiText; BERT; MLP in physical pilot",
            "evidence": f"GPU-current pilot macro-F1 {gpu_current_f1:.1f}%; matched transfer and physical overhead pending",
            "status": "partial current-paper evidence",
        },
        {
            "method": "Roofline overlap",
            "monitoring_role": "offline characterization, not a detector",
            "overhead_mean_pct": "",
            "overhead_low_pct": "",
            "overhead_high_pct": "",
            "application_scope": "prior sweep: 286 non-training configurations",
            "target_applications": "101 configurations inside the training arithmetic-intensity range",
            "evidence": "application role remains ambiguous in roofline space",
            "status": "motivation evidence",
        },
    ]


def write_evidence(rows: list[dict[str, object]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "sensor-motivation-evidence.csv"
    columns = [
        "method",
        "monitoring_role",
        "overhead_mean_pct",
        "overhead_low_pct",
        "overhead_high_pct",
        "application_scope",
        "target_applications",
        "evidence",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )


def make_figure(rows: list[dict[str, object]]) -> None:
    by_method = {row["method"]: row for row in rows}
    wave = by_method["WAVE"]

    fig = plt.figure(figsize=(7.15, 3.85))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.08)
    scope_ax = fig.add_subplot(grid[0, 0])
    overhead_ax = fig.add_subplot(grid[0, 1], sharey=scope_ax)

    for axis in (scope_ax, overhead_ax):
        axis.set_ylim(0.45, 3.55)
    scope_ax.set_xlim(0, 1)
    scope_ax.axis("off")
    for y in (1.5, 2.5):
        scope_ax.axhline(y, color=GRID, linewidth=0.7, xmin=0.0, xmax=0.98)

    method_rows = [
        (3, "Prior NVML", NVML,
         "162 workloads (106 training / 40 inference / 16 other)",
         "Unseen family: 0/23 detected; 10/95 false positives"),
        (2, "WAVE", WAVE,
         "3 decoder families: GPT-2, LLaMA, Qwen",
         "Architecture verification only; not training detection"),
        (1, "SensorGuard", OURS,
         "Targets: ResNet-50, GPT-2, BERT (+ MLP in the physical pilot)",
         "Controls: 4 inference + 8 non-ML; fused update pending"),
    ]
    for y, method, color, scope, limitation in method_rows:
        scope_ax.text(0.01, y + 0.20, method, color=color, fontsize=9.5,
                      fontweight="bold", ha="left", va="center")
        scope_ax.text(0.26, y + 0.20, scope, color=color, fontsize=7.8,
                      ha="left", va="center")
        scope_ax.text(0.26, y - 0.20, limitation, color=MUTED, fontsize=7.3,
                      ha="left", va="center")

    scope_ax.text(0.01, 3.48, "Method and measured application scope", color=NVML,
                  fontsize=8.4, ha="left", va="bottom")

    overhead_ax.set_xscale("symlog", linthresh=1.0, linscale=0.72)
    overhead_ax.set_xlim(-0.25, 4300)
    overhead_ax.set_xticks([0, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
    overhead_ax.set_yticks([])
    overhead_ax.set_xlabel("Workload penalty (%) - symlog")
    overhead_ax.grid(axis="x", color=GRID, linewidth=0.65)
    overhead_ax.axvspan(-0.2, 1.0, color=OURS, alpha=0.06, linewidth=0)
    overhead_ax.axvline(1.0, color="#999999", linestyle=(0, (3, 3)), linewidth=0.9)
    overhead_ax.text(0.02, 3.48, "Measured monitoring overhead", transform=overhead_ax.get_yaxis_transform(),
                     color=NVML, fontsize=8.4, ha="left", va="bottom")
    overhead_ax.text(0.95, 3.33, "continuous budget <=1%", color=MUTED,
                     fontsize=6.8, ha="right", va="center")

    overhead_ax.scatter(0, 3, s=62, color=NVML, marker="o", edgecolor="white",
                        linewidth=0.8, zorder=3)
    overhead_ax.scatter(0, 1, s=70, color=OURS, marker="D", edgecolor="white",
                        linewidth=0.8, zorder=3)
    overhead_ax.plot(
        [float(wave["overhead_low_pct"]), float(wave["overhead_high_pct"])],
        [2, 2],
        color=WAVE,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=2,
    )
    overhead_ax.scatter(float(wave["overhead_mean_pct"]), 2, s=70, color=WAVE,
                        marker="s", edgecolor="white", linewidth=0.8, zorder=3)

    overhead_ax.text(0.28, 3, "0.0% in our logger check", color=NVML, fontsize=7.3,
                     ha="left", va="center")
    overhead_ax.text(0.28, 1.10, "0.0% base logger*", color=OURS, fontsize=7.5,
                     ha="left", va="center")
    overhead_ax.text(0.28, 0.84, "*physical logger pending", color=MUTED, fontsize=6.7,
                     ha="left", va="center")
    overhead_ax.text(
        650,
        2.20,
        f"{float(wave['overhead_mean_pct']):,.0f}% mean",
        color=WAVE,
        fontsize=7.4,
        ha="left",
        va="center",
    )
    overhead_ax.text(
        650,
        1.82,
        f"range {float(wave['overhead_low_pct']):,.0f}-{float(wave['overhead_high_pct']):,.0f}%",
        color=MUTED,
        fontsize=6.8,
        ha="left",
        va="center",
    )

    fig.suptitle("Motivation: broad hidden-training detection needs independent low-cost evidence",
                 fontsize=10.7, y=0.985)
    fig.text(
        0.5, 0.045,
        "Roofline ambiguity: 101/286 non-training configurations fell inside the training arithmetic-intensity range - characterization is not detection.",
        ha="center", va="bottom", fontsize=7.1, color=MUTED,
    )
    fig.text(
        0.5, 0.015,
        "SensorGuard's end-to-end physical-logger overhead and matched held-out-family result remain pending.",
        ha="center", va="bottom", fontsize=6.8, color=MUTED,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.88, bottom=0.20)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = FIGURES / f"sensor-motivation-evidence.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05,
                    dpi=280 if suffix == "png" else None)
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> int:
    configure()
    rows = collect_evidence()
    write_evidence(rows)
    make_figure(rows)
    print(f"Wrote {TABLES / 'sensor-motivation-evidence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
