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


def make_figure(rows: list[dict[str, object]]) -> None:
    by_method = {row["method"]: row for row in rows}
    wave = by_method["WAVE"]

    roofline_total = 286
    roofline_overlap = 101
    roofline_separated = roofline_total - roofline_overlap
    roofline_pct = [
        100.0 * roofline_overlap / roofline_total,
        100.0 * roofline_separated / roofline_total,
    ]

    wave_multiplier = 1.0 + float(wave["overhead_mean_pct"]) / 100.0
    wave_low = 1.0 + float(wave["overhead_low_pct"]) / 100.0
    wave_high = 1.0 + float(wave["overhead_high_pct"]) / 100.0
    logger_multiplier = 1.0

    fig, (roofline_ax, overhead_ax) = plt.subplots(
        1, 2, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.26}
    )

    # A: Roofline overlap. A substantial fraction of non-training work occupies
    # the same arithmetic-intensity range as training, so location is ambiguous.
    roofline_bars = roofline_ax.bar(
        [0, 1], roofline_pct, width=0.68,
        color=[WAVE, "#AFC4DA"], edgecolor="#555555", linewidth=0.45,
    )
    roofline_ax.set_ylabel("Non-training configs (%)")
    roofline_ax.set_xticks(
        [0, 1],
        [f"Inside training range\n({roofline_overlap} cases)",
         f"Outside training range\n({roofline_separated} cases)"],
    )
    roofline_ax.set_ylim(0, 78)
    roofline_ax.set_yticks([0, 20, 40, 60])
    roofline_ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)
    roofline_ax.set_axisbelow(True)
    roofline_ax.set_title("A", pad=3)
    for bar, value in zip(roofline_bars, roofline_pct):
        roofline_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center", va="bottom", fontsize=10.5,
        )
    roofline_ax.text(
        0.5, 0.96,
        "NVML unseen family: 0/23 training runs detected",
        transform=roofline_ax.transAxes, ha="center", va="top",
        fontsize=7.2, color=MUTED,
    )

    # B: WAVE is a high-overhead architectural verifier. The SensorGuard value
    # is the measured base logger only; the physical logger is not yet measured.
    overhead_bars = overhead_ax.bar(
        [0, 1], [wave_multiplier, logger_multiplier], width=0.68,
        color=["#F4EADF", "#78E6C9"], edgecolor="#555555", linewidth=0.45,
        yerr=[[wave_multiplier - wave_low, 0.0], [wave_high - wave_multiplier, 0.0]],
        capsize=3, error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#555555"},
    )
    overhead_ax.set_ylabel("Runtime multiplier (x)")
    overhead_ax.set_xticks(
        [0, 1],
        ["WAVE\nGPT-2 / LLaMA\nQwen (3 families)",
         "SensorGuard\nResNet-50 / GPT-2\nBERT (3 targets)"],
    )
    overhead_ax.tick_params(axis="x", labelsize=7.6, pad=4)
    overhead_ax.set_ylim(0, 36)
    overhead_ax.set_yticks([0, 10, 20, 30])
    overhead_ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)
    overhead_ax.set_axisbelow(True)
    overhead_ax.set_title("B", pad=3)
    overhead_ax.text(
        overhead_bars[0].get_x() + overhead_bars[0].get_width() / 2,
        wave_high + 0.6,
        f"{wave_multiplier:.1f}x",
        ha="center", va="bottom", fontsize=10.5,
    )
    overhead_ax.text(
        overhead_bars[1].get_x() + overhead_bars[1].get_width() / 2,
        logger_multiplier + 1.0,
        f"{logger_multiplier:.1f}x*",
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
    )
    overhead_ax.text(
        0.99, 0.96,
        "*base logger; physical logger pending",
        transform=overhead_ax.transAxes, ha="right", va="top",
        fontsize=6.5, color=MUTED,
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
    rows = collect_evidence()
    write_evidence(rows)
    make_figure(rows)
    print(f"Wrote {TABLES / 'sensor-motivation-evidence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
