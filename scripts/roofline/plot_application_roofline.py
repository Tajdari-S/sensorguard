#!/usr/bin/env python3
"""Plot the measured RTX 3090 application roofline from repeated runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


LABELS = {
    "resnet50-train-b64": "ResNet-50 train\n(batch 64)",
    "resnet50-infer-b64": "ResNet-50 inference\n(batch 64)",
    "gpt2-train-b2-s256": "GPT-2 train\n(batch 2, seq. 256)",
    "gpt2-prefill-b2-s256": "GPT-2 prefill\n(batch 2, seq. 256)",
    "gpt2-decode-b2-s256-n32": "GPT-2 decode\n(batch 2, 32 tokens)",
}

FAMILY_COLORS = {
    "GPT-2": "#0072B2",
    "ResNet-50": "#D55E00",
}

LABEL_OFFSETS = {
    "resnet50-train-b64": (10, 12),
    "resnet50-infer-b64": (10, 10),
    "gpt2-train-b2-s256": (10, -29),
    "gpt2-prefill-b2-s256": (-105, -32),
    "gpt2-decode-b2-s256-n32": (10, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("points", type=Path, help="Application roofline points CSV")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def family(case_id: str) -> str:
    return "GPT-2" if case_id.startswith("gpt2-") else "ResNet-50"


def role(case_id: str) -> str:
    return "Training" if "train" in case_id else "Inference"


def read_points(path: Path) -> tuple[dict[str, list[dict[str, float]]], float, float, str]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    peaks: set[tuple[float, float]] = set()
    platforms: set[str] = set()
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {
                "ai": float(raw["arithmetic_intensity"]),
                "performance": float(raw["wall_tflops"]),
            }
            if not all(math.isfinite(value) and value > 0 for value in row.values()):
                raise ValueError(f"non-positive or non-finite point in {raw['case_id']}")
            grouped[raw["case_id"]].append(row)
            peaks.add((float(raw["peak_tflops"]), float(raw["peak_gbps"])))
            platforms.add(raw["platform"])
    if len(peaks) != 1 or len(platforms) != 1:
        raise ValueError("the figure requires one platform and one measured ceiling pair")
    if set(grouped) != set(LABELS):
        raise ValueError(f"unexpected application cases: {sorted(grouped)}")
    if any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("each application case must have exactly three repetitions")
    peak_tflops, peak_gbps = peaks.pop()
    return grouped, peak_tflops, peak_gbps, platforms.pop()


def main() -> int:
    args = parse_args()
    grouped, peak_tflops, peak_gbps, platform = read_points(args.points)
    ridge = peak_tflops * 1000.0 / peak_gbps

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.55), constrained_layout=True)

    xmin, xmax = 8.0, 180.0
    ymin, ymax = 0.45, 90.0
    roof_x = [xmin * (xmax / xmin) ** (index / 399) for index in range(400)]
    roof_y = [min(peak_tflops, value * peak_gbps / 1000.0) for value in roof_x]
    ax.plot(roof_x, roof_y, color="#333333", linewidth=1.8, zorder=1)
    ax.axvline(ridge, color="#777777", linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)

    for case_id, rows in grouped.items():
        xs = [row["ai"] for row in rows]
        ys = [row["performance"] for row in rows]
        x = median(xs)
        y = median(ys)
        marker = "^" if role(case_id) == "Training" else "o"
        color = FAMILY_COLORS[family(case_id)]
        ax.errorbar(
            x,
            y,
            xerr=[[x - min(xs)], [max(xs) - x]],
            yerr=[[y - min(ys)], [max(ys) - y]],
            fmt=marker,
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
        ax.annotate(
            LABELS[case_id],
            xy=(x, y),
            xytext=LABEL_OFFSETS[case_id],
            textcoords="offset points",
            color="#222222",
            fontsize=8.3,
            linespacing=1.12,
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.7},
            zorder=4,
        )

    ax.text(
        10.0,
        10.0 * peak_gbps / 1000.0 * 0.80,
        f"Measured bandwidth ceiling: {peak_gbps:.0f} GB/s",
        rotation=math.degrees(math.atan(1.0)),
        fontsize=8.2,
        color="#333333",
        ha="left",
        va="top",
    )
    ax.text(
        157,
        peak_tflops * 1.08,
        f"Measured FP16 ceiling: {peak_tflops:.1f} TFLOP/s",
        fontsize=8.2,
        color="#333333",
        ha="right",
        va="bottom",
    )
    ax.text(
        ridge * 1.025,
        peak_tflops * 0.72,
        f"ridge: {ridge:.1f} FLOP/byte",
        fontsize=8,
        color="#555555",
        ha="left",
        va="top",
    )

    family_legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor="white", markersize=7, label=name)
        for name, color in FAMILY_COLORS.items()
    ]
    role_legend = [
        Line2D([0], [0], marker="^", linestyle="none", color="#555555",
               markerfacecolor="#555555", markersize=7, label="Training"),
        Line2D([0], [0], marker="o", linestyle="none", color="#555555",
               markerfacecolor="#555555", markersize=7, label="Inference"),
    ]
    ax.legend(
        handles=family_legend + role_legend,
        ncol=4,
        loc="upper left",
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.05,
        borderaxespad=0.45,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Arithmetic intensity (FLOP/DRAM byte)")
    ax.set_ylabel("Measured wall throughput (TFLOP/s)")
    power = platform.rsplit("-", 1)[-1].upper().replace("W", " W")
    ax.set_title(
        f"RTX 3090 application roofline (FP16, {power})\n"
        "Matched training and inference; median of 3 runs, whiskers show range"
    )
    ax.grid(which="major", color="#D8D8D8", linewidth=0.7)
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.5)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        output = args.output_prefix.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
