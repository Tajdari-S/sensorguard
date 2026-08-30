#!/usr/bin/env python3
"""Score every PicoScope channel against one-at-a-time GPU load markers."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def robust_rms(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    centered = values.astype(np.float64) - np.median(values)
    return float(np.sqrt(np.mean(centered * centered)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pico-prefix", type=Path, required=True)
    parser.add_argument("--marker-json", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--planned-pulse", action="append", default=[],
        help="label,gpu_uuid,start_epoch_s,cycles,on_s,off_s; use only after marker logs verify schedule",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-s", type=float, default=1.0)
    args = parser.parse_args()

    markers = [json.loads(path.read_text()) for path in args.marker_json]
    for spec in args.planned_pulse:
        label, gpu_uuid, start_text, cycles_text, on_text, off_text = spec.split(",")
        start = float(start_text)
        cycles = int(cycles_text)
        on_s = float(on_text)
        off_s = float(off_text)
        active = []
        inactive = []
        for cycle in range(cycles):
            on_start = start + cycle * (on_s + off_s)
            active.append([on_start, on_start + on_s])
            inactive.append([on_start + on_s, on_start + on_s + off_s])
        markers.append({
            "label": label,
            "gpu_uuid": gpu_uuid,
            "actual_start_epoch_s": start,
            "actual_end_epoch_s": start + cycles * (on_s + off_s),
            "active_intervals_epoch_s": active,
            "inactive_intervals_epoch_s": inactive,
        })
    if not markers:
        parser.error("provide --marker-json or --planned-pulse")
    channels = []
    for meta_path in sorted(args.pico_prefix.parent.glob(args.pico_prefix.name + "_u*_meta.json")):
        meta = json.loads(meta_path.read_text())
        unit_token = meta_path.name.split("_meta.json")[0]
        start_s = float(meta["capture_start_epoch_ns"]) / 1e9
        end_s = float(meta["capture_end_epoch_ns"]) / 1e9
        for channel in ("A", "B"):
            raw_path = meta_path.with_name(unit_token + f"_ch{channel}.npy")
            raw = np.load(raw_path, mmap_mode="r")
            times = np.linspace(start_s, end_s, raw.size, endpoint=False)
            channels.append((meta["serial"], channel, raw, times))

    rows = []
    for marker in markers:
        burst_active = marker.get("active_intervals_epoch_s") or []
        burst_inactive = marker.get("inactive_intervals_epoch_s") or []
        if burst_active:
            active_windows = [
                (float(start) + args.guard_s, float(end) - args.guard_s)
                for start, end in burst_active
            ]
            baseline_windows = [
                (float(start) + args.guard_s, float(end) - args.guard_s)
                for start, end in burst_inactive
            ]
        else:
            active_start = float(marker["actual_start_epoch_s"]) + max(3.0, args.guard_s)
            active_end = float(marker["actual_end_epoch_s"]) - max(3.0, args.guard_s)
            active_windows = [(active_start, active_end)]
            baseline_windows = [(active_start - 18.0, active_start - 5.0)]
        for serial, channel, raw, times in channels:
            active_mask = np.zeros(times.shape, dtype=bool)
            baseline_mask = np.zeros(times.shape, dtype=bool)
            for start, end in active_windows:
                active_mask |= (times >= start) & (times <= end)
            for start, end in baseline_windows:
                baseline_mask |= (times >= start) & (times <= end)
            active = np.asarray(raw[active_mask])
            baseline = np.asarray(raw[baseline_mask])
            active_rms = robust_rms(active)
            baseline_rms = robust_rms(baseline)
            ratio = active_rms / baseline_rms if baseline_rms > 0 else float("nan")
            baseline_mean = float(np.mean(baseline))
            active_mean = float(np.mean(active))
            mean_shift = abs(active_mean - baseline_mean)
            mean_shift_sigma = mean_shift / baseline_rms if baseline_rms > 0 else float("nan")
            response_score = mean_shift_sigma + abs(math.log(ratio))
            rows.append({
                "gpu_label": marker["label"],
                "gpu_uuid": marker["gpu_uuid"],
                "scope_serial": serial,
                "scope_channel": channel,
                "baseline_rms_adc": baseline_rms,
                "active_rms_adc": active_rms,
                "active_to_baseline_ratio": ratio,
                "baseline_mean_adc": baseline_mean,
                "active_mean_adc": active_mean,
                "absolute_mean_shift_adc": mean_shift,
                "mean_shift_baseline_sigma": mean_shift_sigma,
                "response_score": response_score,
                "active_start_epoch_s": active_windows[0][0],
                "active_end_epoch_s": active_windows[-1][1],
                "active_intervals": len(active_windows),
            })

    for gpu_label in {row["gpu_label"] for row in rows}:
        selected = [row for row in rows if row["gpu_label"] == gpu_label]
        selected.sort(key=lambda row: row["response_score"], reverse=True)
        for rank, row in enumerate(selected, start=1):
            row["rank_for_gpu"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["gpu_label"], row["rank_for_gpu"])))

    for gpu_label in sorted({row["gpu_label"] for row in rows}):
        best = min(
            (row for row in rows if row["gpu_label"] == gpu_label),
            key=lambda row: row["rank_for_gpu"],
        )
        print(
            f"{gpu_label}: {best['scope_serial']} ch{best['scope_channel']} "
            f"score={best['response_score']:.3f} "
            f"rms_ratio={best['active_to_baseline_ratio']:.3f} "
            f"mean_shift_sigma={best['mean_shift_baseline_sigma']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
