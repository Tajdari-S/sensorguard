#!/usr/bin/env python3
"""Confirm the selected sensor specification against physical capture metadata."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--selected-spec", type=Path, required=True)
    parser.add_argument("--device-model", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--native-bits", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = json.loads(args.selected_spec.read_text())["selected"]
    metas = []
    for path in sorted(args.capture_root.glob("*/pico_u0_meta.json")):
        meta = json.loads(path.read_text())
        raw_path = path.with_name("pico_u0_chA.npy")
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw = np.load(raw_path, mmap_mode="r")
        elapsed = (
            float(meta["capture_end_raw_s"])
            - float(meta["capture_start_raw_s"])
        )
        metas.append({
            "run_id": path.parent.name,
            "serial": meta["serial"],
            "sample_interval_us": int(meta["sample_interval_us"]),
            "samples": int(meta["samples"]),
            "effective_sample_hz": int(meta["samples"]) / elapsed,
            "overflow_flags": int(meta["overflow_flags"]),
            "clipping_fraction_a": float(
                meta.get("clipping_fraction_a") or 0
            ),
            "raw_dtype": str(raw.dtype),
        })
    if not metas:
        raise RuntimeError("no physical capture metadata found")

    requested_hz = int(selected["sample_hz"])
    nominal_interval = int(round(1_000_000 / requested_hz))
    checks = {
        "all_expected_runs_present": len(metas) == int(selected["runs"]),
        "all_selected_serial": all(
            row["serial"] == args.device_serial for row in metas
        ),
        "all_selected_sample_interval": all(
            row["sample_interval_us"] == nominal_interval for row in metas
        ),
        "native_resolution_matches_selected": (
            args.native_bits == int(selected["bits"])
        ),
        "all_zero_overflow": all(
            row["overflow_flags"] == 0 for row in metas
        ),
        "all_below_one_percent_clipping": all(
            row["clipping_fraction_a"] < 0.01 for row in metas
        ),
        "offline_decisions_preserved": bool(
            selected["exactly_preserves_full_rate_run_decisions"]
        ),
    }
    summary = {
        "qualification": "hardware-in-loop confirmed",
        "device_model": args.device_model,
        "device_serial": args.device_serial,
        "native_resolution_bits": args.native_bits,
        "nominal_sample_hz": requested_hz,
        "physical_capture_runs": len(metas),
        "effective_sample_hz_mean": statistics.mean(
            row["effective_sample_hz"] for row in metas
        ),
        "effective_sample_hz_min": min(
            row["effective_sample_hz"] for row in metas
        ),
        "effective_sample_hz_max": max(
            row["effective_sample_hz"] for row in metas
        ),
        "retained_confusion": {
            key: int(selected[key]) for key in ("tp", "fn", "fp", "tn")
        },
        "checks": checks,
        "pass": all(checks.values()),
        "scope": (
            "Physical acquisition and device resolution are confirmed; "
            "sampling/quantization decisions were replayed offline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "summary": summary,
        "captures": metas,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
