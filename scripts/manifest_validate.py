#!/usr/bin/env python3
"""Validate a filled run manifest produced by scripts/loggers/supervisor.py."""

import sys
from pathlib import Path

import yaml

REQUIRED_TOP = ["run_id", "git_commit", "start_utc", "end_utc", "workload",
                "sensors", "sensor_channels", "profiled", "artifact_checksums", "status"]
REQUIRED_CHANNEL = ["channel_id", "sample_rate_hz", "clock_source", "health"]


def main() -> int:
    errors = []
    for arg in sys.argv[1:] or ["-"]:
        path = Path(arg)
        doc = yaml.safe_load(path.read_text())
        for key in REQUIRED_TOP:
            if key not in doc:
                errors.append(f"{path}: missing key {key}")
        for i, ch in enumerate(doc.get("sensor_channels", [])):
            for key in REQUIRED_CHANNEL:
                if key not in ch:
                    errors.append(f"{path}: channel[{i}] missing {key}")
            if ch.get("health") not in ("pass", "fail"):
                errors.append(f"{path}: channel[{i}] health must be pass|fail")
        if not doc.get("artifact_checksums"):
            errors.append(f"{path}: empty artifact_checksums")
        active = [s for s, on in doc.get("sensors", {}).items() if on]
        listed = {c["channel_id"].split(".")[0] for c in doc.get("sensor_channels", [])}
        for s in active:
            if s not in listed:
                errors.append(f"{path}: sensor '{s}' active but has no channel entry")
    if errors:
        print("\n".join(errors))
        return 1
    print("Manifest(s) valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
