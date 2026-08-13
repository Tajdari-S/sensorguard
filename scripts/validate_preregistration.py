#!/usr/bin/env python3
from pathlib import Path
import sys

REQUIRED_KEYS = [
    "project:", "freeze_date:", "deadline:", "primary_endpoint:",
    "false_alert_budget_per_gpu_hour:", "split_unit:", "group_fields:",
    "sensor_retention:", "required_holdouts:", "test_open_count:"
]

path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/preregistration.yaml")
text = path.read_text(encoding="utf-8")
missing = [key for key in REQUIRED_KEYS if key not in text]
unresolved = [line.strip() for line in text.splitlines() if "TBD" in line]

if missing or unresolved:
    if missing:
        print("Missing required keys:", ", ".join(missing))
    if unresolved:
        print("Unresolved values:")
        for line in unresolved:
            print(" -", line)
    raise SystemExit(1)

print(f"Preregistration structure validated: {path}")

