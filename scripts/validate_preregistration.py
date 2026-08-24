#!/usr/bin/env python3
import hashlib
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

# An `unresolved:` block with any list entries means preregistered values are
# still unset even though no line contains the literal string "TBD".
lines = text.splitlines()
for i, line in enumerate(lines):
    if line.strip() == "unresolved:" or line.strip().startswith("unresolved:"):
        for follow in lines[i + 1:]:
            stripped = follow.strip()
            if stripped.startswith("- "):
                unresolved.append(f"unresolved: {stripped[2:]}")
            elif stripped and not stripped.startswith("#"):
                break

if missing or unresolved:
    if missing:
        print("Missing required keys:", ", ".join(missing))
    if unresolved:
        print("Unresolved values:")
        for line in unresolved:
            print(" -", line)
    raise SystemExit(1)

digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
print(f"Preregistration structure validated: {path}")
print(f"sha256: {digest}")
print("Record this hash in docs/DECISION_LOG.md at freeze.")
