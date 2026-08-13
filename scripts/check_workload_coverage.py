#!/usr/bin/env python3
"""Validate the public SensorGuard workload coverage contract using the stdlib only."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"id", "category", "application", "variants", "source", "priority", "status", "fit_3090"}
ALLOWED_STATUS = {"runnable", "adapter_needed", "external_hardware", "blocked_exact_name"}
PROJECT_CRITICAL = {
    "attack_b_low_util", "attack_j_pid", "attack_l_diluted", "attack_whitebox_full",
    "attack_whitebox_lora", "eval_fused_update", "eval_latency",
    *(f"custom_kernel_variant_{i}" for i in range(1, 6)),
    "train_ppo", "data_etl", "database_acceleration", "mixed_ml_hpc", "jax_xla",
    "infer_multi_query", "infer_multi_user_serving", "eval_amd_cross_vendor",
    "eval_virtualization_mig", "eval_signal_robustness",
    "eval_adaptive_surrogate", "eval_telemetry_integrity",
}
REQUIRED_ATTACKS = {
    "attack_a_util_modulation", "attack_b_low_util", "attack_d_temporal_disruption",
    "attack_e_memory_minimal", "attack_f_interleave", "attack_g_clock_throttled",
    "attack_h_mimicry", "attack_i_stochastic", "attack_j_pid", "attack_k_online_learning",
    "attack_l_diluted", "attack_m_composite_memmin", "attack_n_grad_accum",
    "attack_o_composite_idle_pad", "attack_k_ddp", "attack_k_ddp_accum",
    "attack_l_ddp", "attack_l_ddp_stagger", "attack_whitebox_full", "attack_whitebox_lora",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="configs/workloads.json")
    parser.add_argument("--table", action="store_true", help="print compact coverage table")
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text())
    rows = data.get("workloads", [])
    errors = []
    ids = [row.get("id") for row in rows]
    duplicates = sorted(k for k, n in Counter(ids).items() if k and n > 1)
    if duplicates:
        errors.append(f"duplicate ids: {', '.join(duplicates)}")
    for index, row in enumerate(rows, 1):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"row {index} missing: {', '.join(sorted(missing))}")
        if row.get("status") not in ALLOWED_STATUS:
            errors.append(f"{row.get('id', index)} invalid status: {row.get('status')}")
        if not isinstance(row.get("variants"), list) or not row.get("variants"):
            errors.append(f"{row.get('id', index)} needs at least one variant")
    present = set(ids)
    for label, required in (("public-paper attack", REQUIRED_ATTACKS), ("project-critical", PROJECT_CRITICAL)):
        missing = sorted(required - present)
        if missing:
            errors.append(f"missing {label} ids: {', '.join(missing)}")
    if args.table:
        print("id\tpriority\tstatus\tapplication")
        for row in rows:
            print(f"{row['id']}\t{row['priority']}\t{row['status']}\t{row['application']}")
    counts = Counter(row["category"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    print(f"Validated {len(rows)} workload/application records")
    print("Categories:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("Statuses:", ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    if errors:
        print("ERROR:", *errors, sep="\n- ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
